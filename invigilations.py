# version 11
# 12/20/2020
# Author: Nick Sebasco

from costmatrix import CostMatrix
from functools import reduce
from random import sample, randint
from calendar import month_abbr
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--init_random_csv', help='Invokes the init_random_csv function.', action='store_true')
parser.add_argument('--affinity', help='LOW, HIGH, or RAND.')
parser.add_argument('--iname', help='Name of initialized csv.')
parser.add_argument('--fname', help='Name of csv file passed to build_from_csv.')
parser.add_argument('--oname', help='Name of csv file created by create_csv.')
parser.add_argument('--algorithm', help='linear_sum_assignment, optimal_selection, or random_assignment.')
parser.add_argument('--depth', type=int, help='How many conflict free assignments to find from batched_linear_assignment_search.')
parser.add_argument('--batch_size', type=int, help='The size of each batch in batched_linear_assignment_search.')
args = parser.parse_args()

def get_workers(fname: str = "data/workers") -> list:
    """Create a list of workers.
    """
    with open(fname + ".csv", "r+") as f:
        return reduce(lambda a, b: a + b,
            [[j.strip() for j in i.split(",") if j.strip() != ""] for i in f.readlines()])[1:]

def get_tasks(fname: str = "data/tasks") -> list:
    """Create a list of tasks.
    """
    with open(fname + ".csv", "r+") as f:
        return [k for k in [[j.strip() for j in i.split(",") if j.strip() != ""] 
            for i in f.readlines()] if k != []]

def init_random_csv(workers: list, tasks: list, fname: str = "data/schedule", DELIM: str = "\t", interval = None, affinity: str = "LOW"):
    """Initialize a random invigilation csv.

    Optional parameters:
    fname <- output csv file name.
    delim <- delimeter used to concatenate complex task names.
    interval <- range of random integers used.
    affinity <- "LOW" - all values will be set as the max value in interval.
                "HIGH" - all values will be set as the min value in interval.
                "RAND" - all values will be set as: a <= random integer <= b | a,b in interval.
    """
    interval = interval or (1, len(tasks))
    with open(f"{fname}.csv", "w+") as f:

        f.write("worker," + ",".join([DELIM.join(i) for i in tasks[1:]]) + "\n")
        for worker in workers:
            if affinity == "LOW":
                f.write(f"{worker}," + ",".join([str(interval[1]) for _ in range(len(tasks) - 1)]) + "\n")
            elif affinity == "HIGH":
                f.write(f"{worker}," + ",".join([str(interval[0]) for _ in range(len(tasks) - 1)]) + "\n")
            elif affinity == "RAND":
                f.write(f"{worker}," + ",".join([str(randint(*interval)) for _ in range(len(tasks) - 1)]) + "\n")
               
    print(f"Created: {fname}.csv.")

# helpers
def convert(assignments: list, DELIM: str = "\t") -> list:
    """Convert assignments from batched_linear_assignment into data format
        to be consumed by create_output_csv.
        output format: [(worker, task), ...]
    """
    converted = []
    for worker in assignments:
        for task in assignments[worker]:
            task = task.split(DELIM)
            task.pop()
            task = DELIM.join(task)
            converted.append((worker, task))
    return converted

def hasConflicts(assignments: list, DELIM: str = "\t") -> bool:
    """Given a list of assignments from batched_linear_assignment
        return a boolean indicating whether or not a time conflict
        exists in the set of assignments.
    """
    CONFLICT = False
    for worker in assignments:
        if len(assignments[worker]) != len(set([tuple(t.split(DELIM)[:2]) for t in assignments[worker]])):
            CONFLICT = True
            break
    return CONFLICT
       
def filter_tasks_by_time(tasks: list, date: str, time: str) -> list:
    """Given a large set of tasks, a date, and a time.  Find all tasks
        in the set that have date == date and time == time.
    """
    return [i[3] for i in list(filter(lambda x: x[0][0] == date and x[0][1] == time ,tasks))[0]]

def filter_workers_by_name(workers: list, name: str) -> list:
    """Given a large set of workers, find all workers with name == name.
    """
    return [i[0] for i in list(filter(lambda x: x[1] == name ,workers))]

def set_max_val(assignments: list, tasksByTime: list, workerList: list, dataList_copy: list, MAX_VAL: float, DELIM = "\t"):
    """Find tasks for each worker with the same time as the time just set.  Set all of
    the scores corresponding to these tasks to some arbitrarily high MAX_VAL.  This 
    will highly incentivize the linear assignment algorithms to not make assignments with
    time conflicts.

    Side effects:
    dataList_copy <- several values in various sublists will be updated to MAX_VAL.
    """
    for worker, task in assignments:
            wn, wi = worker.split(DELIM)
            wi = int(wi)
            td, tt, _, _ = task.split(DELIM)

            for index in filter_tasks_by_time(tasksByTime, td, tt):
                for worker_index in filter_workers_by_name(list(enumerate(workerList)), wn):
                    dataList_copy[worker_index][index] = MAX_VAL

def batched_linear_assignment_search(workers: list, tasks: list, data: list,
    DELIM = "\t",
    DEPTH: int = 5, 
    MAX_COUNT: int = 100, 
    MAX_VAL: float = 1e200,
    BATCH_SIZE: int = 5,
    ALGORITHM: str = "linear_sum_assignment") -> (float, list, int, float):
    """Iteratively performed batched linear assignments in an effort to:
    A) find an assignment without time conflicts.
    B) further reduce cost.

    Optional <hyper-parameters>
    MAX_COUNT <- Max number of iterations.
    DEPTH <- Number of different assignments to find.
    MAX_VAL, BATCH_SIZE, ALGORITHM <- see batched_linear_assignment
    """
    costList, assignmentList, totalcount, depthcount = [], [], 0, 0
    while depthcount < DEPTH and totalcount < MAX_COUNT:
        cost, assignments = batched_linear_assignment_T(workers, tasks, data, MAX_VAL=MAX_VAL, BATCH_SIZE=BATCH_SIZE, ALGORITHM=ALGORITHM, DELIM=DELIM)
        if not hasConflicts(assignments):
            assignmentList.append(assignments)
            costList.append(cost)
            depthcount += 1
        totalcount += 1

    return (costList[np.argmin(costList)], 
            assignmentList[np.argmin(costList)], 
            totalcount, 
            costList[np.argmax(costList)] - costList[np.argmin(costList)])

def batched_linear_assignment_T(workerList: list, 
    tasksByTime: list, 
    dataList: list,
    DELIM: str = "\t",  
    MAX_VAL: float = 1e200, 
    BATCH_SIZE: int = 5,
    ALGORITHM: str = "linear_sum_assignment") -> (float, list):
    """Recursively minimizes batches of cost matrices in order to assign tasks to workers.
    Optional 
    DELIM <- charcter used to separate important data in string: ex.  'dateDELIMtime'
    <hyper-parameters>
    ALGORITHM <- specify the assignment algorithm: "linear_sum_assignment"; "optimal_selection"; "random_assignment"
    MAX_VAL <- This value should be large.  It is used to heavily deincentivize multiple assignments at 
    the same time.  
    BATCH_SIZE <- Size of each cost Matrix.
    """
    workerList_copy = list(enumerate(deepcopy(workerList)))
    tasksByTime_copy = deepcopy(tasksByTime)
    dataList_copy = deepcopy(dataList)

    batch_number = 1

    reserved_tasks = {} # used by "random_selection" algorithm
    master_assignments = {}
    master_cost = 0

    while len(tasksByTime_copy) > 0:
        # If the number of tasks is not divisible by the batch_size 
        # we will have remainder tasks that need to be assigned.
        k = BATCH_SIZE if len(tasksByTime_copy) >= BATCH_SIZE else len(tasksByTime_copy)
        task_batch_indices = sample(range(len(tasksByTime_copy)), k=k)
        worker_batch_indices = sample(range(len(workerList_copy)), k=k)
        worker_batch = [workerList_copy[int(i)] for i in worker_batch_indices]
        task_batch = [tasksByTime_copy[i].pop() for i in task_batch_indices]
        data_batch = np.matrix([[dataList_copy[i][j] for _, __, ___, j in task_batch] for i, _ in worker_batch],dtype=float)

        costmatrix = CostMatrix(G=data_batch,
            N=k,
            workerNames=[i[1] + DELIM + str(i[0]) for i in worker_batch],
            taskNames=[DELIM.join(list(str(i) for i in j)) for j in task_batch])

        cost, assignments = None, None
        if ALGORITHM == "linear_sum_assignment":
            cost, assignments = costmatrix.scipy_RLAP()
            set_max_val(assignments, tasksByTime, workerList, dataList_copy, MAX_VAL, DELIM=DELIM)
        elif ALGORITHM == "optimal_selection":
            assignments, cost, _ = costmatrix.get_optimal_assignment(LOG=False)
            assignments = [tup[::-1] for tup in assignments] # reverse the assignment ordering
            set_max_val(assignments, tasksByTime, workerList, dataList_copy, MAX_VAL, DELIM=DELIM)
        elif ALGORITHM == "random_assignment":
            workerNames = [i[1] + DELIM + str(i[0]) for i in worker_batch]
            taskNames = [DELIM.join(list(str(i) for i in j)) for j in task_batch]
            cost = 0
            assignments = []
            
            it, max_it = 0, 1e6
            while len(taskNames) > 0:
                tsk = taskNames[-1]
                td, tt, _, _ = tsk.split(DELIM)
                for wrkr in workerNames:
                    wn, _ = wrkr.split(DELIM)
                    if wn in reserved_tasks:
                        if (td, tt) in reserved_tasks[wn]:
                            continue
                        else:
                            reserved_tasks[wn].add((td, tt))
                            assignments.append((wrkr, tsk))
                            del taskNames[-1]
                            break
                    else:
                        reserved_tasks[wn] = set()
                        reserved_tasks[wn].add((td, tt))
                        assignments.append((wrkr, tsk))
                        del taskNames[-1]
                        break
                if it >= max_it:
                    print("ERROR: Iteration limit exceeded.")
                    break
                it += 1
            
            for worker, task in assignments:
                wi = int(worker.split(DELIM)[1])
                ti = int(task.split(DELIM)[3])
                cost += dataList_copy[wi][ti]
        else:
            raise(Exception("Invalid Algorithm."))

        # update master assignments
        for worker, task in assignments:
            name, _ = worker.split(DELIM)
            master_assignments[name] = master_assignments[name] + [task] if name in master_assignments else [task]

        # remove used indices (important to reverse indices before deleting)
        for i in sorted(worker_batch_indices, reverse=True):
            del workerList_copy[i]

        # remove empty lists from tasksByTime_copy
        tasksByTime_copy = list(filter(lambda x: x != [], tasksByTime_copy))

        # increment batch number and master_cost
        master_cost += cost
        batch_number += 1

    return (master_cost, master_assignments)

def build_from_csv(fname: str = "schedule", DELIM: str = "\t") -> (list, list, list):
    """Generate the building block for building the cost matrices.
        Reads data from an input csv.

        optional:
        fname <- name of csv file containing data.
        fpath <- file path of file containing data.
        DELIM <- the character sepearting (date, time, exam_room) in the csvs column names.
    """
    with open(f"{fname}.csv", "r+") as f:
        data = [k for k in [[j.strip() for j in i.split(",") if j.strip() != ""] 
            for i in f.readlines()] if k != []]
        tasks = data.pop(0)
        workers = [i.pop(0) for i in data]
        workerList = []
        dataList = []
        tasks.pop(0)
        
        tasksPerWorker = len(tasks)//len(workers)
        surplusTasks = len(tasks) % len(workers)
        surplusWorkers = sample(workers, k=surplusTasks)

        for worker, dat in zip(workers, data):
            isSurplusWorker = int(worker in surplusWorkers)
            workerList += [worker] * (tasksPerWorker + isSurplusWorker)
            dataList += [dat] * (tasksPerWorker + isSurplusWorker)

    dataList = [[int(j) for j in i] for i in dataList]
    taskParts = [(i[:-3], i[-3:]) for i in tasks] # [(date-time, exam room), ...]

    # Raw materials
    # examRooms = set([i[1] for i in taskParts])
    dateTimes = set([tuple(j.strip() for j in i[0].split(DELIM))[:2] for i in taskParts])
    tasksByTime = []

    for date, time in dateTimes:
        row = []
        for i in range(len(taskParts)):
            tup = taskParts[i]
            idate, itime = tuple(j.strip() for j in tup[0].split(DELIM))[:2]
            if idate == date and itime == time:
                row.append((idate, itime, tup[1], i)) # date, time, exam room, column in data matrix
        tasksByTime.append(row)

    return (workerList, tasksByTime, dataList)

def create_output_csv(assignments: list, fname: str = "data/optimal_schedule", DELIM: str = "\t"):
    """Create csv file displaying all of the invigilation assignments.
    Data:
    assignments -> data: {
        date: {
            time: {
                examRoom: 
                    worker
            }
        }
    } -> csv
    Output: 
    date, time, examroom 1, examroom 2, ...
    month day,  start - end, worker i , worker j, ...
    ...
    """
    # Convert assignments into data dictionary (see format above).
    data = {}
    examRooms = set()
    for worker, task in assignments:
        try: 
            date, time, examRoom = task.split(DELIM)
            examRooms.add(examRoom)
            if date not in data:
                data[date] = {time: {examRoom: worker}}
            else:
                if time not in data[date]:
                    data[date][time] = {examRoom: worker}
                else:
                    data[date][time][examRoom] = worker
        except ValueError:
            print("TASK ERROR: ",worker, task)
    
    # make sure each data[date][time] has all exam rooms
    for date in data:
        for time in data[date]:
            for examRoom in examRooms:
                if examRoom not in data[date][time]:
                    data[date][time][examRoom] = ""
    
    # Score all of the dates, used to sort from soonest to farthest date.
    datekeys = []
    for key in data:
        _, date = key.split("-")
        month, day =  date.strip().split(" ")
        score = list(month_abbr).index(month) * 1e4 + int(day)
        datekeys.append((key, score))
    datekeys.sort(key=lambda a: a[1])
    
    # Initialize a string to write sorted csv data.
    csvString = ""
    for key in [i[0] for i in datekeys]:
        subkeys = []
        for skey in data[key]:
            score, _ = [int(i.strip().replace(":","")) for i in skey.split("-")]
            subkeys.append((skey, score))
        subkeys.sort(key=lambda a: a[1])

        for skey in [i[0] for i in subkeys]:
            line = f"{key},{skey}"
            supersubkeys = []
            for sskey in data[key][skey]:
                score = int(sskey.strip().replace("/",""))
                supersubkeys.append((sskey,score))
            supersubkeys.sort(key=lambda a: a[1])

            for sskey in [i[0] for i in supersubkeys]:
                line += f",{data[key][skey][sskey]}"
                
            line += ",\n"
            csvString += line
    
    # Create column names and create the output csv file.
    sortedExamRooms = sorted(list(examRooms), key= lambda x: int(x.replace("/",""))) 
    with open(f"{fname}.csv", "w+") as f:
        f.write("date, time," + ",".join(sortedExamRooms) + ",\n")
        f.write(csvString)

def gen_graph(BATCHES = range(3, 50, 7)):
    """Create performance graph: Cost v. Batch Size
    Optional:
    BATCHES <- The list of BATCH_SIZE's to feed to batched_linear_assignment_search
    """
    # 1) Create the absolute minimum cost
    costmatrix = CostMatrix()
    costmatrix.build_from_csv("data/schedule")
    min_cost, _ = costmatrix.scipy_RLAP()
    workers, tasks, data = build_from_csv(fname="data/schedule")

    # 2) Create random cost
    random_cost, _, _, _ = batched_linear_assignment_search(workers,tasks, data, DEPTH=10, BATCH_SIZE=20, ALGORITHM="random_assignment")
    
    # 3) Create batch costs
    out = []
    for batch_size in BATCHES:
        cost, _, _, _ = batched_linear_assignment_search(workers,tasks, data, DEPTH=5, BATCH_SIZE=batch_size, ALGORITHM="linear_sum_assignment")
        out.append((batch_size,cost,min_cost/cost, cost/random_cost))

    # 4) print the costs
    print("OUT: ",out)
    print(random_cost)
    print(min_cost)

    # 5) create a plot
    fig = plt.figure()
    fig.suptitle('Cost vs. Batch size')
    line_up, = plt.plot([x[0] for x in out], [x[2] for x in out],  label='min cost to batch cost')
    line_down, = plt.plot([x[0] for x in out], [x[3] for x in out],  label='batch cost to random cost')
    plt.xlabel("Batch size")
    plt.legend(handles=[line_up, line_down])
    plt.show()

def main():
    if args.init_random_csv:
        # If script is run with --init_random_csv
        workers = get_workers()
        tasks = get_tasks()
        DELIM = "\t"
        init_random_csv(workers, tasks, fname=args.iname or "data/schedule", DELIM=DELIM, affinity=args.affinity or "LOW") 
    else:
        DEPTH, BATCH_SIZE, ALGORITHM = args.depth or 10, args.batch_size or len(workers), args.algorithm or "linear_sum_assignment"
        workers, tasks, data = build_from_csv(fname=args.fname or "data/schedule")
        cost, assignments, count, savings = batched_linear_assignment_search(workers,
            tasks, data, DEPTH=DEPTH, BATCH_SIZE=BATCH_SIZE, ALGORITHM=ALGORITHM)
        create_output_csv(convert(assignments), args.oname or "data/BLAS_optimal_schedule")
        
        print("\nASSIGNMENTS: ", assignments)
        print(f"Depth: {DEPTH} | Batch size: {BATCH_SIZE} | Algorithm: {ALGORITHM}")
        print("COST: ", cost, " savings: ", savings)
        print("Time conflicts = ", hasConflicts(assignments))
        print("Iteration depth: ", count)

if __name__ == "__main__": main()

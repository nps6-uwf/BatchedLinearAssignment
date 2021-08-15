# name:    costmatrix.py
# version: 24
# python:  3.8.6
# Author:  Nick Sebasco
# Date:    12/13/2020
# ---------------------------------------------------
# Updates:
# 1. ________


"""
Problem: optimal group assignemnt given preference

 Group|  A   |  B  | ... |  N  |
Person|-------------------------
    P1|  wa1 | wb1 |
    P2|
   ...| 
    PM| 

Let each person assign a group a score of 1 - 5.
The weight that comprises the cells of the cost
matrix will be the score/ N, where N is the # of
groups.

case 1) 1 person per group

case 2) N persons per group

"""
# 0) imports
import numpy as np
from random import randint, choice, sample
from time import time
import copy
from colorama import Fore, Style
from functools import reduce
from typing import Callable
import scipy.optimize as sopt

class CostMatrix:
    def __init__(self, G: np.matrix = np.matrix([], dtype=float),
     N: int = 0, taskNames: list = [], workerNames: list = [], THRESHOLD: float = 1e-16):
        self.taskNames = taskNames if taskNames != [] else [f"Task {i}" for i in range(1, N+1)]
        self.workerNames = workerNames if workerNames != [] else [f"Worker {i}" for i in range(1, N+1)]
        self.G = G
        self.G0 = np.matrix(np.copy(G))
        self.N = N
        self.THRESHOLD = THRESHOLD # (x <= THRESHOLD) -> x == 0
        self.useMax = False
    
    def __repr__(self):
        return self.G0

    def find_zeros(self, MAT: np.matrix) -> list:
        """Returns a list of indices of all 0s in a matrix.
        """
        zer = np.where(MAT <= self.THRESHOLD)
        return [(i, j) for i, j in zip(zer[0],zer[1])]
    
    def subtract_min(self, MAT: np.matrix, LOG = None) -> list:
        """Subtract all elements in each row by smallest element in that row.  Same for columns.
        """
        mrows, ncols = MAT.shape
        # Subtract smallest value in each row from every element in that row.
        for m in range(mrows):
            a=np.empty(ncols); a.fill(np.matrix.min(MAT[m]))
            MAT[m] -= a
        if LOG:
            print("\n\u21E8 Subtracting every element in every row by smallest value in that row.")
            LOG(MAT)

        # Subtract smallest value from each column from every element in that column.
        for n in range(ncols):
            a=np.empty(ncols); a.fill(np.matrix.min(MAT[:,n]))
            MAT[:,n] -= a.reshape(mrows, 1)

        if LOG:
            print("\n\u21E8 Subtracting every element in every column by smallest value in that column.")
            LOG(MAT)
            print()

        return self.find_zeros(MAT)

    def bifurcate(self, zeroList: list, N: int, count: int = 0, res: list = []) -> list:
        """make smallest number of cuts (K) to extract 0's.  IF K == N, stop.
        """
        if len(zeroList) > 0 and count < N:
            r, c = zeroList[-1]
            c_zeroList = [ x for x in zeroList if x[1] != c ]
            r_zeroList = [ x for x in zeroList if x[0] != r ]
            return self.bifurcate(c_zeroList, N, count + 1, res + [("C", c)]) + self.bifurcate(r_zeroList, N, count + 1, res + [("R", r)])
        else:
            if count <= N:
                return [res + [count]]
            else:
                return []

    def find_min(self, covers: list) -> (float, list):
        """Find min covers needed to cover all 0s 

            Parameters:
            covers <- list of all possible covers <= self.N.

            Output:
            if min = N optimal assignment possible. returns ( min covers needed, covered rows/ cols).
        """
        cover = min(covers, key=lambda x: x[-1])
        return (cover.pop(), cover)

    def reshapify(self, MAT: np.matrix, covers) -> (np.matrix, float):
        """Finds minimum element in matrix that is not covered.
        """
        # If min != N, find min element (d) in matrix not covered
        GAT = np.copy(MAT)
        for cat, n in covers:
            row = np.empty(GAT.shape[1]); row.fill(float("inf"))
            if cat == "C":
                GAT[:,n] = row
            elif cat == "R":
                GAT[n] = row
        return (GAT, GAT.min())

    def hungarian_update(self, Z: list, MAT: np.matrix, N: int, max_iterations: int = 20, LOG: bool = True) -> list:
        """iteratively transform matrix until minimum covers needed to cover 0s == N.
        """
        _min, _path = None, None
        _iter_count = 0
        times = []
        Z_copy = copy.deepcopy(Z)

        # 2.8) repeat 2.3 - 2.7
        while _iter_count < max_iterations:
            Z_copy = self.find_zeros(MAT)

            t06 = time()
            res7 = self.bifurcate(Z_copy[:], N, res = [])
            times.append(time() - t06) # time each birfurcation.

            copy_res7 = copy.deepcopy(res7)
            _iter_count += 1
            _min, _path = self.find_min(copy_res7)
            
            if LOG:
                #print(res7, len(res7))
                print(f"covers <{_min}> == N <{N}> = {_min == N}")
                #print("path: ", _path)
                #print("min == N: ", N == _min)

            if N == _min:
                if LOG:
                    print(f"Found an optimal assignment after {_iter_count} iterations.")
                break
            else:
                _, minElement = self.reshapify(MAT, _path)

                # 2.6) Subtract min elenment (d) from each uncovered row 
                uncovered_rows = set(range(MAT.shape[0])).difference(set([i[1] for i in _path if i[0] == "R"]))
                #print(MAT)
                delta =  np.matrix([minElement if i in uncovered_rows else 0 for i in range(MAT.shape[1])]).T
                MAT -= delta
                if LOG:
                    print(f"Searching for optimal assignment: {_iter_count}/ {max_iterations}")
                    #print("uncovered: ",uncovered)
                    print("min element: ", minElement)
                    print("2.6)")
                    #print("uncovered rows: ",uncovered_rows)
                    print(MAT)
                    #print("DELTA: ", delta)

                # 2.7) Add min element (d) to each covered col
                covered_cols = [i[1] for i in _path if i[0] == "C"]
                for col in covered_cols:
                    for row in range(MAT.shape[0]):
                        MAT[row, col] = MAT[row, col] + minElement
                if LOG:
                    print("2.7)")
                    print("covered cols: ",covered_cols)
                    print(MAT)
        return times
    
    def col_swap(self, MAT, MAT0, k = 0, res = [], pro = []):
        """Every N elements in paths corresponds to a valid output selection in the cost matrix.
        """
        row, col = MAT.shape
        if k < row:
            cols = set(range(col)).difference(pro)
            paths = []
            for c in cols:
                if MAT[k, c] == 0:
                    newRes = res + [(MAT0[k,c],(k, c))]
                    paths += self.col_swap(MAT, MAT0, k + 1, newRes, pro + [c])
            return paths
        else:
            return [res]

    def get_optimal_assignment(self, LOG: bool = True) -> (list, float, float):
        """Find optimal assignment of N tasks to N workers by minimizing cost.
        """
        logger = lambda MAT: self.pretty_print_matrix(MAT,
            color="RED",
            showCols=False, 
            showRows=False, 
            criteria=lambda x: x == 0)
        Z = self.subtract_min(self.G, LOG=logger) if LOG else self.subtract_min(self.G)
        
        Z_COPY_1 = copy.deepcopy(Z)
        times = self.hungarian_update(Z_COPY_1, self.G, self.N,LOG=LOG)

        if LOG: 
            print("G:")
            if self.N <= 10:
                self.pretty_print_matrix(self.G,
                    color="RED",
                    showCols=False, 
                    showRows=False, 
                    criteria=lambda x: x == 0)
            else:
                print(self.G)
            
            print("G0:")
            print(self.G0)

        res = self.col_swap(self.G, self.G0)
        opt = choice(res)

        cost = float(sum([1/x[0] if self.useMax else x[0] for x in opt]))
        optimal_assignment = [[self.taskNames[j[1]], self.workerNames[j[0]]] for _, j in opt]

        return (optimal_assignment, cost, sum(times)/len(times))

    def scipy_RLAP(self) -> (float, list):
        """Use scipy optimize to solve linear assignment problem.

        output: tuple
        cost: float representing the total cost.
        assignments: list of assignments.
        """
        row_ind, col_ind = sopt.linear_sum_assignment(np.array(self.G))

        return (np.matrix.sum(self.G0[row_ind, col_ind]),
            [(self.workerNames[i], self.taskNames[j]) for i, j in zip(row_ind, col_ind)])

    def maximize(self):
        """Take reciprocal of self.G & self.G0
        
        Purpose:
        Optimal assignment is achieved by minimizing cost.  In order to
        find optimal assignment that maximizes cost, simply take the 
        reciprocal forcing large values to become small.

        Side effects:  
        self.G & self.G0 are reassigned as the elementwise reciprocal.
        self.useMax is set to True
        """
        self.G = np.reciprocal(self.G.astype(np.float))
        # self.G0 = np.copy(self.G)
        self.useMax = True

    def init_random(self, N: int):
        """Initialize random NxN dimensional CostMatrix.

        Parameters:
        N <- Integer specifying the # of columns and rows
            in the matrix
            
        Side effects:
        self.N , self.taskNames, self.workerNames, self.G, self.G0
        """
        self.taskNames = [f"Task {i}" for i in range(1, N+1)]
        self.workerNames = [f"Worker {i}" for i in range(1, N+1)]
        self.N = N
        self.G = np.matrix(np.random.randint(1,N,(N,N)),dtype=float)
        self.G0 = np.copy(self.G)

    def build_from_csv(self,fname="schedule"):
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

        self.G = np.matrix([[int(j) for j in i] for i in dataList])
        self.G0 = np.matrix(np.copy(self.G))
        self.taskNames = tasks
        self.workerNames = workerList
    
    def set_console_width(self, desired_width = 500):
        """Uses pandas module to set console width.

        Purpose:
        Can be used in conjunction with pretty_print_matrix to
        help view larger cost matrices.
        """
        try:
            import pandas as pd
            pd.set_option('display.width', desired_width)
        except ModuleNotFoundError:
            print("pandas module required.")
            print("pip install pandas")
        
    def pretty_print_matrix(
        self,
        MAT: np.matrix, 
        color: str = "BLUE",
        frame: dict = {
            "col": "\u2506 ", 
            "row": "-"
        }, 
        showCols: bool = True, 
        showRows: bool = True, 
        criteria: Callable[[float], bool] = lambda x: x == x,
        rowLabels: list = [],
        colLabels: list = [],
        setCellSize: int = -1
        ) -> None:
        """Prints an np.matrix to console, colors elements meeting specific criteria.

            Purpose:
            Help visualize structure & transformations of matricies.  For large matricies
            The output will be deformed because of the console width. 

            Parameters:
            MAT <- NxM dimensional np.matrix
            color [optional] <- A string specifying a valid color from colorama's Fore.
            frame [optional] <- A dictionary with key/values specifying the structural 
                components of the output table.
            showCols [optional] <- Boolean indicating whether to delineate columns by the 
                character specified in the frame dictionary's "col" value.
            showRows [optional] <- Boolean indicating whether to delineate rows by the 
                character specified in the frame dictionary's "row" value.
            criteria [optional] <- A function that takes each of the matrix elements as
                input and decides whether to color the element.
            rowLabels [optional] <- A list of strings, which if added will be prepended 
                to each row as a row label.
            colLabels [optional] <- A list of strings, which if added will be prepended 
                to each col as a col label.
            setCellSize [optional] <- An integer to specify the width of each cell so that
                full column names can be shown.  This parameter will only expand cells.

            Output:
            Values of a matrix in tabular format styled according to parameters.

            Improvements:
            - [FIXED] If col name < cell size; columns will not align.
            - [FIXED] Col names also get auto reduced to fit in cell,
                maybe add optional parameter to manually expand
                cell size.
        """
        
        if len(rowLabels) > 0:
            # ensure row labels have equal length
            rl_max = len(max(rowLabels,key= lambda x: len(x)))
            rowLabels = [rlabel + " " * (rl_max - len(rlabel))  for rlabel in rowLabels]
        
        frame_col = frame["col"] if "col" in frame else "\u2506" # [\u2582, \u2506]
        frame_row = frame["row"] if "row" in frame else "-"

        deconstructed = np.array(MAT).reshape((1, reduce(lambda a,b: a*b, MAT.shape)))[0]
        cell_size = max([len(str(i)) for i in deconstructed]) 
        cell_size = cell_size if setCellSize < cell_size else setCellSize
        offset = {}

        # correction term needed because hex characters effect string length
        corr = len(Style.RESET_ALL) + len(getattr(Fore, color))

        for r in range(MAT.shape[0]):
            offset[r] = 0
            for c in range(MAT.shape[1]):
                if criteria(MAT[r, c]):
                    offset[r] = corr + offset[r] 


        cells = [getattr(Fore, color) + str(i) + (cell_size - len(str(i)))*" " + Style.RESET_ALL 
                    if criteria(i) else str(i) + (cell_size - len(str(i)))*" " for i in deconstructed]
        rows = []

        curr_row = f"{frame_col}" if showCols else ""
        count = 0
        for i in range(len(cells)):
            curr_row += cells[i] + " "+ frame_col if showCols else cells[i] + " "
            if i % MAT.shape[1] == MAT.shape[1] -1  and (i > 0 or MAT.shape[1] == 1):
                rows.append(curr_row if len(rowLabels) == 0 else rowLabels[count] + curr_row)
                if showRows:
                    if len(rowLabels) > 0:
                        rows.append("".join([frame_row for _ in range(len(curr_row)-offset[count] - 1 + len(rowLabels[count]))]))
                    else:
                        rows.append("".join([frame_row for _ in range(len(curr_row)-offset[count] - 1  )]))
                curr_row = f"{frame_col}" if showCols else ""
                count += 1

        if len(colLabels) > 0:
            colLabels = [ clabel + " " * (cell_size - len(clabel)) if len(clabel) < cell_size else clabel[:cell_size] for clabel in colLabels]
            roff = "\u2591"*len(rowLabels[0]) if len(rowLabels) > 0 else ""
            if showRows:
                print("".join([frame_row for _ in range(len(rows[-2])-offset[count - 1] - 1)]))
            if showCols:
                print( roff + f"{frame_col}" + f" {frame_col}".join(colLabels) +f" {frame_col}")
            else:
                print(roff + f" ".join(colLabels))

        if showRows:
            print("".join([frame_row for _ in range(len(rows[-2])-offset[count - 1] - 1)]))

        print("\n".join(rows))


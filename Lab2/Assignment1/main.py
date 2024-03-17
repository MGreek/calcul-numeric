import numpy as np


def handleZeros(matrix: np.ndarray):
    assert(len(matrix.shape) == 2)

    for row in matrix:
        for element in row:
            if np.isclose(element, 0):
                element = 0

def compareRows(row1: np.ndarray, row2: np.ndarray):
    assert(len(row1.shape) == len(row2.shape) == 1)

    for i in range(row1.shape[0]):
        if np.isclose(row1[i], row2[i]):
            continue
        return -1 if (np.abs(row1[i]) < np.abs(row2[i])) else 1
    return 0

def swapRows(matrix: np.ndarray):
    assert(len(matrix.shape) == 2)

    handleZeros(matrix)
    for i in range(matrix.shape[0] - 1):
        for j in range(i + 1, matrix.shape[0]):
            if compareRows(matrix[i], matrix[j]) < 0:
                matrix[[i, j]] = matrix[[j, i]]

def findLeadingCoeficient(matrix: np.ndarray, rowIndex: int):
    assert(len(matrix.shape) == 2)

    columnIndex = 0
    while columnIndex < matrix[rowIndex].shape[0] and np.isclose(matrix[rowIndex][columnIndex], 0):
        columnIndex += 1

    return columnIndex

def checkIfSolvable(row: np.ndarray, partialSolution: np.ndarray):
    assert(len(row.shape) == len(partialSolution.shape) == 1)
    assert(partialSolution.shape[0] == (row.shape[0] - 1))

    total = row[-1]
    for i in range(partialSolution.shape[0]):
        if partialSolution[i] == None and not np.isclose(row[i], 0):
            return True
        if np.isclose(row[i], 0):
            continue
        total -= partialSolution[i] * row[i]
    return np.isclose(total, 0) 

def deduceVars(row: np.ndarray, partialSolution: np.ndarray):
    assert(len(row.shape) == len(partialSolution.shape) == 1)
    assert(partialSolution.shape[0] == (row.shape[0] - 1))
    
    if not checkIfSolvable(row, partialSolution):
        return False
    
    total = row[-1]
    index = None
    for i in range(partialSolution.shape[0]):
        if partialSolution[i] == None and not np.isclose(row[i], 0):
            if index == None:
                index = i
                continue
            else:
                partialSolution[i] = 1
        if np.isclose(row[i], 0):
            continue
        total -= partialSolution[i] * row[i]
    if index != None:
        partialSolution[index] = total / row[index]
    
    return True 

def solve(matrix: np.ndarray, vector: np.ndarray):
    assert(len(matrix.shape) == 2)
    assert(vector.shape[0] == matrix.shape[0])

    x = np.concatenate((matrix, vector), axis=1)
    swapRows(x)
    for rowIndex in range(x.shape[0] - 1):
        columnIndex = findLeadingCoeficient(x, rowIndex)
        if columnIndex >= x.shape[1]:
            break
        for newRowIndex in range(rowIndex + 1, x.shape[0]):
            x[newRowIndex] = x[newRowIndex] - x[rowIndex] * (x[newRowIndex][columnIndex] / x[rowIndex][columnIndex])
        swapRows(x)
    solution = np.array([None for _ in range(matrix.shape[1])])
    for row in x[::-1]:
        if not deduceVars(row, solution):
            return None
    for val in solution:
        if val == None:
            val = 1
    return np.array([solution]).transpose()


A = np.array([[10, 7, 8, 7], [7, 5, 6, 5], [8, 6, 10, 9], [7, 5, 9, 10]])
b = np.array([[32, 23, 33, 31]]).transpose()
print(solve(A, b))

Ap = np.array([[10, 7, 8.1, 7.2], [7.08, 5.04, 6, 5], [8, 6, 9.98, 9], [6.99, 4.99, 9, 9.98]])
bp = np.array([[32.1, 22.9, 33.1, 30.9]]).transpose()
print(solve(Ap, bp))
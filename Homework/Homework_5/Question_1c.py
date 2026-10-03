import numpy as np

def evalf(x,y):
    return 3*x[0]**2 - y[0]**2

def evalg(x,y):
    return 3*x[0]*y[0]**2 - x[0]**3 - 1

def evalJ(x,y):
    J = np.zeros(2,2)
    J[0] = [6*x[0],-2*y[0]]
    J[1] = [3*x[0]*y[0]**2, 6*x[0]*y[0]]

def driver():
    arr0 = np.array([[1],[1]])
    arr1 = np.array([[0],[0]])
    const = np.array([[(1/6),(1/18)],[(0),(1/6)]])

    nMax = 100

    for j in range(nMax):
        fval = evalf(arr0[0],arr0[1])
        gval = evalg(arr0[0],arr0[1])
        vals = np.array([[fval],[gval]])
        update = const @ vals
        arr1 = arr0 - update

        print(f"{j}: x = {arr0[0]} y = {arr0[1]}")
        arr0 = arr1


if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()   
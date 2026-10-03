import numpy as np

def f(p):
    return p[0]**2 + 4*p[1]**2 + 4*p[2]**2 - 16

def grade(p):
    return np.array([2*p[0],8*p[1],8*p[2]])

def driver():
    vals = np.array([1.0,1.0,1.0])
    results = np.zeros((11,3))
    results[0] = vals

    for i in range(10):
        g = grade(vals)
        temp = vals - f(vals)/np.dot(g,g)*g
        results[i+1] = (temp)
        vals = temp

    pstar = results[-1]
    print(pstar)
    err = np.zeros(11)
    for i in range(10):
        err[i] = np.linalg.norm(results[i]-pstar)
    print(err)



driver()
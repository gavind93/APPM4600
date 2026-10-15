import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv

def evalLine(x0,x1,f0,f1,alpha):
    m = (f1-f0)/(x1-x0)
    return  f0+m*(alpha-x0)

def driver():
    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 200
    xeval = np.linspace(a,b,Neval)

    ''' number of intervals'''
    Nint = 10

    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)

    ''' evaluate f at the evaluation points'''
    fex = f(xeval)

    plt.plot(xeval,fex,"ro-",label="exact function")
    plt.plot(xeval,yeval,"bs--",label="natural spline")
    plt.legend
    plt.show()


    err = abs(yeval-fex)
    plt.figure()
    plt.semilogy(xeval,err,"ro--",label="absolute error")
    plt.legend()
    plt.show()

    err = abs(yeval-fex)
    plt.figure()
    plt.semilogy(xeval,err,"ro--",label="absolute error")
    plt.legend()
    plt.show()
def eval_lin_spline(xeval,Neval,a,b,f,Nint):
    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)

    n = int(Neval/Nint)

    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval)
    for jint in range(Nint):

        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''

        ind = np.zeros(n)
        for k in range(n):
            ind[k] = xeval[(jint*n)+k]

    
        '''temporarily store your info for creating a line in the interval of interest'''
        a1= xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)

        for kk in range(n):
            '''use your line evaluator to evaluate the lines at each of the points in the interval'''
            '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with the points (a1,fa1) and (b1,fb1)'''
            yeval[(jint*n)+kk] = evalLine(a1, b1, fa1, fb1, ind[kk])
    print("check")
    return yeval



if __name__ == "__main__":
    # run the drivers only if this is called from the command line
    driver()
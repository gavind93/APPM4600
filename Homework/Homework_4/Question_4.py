# This code compares the performance of secant method with Newton method
# for finding roots of 1-d nonlinear equations.
#############################################
"""
Copyright (C) 2026 Adrianna M. Gillman
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
#############################################
import matplotlib.pyplot as plt
import scipy
import numpy as np
import math
def driver():
    # f = lambda x: (x-2)**3
    # fp = lambda x: 3*(x-2)**2
    # p0 = -5
    # p1 = 3
    f = lambda x: x**6-x-1
    fp = lambda x: 6*x**5-1
    p0 = 2
    p1 = 1
    Nmax =100
    tol = 1e-8
    [p,pstar,info,it1] = newton(f,fp,p0,tol,Nmax)
    print('real','Newton: the approximate root is:',pstar)
    print('inter','Newton: Error message reads:',info)
    print('inter','Newton: Number of iterations:',it1)
    print(p)
    err1 = abs(p[1:it1]-pstar);
    [p,pstar,info,it2] = secant(f,p0,p1,tol,Nmax)
    print('real','Secant: the approximate root is:',pstar)
    print('inter','Secant: Error message reads:',info)
    print('inter','Secant: Number of iterations:',it2)
    print(p)
    err2 = abs(p[1:it2]-pstar)
    plt.loglog(err1[0:it1-2],err1[1:it1])
    plt.xlabel('|x_k-alpha|')
    plt.ylabel('|x_{k+1}-alpha|')
    plt.loglog(err2[0:(it2-2)],err2[1:it2])
    # plt.savefig('errors.pdf')
    plt.show()
def newton(f,fp,p0,tol,Nmax):
    """
    Newton iteration.
    Inputs:
    f,fp - function and derivative
    p0 - initial guess for root
    tol - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
    Returns:
    p - an array of the iterates
    pstar - the last iterate
    info - success message
    - 0 if we met tol
    - 1 if we hit Nmax iterations (fail)
    """
    p = np.zeros(Nmax+1)
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]
def secant(f,p0,p1,tol,Nmax):
    """
    Secant iteration.
    Inputs:
    f,fp - function and derivative
    p0 - initial guess for root
    p1 - a second guess for the root
    tol - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
    Returns:
    p - an array of the iterates
    pstar - the last iterate
    info - success message
    - 0 if we met tol
    - 1 if we hit Nmax iterations (fail)
    """
    p = np.zeros(Nmax+1)
    p[0] = p0
    p[1] = p1
    fp0 = f(p0)
    fp1 = f(p1)
    for it in range(1,Nmax):
        p2 = p1-fp1*(p1-p0)/(fp1-fp0)
        p[it+1] = p2
        if (abs(p1-p2) < tol):
            pstar = p2
            info = 0
            return [p,pstar,info,it]
        p0 = p1
        fp0 = fp1
        p1 = p2
        fp1 = f(p2)
    pstar = p2
    info = 1
    return [p,pstar,info,it]
if __name__ == '__main__':
# run the drivers only if this is called from the command line
    driver()


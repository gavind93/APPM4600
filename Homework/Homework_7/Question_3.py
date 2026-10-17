"""
Copyright (C) 2025 Adrianna M. Gillman
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
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
def driver():
    # f = lambda x: np.exp(x)
    f = lambda x: 1/(1+(10*x)**2)
    N = 50
    ''' interval'''
    a = -1
    b = 1
    ''' create equispaced interpolation nodes'''
    xint = np.zeros(N)
    for j in range(1,N+1):
        xint[j-1] = np.cos(((2*j-1)*np.pi)/(2*N))
    ''' create interpolation data'''
    yint = f(xint)
    ''' create points for evaluating '''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    V = matrixCalc(xint)
    prod = la.inv(V) @ yint

    p = np.zeros(Neval + 1)
    for val in prod:
        p = p*xeval + val

    fex = f(xeval)

    plt.figure()
    plt.plot(xeval,fex,'r-', label = 'f(x)')
    plt.plot(xeval,p,'b--', label = 'p(x)')
    plt.legend()

    plt.figure()
    err = abs(p-fex)
    plt.semilogy(xeval,err,'r--',label='lagrange')
    plt.legend()
    plt.show()

def matrixCalc(x):
    n = len(x)
    V = np.zeros((n,n))
    for k in range(n):
        V[:,k] = x**(n-1-k)
    return V

driver()

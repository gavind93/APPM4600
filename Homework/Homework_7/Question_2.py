"""
 This script code compares the use of Lagrange and Newton Divided 
 Differences for interpolation. 
"""

############################################# 
"""
Copyright (C) 2025  Adrianna M. Gillman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
############################################# 



import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():


#    f = lambda x: np.exp(x)
    f = lambda x: 1/(1+(10*x)**2)


    N = 20
    ''' interval'''
    a = -1
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    
    ''' create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1) # lagrange
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange_bary(xeval[kk],xint,yint)

    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure()    
    plt.plot(xeval,fex,'ro-',label = "f(x)")
    plt.plot(xeval,yeval_l,'bs--', label = "p(x)") 
    plt.legend()

    plt.figure() 
    err_l = abs(yeval_l-fex)
    plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    plt.legend()
    plt.show()

def eval_lagrange_bary(xeval,xint,yint):
    for i in range(len(xint)):
        if xeval - xint[i] == 0:
            return yint[i]

    
    w = np.ones(len(xint))
    for j in range(len(xint)):
        for i in range(len(xint)):
            if i != j:
                w[j] /= (xint[j] - xint[i])

    return np.sum((w/(xeval-xint))*yint)/np.sum((w/(xeval-xint)))

def evalW(xint,j):
    prod = 1
    for i in range(len(xint)):
        if(i != j):
            prod *= (xint[j]-xint[i])
    return 1/prod
  
def evalPhi(x,xint):
    prod = 1
    for i in range(len(xint)):
        prod *= (x-xint[i]) 
    return prod



       

driver()        

"""
 This script utilized the fixed point method to find the fixed
 Point of the nonlinear system of equations.
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
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver():

    x0 = np.array([-3, 3])
    
    Nmax = 100
    tol = 1e-4
    
    t = time.time()
    for j in range(50):
      [xstar,ier,its] =  fixedSys(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Fixed pt: the error message reads:',ier) 
    print('Fixed pt: took this many seconds:',elapsed/50)
    print('Fixed pt: number of iterations is:',its)


    [xstar,ier,its] =  Newton(x0,tol,Nmax)

    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Netwon: number of iterations is:',its)

def evalF(x): 

    F = np.zeros(2)
    
    F[0] = 3*x[0]**2 + 4*x[1]**2 - 1
    F[1] = x[1]**3 - 8*x[0]**3 - 1
 
    return F
    
def evalJ(x):
    return np.array([[6*x[0],       8*x[1]],
                     [-24*x[0]**2,  3*x[1]**2]])

def evalG(x):
    A = np.array([[0.016, -0.17],[.52,-0.26]])
    return x - A @ evalF(x)

def fixedSys(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       
       x1 = evalG(x0)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

driver()       

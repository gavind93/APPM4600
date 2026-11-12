"""
 This script provides you the opprotunity play with different linear algebra solvers.
You will learn about the performance of the different solvers and how to build
test some test matrices to challenge the different methods. 
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


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time


def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     N = 5000
 
     ''' Right hand side'''
     b = np.random.rand(N,1)
     A = np.random.rand(N,N)

     t1 = time.time()
     x = scila.solve(A,b)
     t2 = time.time()
     scila_time = t2-t1
     
     test = np.matmul(A,x)
     r = la.norm(test-b)
     
     print(r)

     lu_res,ctime,stime = lu_fact(A,b)

     print(f'Scila time is {scila_time} seconds')
     print(f'LU construction time is {ctime} seconds')
     print(f'LU solving time is {stime} seconds')
     print(f'LU total time is {ctime+stime} seconds')



     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)




     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          

def lu_fact(A,b):
     t1 = time.time()
     lu_decomp,piv = scila.lu_factor(A)
     t2 = time.time()
     con_time = t2-t1
     t1 = time.time()
     res = scila.lu_solve((lu_decomp,piv),b)
     t2 = time.time()
     sol_time = t2-t1
     return res, con_time, sol_time


  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       

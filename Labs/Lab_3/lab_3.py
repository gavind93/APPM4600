"""
This script explores the use of the fixed point method.
Two functions are considered that have different properties.
I like to use this code before I talk about convergence analysis
for the fixed point method as motivation.
"""
#############################################
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
"""
This code has been modified from it original published state for the
purposes of completing Lab Assignment number 3 for APPM 4600. 
The modifications only entail changing the subroutine to return
a vector of each fixed point approximation during each step of the process. 
The original unmodified code still belongs to the publisher Adrianna M. Gillman.
The modifications were made by her student Gavin Doan.
"""
# import libraries
import numpy as np
def driver():
# test functions
    f1 = lambda x: 1+0.5*np.sin(x)
# fixed point is alpha1 = 1.4987....
    f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09...
    Nmax = 100
    tol = 1e-6
# test f1 '''
    x0 = 0.0
    [xstar,ier,x_vals] = fixedpt(f1,x0,tol,Nmax)
    # print('the approximate fixed point is:',xstar)
    # print('f1(xstar):',f1(xstar))
    # print('Error message reads:',ier)
    # print('Predictions are: ', x_vals)
#test f2 '''
    x0 = 0.0
    [xstar,ier,x_vals] = fixedpt(f2,x0,tol,Nmax)
    # print('the approximate fixed point is:',xstar)
    # print('f2(xstar):',f2(xstar))
    # print('Error message reads:',ier)
#lab f '''
    f3 = lambda x: (10/(x+4))**.5
    x0=1.5
    tol = 1e-10
    [xstar,ier,x_vals] = fixedpt(f3,x0,tol,Nmax)
    x_preds = np.trim_zeros(x_vals)
    x_len = len(x_preds)
    p = 1.3652300134140976
    OoC = np.log(np.abs((x_preds[x_len-1]-p)/(x_preds[x_len-2]-p)))/np.log(np.abs((x_preds[x_len-2]-p)/(x_preds[x_len-3]-p)))
    # print('Predictions are: ', x_preds)
    # print('Number of iterations required is: ', x_len)
    # print('Order of Convergence is: ', OoC)

# define routines
def fixedpt(f,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    x_vals = np.zeros((Nmax+1,1))
    x_vals[count] = x0
    while (count < Nmax):
        count = count +1
        x1 = f(x0)
        x_vals[count] = x1
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [xstar,ier,x_vals]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier,x_vals]
driver()


def aitken(p_aprox,tol,Nmax):
    count = 0
    x_vals = np.zeros((Nmax+1,1))
    fa = lambda x,y,z: x - ((y-x)**2/(z-2*y+z))
    x_vals[count] = fa(p_aprox[count,count+1,count+2])
    while (count < Nmax-2):
        count = count +1
        x_vals[count] = fa(p_aprox[count,count+1,count+2])
        if (abs(x_vals[count]-x_vals[count-1]) <tol):
            xstar = x_vals[count]
            ier = 0
            return [xstar,ier,x_vals]
    xstar = x_vals[count]
    ier = 1
    return [xstar, ier,x_vals]


Nmax = 100
tol = 1e-6

f3 = lambda x: (10/(x+4))**.5
x0=1.5
tol = 1e-10
[xstar,ier,x_vals] = fixedpt(f3,x0,tol,Nmax)
x_preds = np.trim_zeros(x_vals)
x_len = len(x_preds)
p = 1.3652300134140976
OoC = np.log(np.abs((x_preds[x_len-1]-p)/(x_preds[x_len-2]-p)))/np.log(np.abs((x_preds[x_len-2]-p)/(x_preds[x_len-3]-p)))
print('Predictions are: ', x_preds)
print('Number of iterations required is: ', x_len)
print('Order of Convergence is: ', OoC)





aitken(x_preds,tol,Nmax)
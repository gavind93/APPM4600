"""
 This program implements steepest descent for finding the 
 root of a non-linear vector function. 
The vector function F and the corresponding Jacobian are hard 
coded and need to be changed for each problem. 
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




#libraries:
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():

    Nmax = 100
    x0= np.array([0 ,0,1])
    tol = 1e-10
    
    [xstar,gval,ier] = SteepestDescent(x0,tol,Nmax)
    print("the steepest descent code found the solution ",xstar)
    print("g evaluated at this point is ", gval)
    print("ier is ", ier)


    [xstar,ier,its] =  Newton(x0,tol,Nmax)

    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Netwon: number of iterations is:',its)

    [xstar,gval,ier] = SteepestDescent(x0,tol,Nmax)
    [val1,val2,val3] =   Newton(xstar,5e-2,Nmax)
    print(val1)
    print('SD + Newton: the error message reads:',val2) 
    print('SD + Netwon: number of iterations is:',val3)

###########################################################
#functions:
def evalF(x): 
# vector function that you want to find the roots of

    F = np.zeros(3)
    
    F[0] = x[0]+np.cos(x[0]*x[1]*x[2])-1
    F[1] = (1-x[0])**.25 + x[1] + .05*x[2]**2 - .15*x[2]-1
    F[2] = -x[0]**2 - .1*x[1]**2 + .01 * x[1] + x[2] - 1

    return F
    
def evalJ(x): 
# Jacobian of the vector function you want to find the roots of
    
    J = np.array([[1-x[1]*x[2]*math.sin(x[0]*x[1]*x[2]), -x[0]*x[2]*math.sin(x[0]*x[1]*x[2]), -x[0]*x[1]*math.sin(x[0]*x[1]*x[2])],
                  [-.25*(1-x[0])**(-.75), 1.0, .1*x[2]-.15],
                  [-2*x[0], -.2*x[1]+.01, 1.0]])

    return J

def evalg(x):
# The least squares function you are minimizing

    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
# Grad g
    F = evalF(x)
    J = evalJ(x)
    
    gradg = np.transpose(J).dot(F)
    return gradg


###############################
### steepest descent code

def SteepestDescent(x,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)
        
# below is finding the step-length using the quadratic min. 
        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3
#####
        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier]


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
           
    
     
        
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()       

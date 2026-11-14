"""
 This script tests Romberg integration for approximating integrals.
 The first column of the Romberg table is built with the composite
 trapezoidal rule and the remaining columns come from Richardson
 extrapolation.  The error in the diagonal entries is plotted as the
 number of panels is increased.
"""

#############################################
"""
Copyright (C) 2026  Adrianna M. Gillman

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


def driver():

    f = lambda x: np.sin(x)
    a = 0
    b = np.pi

    # exact integral
    I_ex = 2

    nrefine = 5


    ntrap = 2**np.arange(0, nrefine+1)

    nsimp = 2**np.arange(1, nrefine+2)

    RomT = rombergTrap(a, b, nrefine, f)
    RomS = rombergSimpson(a, b, nrefine, f)

# column j is only filled from row j down
    for j in range(0,nrefine+1):
        print('absolute error in column', j, '= ', abs(I_ex-RomT[j:,j]))


    for j in range(0, nrefine+1):
        print('abs error, column', j, '=', abs(I_ex-RomS[j:, j]))
    
    errT = abs(I_ex-np.diag(RomT))
    errS = abs(I_ex-np.diag(RomS))

    print('Trap best approximation errors    =', errT)
    print('Simpson best approximation errors =', errS)

    plt.semilogy(ntrap, errT, 'o-', label='Romberg-Trapezoid')
    plt.semilogy(nsimp, errS, 's--', label='Romberg-Simpson')
    plt.xlabel('n (panels)')
    plt.ylabel('absolute error')
    plt.legend()
    plt.show()


def rombergTrap(a,b,nrefine,f):
# number of panels: 1, 2, 4, ..., 2^nrefine
    ntest = 2**np.arange(0,nrefine+1)

    nn = len(ntest)

    Rom = np.zeros((nn,nn))

# first column: composite trapezoidal rule with n panels
    for j in range(0,nn):
        Rom[j,0] = CompTrap(a,b,ntest[j],f)

# remaining columns: Richardson extrapolation
    for j in range(1,nn):
        for i in range(j,nn):
            Rom[i,j] = Rom[i,j-1] + (Rom[i,j-1]-Rom[i-1,j-1])/(4**j-1)

    return Rom


def rombergSimpson(a, b, nrefine, f):
    ntest = 2**np.arange(1, nrefine+2)      
    nn = len(ntest)
    Rom = np.zeros((nn, nn))

    for j in range(0, nn):
        Rom[j, 0] = CompSimpson(a, b, ntest[j], f)

    for j in range(1, nn):
        for i in range(j, nn):
            Rom[i, j] = Rom[i, j-1] + (Rom[i, j-1]-Rom[i-1, j-1])/(4**(j+1)-1)

    return Rom

def CompTrap(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h

    I_trap = h*f(xnode[0])*1/2

    for j in range(1,n):
         I_trap = I_trap+h*f(xnode[j])
    I_trap= I_trap + 1/2*h*f(xnode[n])

    return I_trap

def CompSimpson(a, b, n, f):
    # n must be even
    h = (b-a)/n
    xnode = a+np.arange(0, n+1)*h
    fx = f(xnode)

    I_simp = fx[0] + fx[n]
    I_simp += 4*np.sum(fx[1:n:2]) 
    I_simp += 2*np.sum(fx[2:n-1:2])

    return h/3*I_simp




driver()

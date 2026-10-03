"""
This code implements Newton's method for finding the root of a
scalar function.
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
# import libraries
import numpy as np
def driver():
    #f = lambda x: (x-2)**3
    #fp = lambda x: 3*(x-2)**2
    #p0 = 1.2
    f = lambda x: np.e**(3*x)-27*(x**6)+27*(x**4)*(np.e**x)-9*(x**2)*(np.e**(2*x))
    fp = lambda x: 3*np.e**(3*x)-162*(x**5)+27*(x**4)*(np.e**x)+108*(x**3)*(np.e**x)-18*(x**2)*(np.e**(2*x))-18*x*(np.e**(2*x))
    fpp = lambda x: 9*(np.e**(3*x))-810*(x**4)+27*(x**4)*(np.e**x)+216*(x**3)*(np.e**x)+324*(x**2)*(np.e**x)-36*(x**2)*(np.e**(2*x))-72*x*(np.e**(2*x))-18*(np.e**(2*x))

    p0 = 4
    Nmax = 100
    tol = 1.e-14
    mult = 3
    (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    (p,pstar,info,it) = newton_2(f,fp,fpp,p0,tol, Nmax)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
    (p,pstar,info,it) = newton_3(f,fp,p0,tol, Nmax,mult)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)
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
    p = np.zeros(Nmax+1);
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
"""
Modifications by gavin doan

"""
def newton_2(f,fp,fpp,p0,tol,Nmax):
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
    p = np.zeros(Nmax+1);
    p[0] = p0
    for it in range(Nmax):
        p1 = p0 - (f(p0)*fp(p0))/(fp(p0)**2-f(p0)*fpp(p0))
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]
def newton_3(f,fp,p0,tol,Nmax,mult):
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
    p = np.zeros(Nmax+1);
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-(mult*f(p0))/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]
driver()
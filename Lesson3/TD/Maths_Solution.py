# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 21:47:00 2018

@author: ndoannguyen
"""

import numpy as np
import scipy as sp
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm #For color
from scipy.stats import norm, binom, poisson

A = np.array([ [ 0, 1,  0,  0], 
               [ 1, 2, -2,  0],
               [ 1, 0,  1, -1],
               [-1, 2, -1,  1]])
               
B = np.array([ [-1,  2,  2, -1], 
               [ 0,  1, -1,  0],
               [ 2, -2, -1,  1],
               [ 0,  0,  1,  1]])
               
MatA = np.matrix([ [ 0, 1,  0,  0], 
                   [ 1, 2, -2,  0],
                   [ 1, 0,  1, -1],
                   [-1, 2, -1,  1]])
                  
MatB = np.matrix([ [-1,  2,  2, -1], 
                   [ 0,  1, -1,  0],
                   [ 2, -2, -1,  1],
                   [ 0,  0,  1,  1]])

v = np.array([2, 3, 1, 3])
Matv = np.matrix(v).transpose()

I = np.identity(4)

MatI = np.matrix(I)

# Exercise 0
def ex0():
    print( "OK")
                
# Exercise 1
def ex1():
    print( A.dot(B).dot(A).dot(B).dot(A).dot(B).dot(A).dot(B).dot(A).dot(B) )
    print( (MatA*MatB)**5 )
    print( (MatA.dot(MatB))**5 )

# Exercise 2
def ex2():
    print( A.dot(v) )
    print( A.dot(v.transpose() ))
    print( MatA*Matv )

    print( B.transpose().dot(v) )
    print( MatB.transpose()*Matv )

    print( v.dot(A).dot(v) )
    print( Matv.transpose() * MatA * Matv)

# Exercise 3
def ex3():
    print ( np.linalg.inv(A + I) )
    print ( np.linalg.inv(MatA + MatI) )

# Exercise 4
def ex4():
    C = A[:3]
    U, eigenvectors, Vtranspose = np.linalg.svd(C)
    Sigma = np.zeros((3, 4))
    np.fill_diagonal(Sigma, eigenvectors)
    V = Vtranspose.transpose()
    print U, "\n", Sigma, "\n", V
    print "Verification: U * Sigma * V^t = ", np.matrix(U) * np.matrix(Sigma) * np.matrix(V).transpose()

# Exercise 5
def ex5():
    eigvals, eigvecs = np.linalg.eig(A)
    print eigvals
    print eigvecs
    print "Verification: ", eigvecs.dot(np.diag(eigvals)).dot(np.linalg.inv(eigvecs))
    print( np.ndim(A) )

# Exercise 6
def ex6():
    eigvals = np.linalg.eigvals(A)
    charpoly = np.poly(eigvals)
    print(charpoly)
    print( np.linalg.det(A) )
    print( np.trace(A) )

# Exercise 7
def ex7():
    print( np.eye(4) )
    print( np.linalg.solve((A + np.eye(4)), v) )

f = lambda x: 4 * np.sin(x) - np.cos(x) + 2 * np.exp(-3*x**2) + 1./x 
F = lambda x, y: ((x - y)**2.0) / (x**2 + y**2)

F_rewrite = lambda x: ((x[0] - x[1])**2.0) / (x[0]**2 + x[1]**2)
#F = lambda x: x[0] + x[1]**2
#F = lambda x, y: 1

# Exercise 8
def ex8():
    print( sp.misc.derivative(f, 3.0, dx = 1e-6, n = 1) )
    print( sp.misc.derivative(f, 3.0, dx = 1e-6, n = 2) )
    print( sp.misc.derivative(f, 3.0, dx = 1e-6, n = 3, order = 7) )
    print( sp.misc.derivative(f, 3.0, dx = 1e-6, n = 4, order = 7) )

#Exercise 9
def ex9():
    print( optimize.approx_fprime((1, -2), F_rewrite, 1e-8))

def ex10():
    print( sp.integrate.quad(f, 1, 2) )
    print( sp.integrate.dblquad(F, 0, 2, lambda x: max(0, np.sqrt(1 - x**2)), lambda x: np.sqrt(4 - x**2) ))

#Exercise 11
def ex11():
    X = np.linspace(-10, 10, 101)
    Y = map(f, X)
    plt.plot(X, Y, linestyle="-")
    plt.title("y = f(x)")
    plt.legend("y = f(x)")
    plt.grid()
    plt.show()
    for x0_estimated in [-9, -6, -3, 3, 6.48, 10]:
        print ( optimize.root(f, x0 = x0_estimated, method='broyden2').x )

#Exercise 12
def ex12():
    print("Local minima: ")
    for x0_estimated in [-8, -1, 0.5, 5]:
        print ( optimize.minimize(f, x0 = x0_estimated, method='Nelder-Mead' ).x ) 
    print("Local maxima: ")
    for x0_estimated in [-5, -0.5, 1, 8]:
        print ( optimize.minimize(lambda x: -f(x), x0 = x0_estimated, method='Nelder-Mead' ).x ) 

#Exercise 13
def ex13():    
    X = np.arange(-5, 5, 0.1)
    Y = np.arange(-5, 5, 0.1)    
    X, Y = np.meshgrid(X, Y)
    Z = F(X, Y) 
          
    CS = plt.contour(X, Y, Z, levels = [0.0, 0.5, 1, 1.5, 2])
    plt.clabel(CS, fontsize = 10, colors='k')
    plt.grid()
    plt.show()
    
    ax = plt.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.colorbar(surf)
    
    plt.show()

#Exercise 14
def ex14():
    cons = ({'type': 'eq',
             'fun' : lambda x: x[0] + x[1] - 5},)
    
    res = optimize.minimize(lambda x: -F_rewrite(x), [1, 1], constraints = cons)
    print(res)

#Exercise 15
def ex15():
    X = norm(2, np.sqrt(6))
    print(X.cdf(5) - X.cdf(-1))
    print(X.interval(0.9)[1] - 2 )

#Exercise 16
def ex16():
    X = binom(10, 0.7)
    print(X.pmf(8))
    print(1 - X.cdf(7))
    sample = X.rvs(1000)
    plt.hist(sample, np.arange(0, 12, 1), normed = True)

#Exercise 17
def ex17():
    X = poisson(1.5)
    mu = X.mean()
    sigma = X.std()
    N = 1000
    
    samples = np.zeros((N, 1000))
    for i in range(N):
        samples[i] = X.rvs(1000)
    
    result = samples.sum(axis = 0)
    normalized_result = (result - N * mu) / (np.sqrt(N) * sigma)
        
    x_range = np.arange(-5, 5, 0.2)
    plt.hist(normalized_result, x_range, normed = True, label = "Normalized sum of Poissons")
    Y = norm(0, 1)
    plt.plot(x_range, Y.pdf(x_range), linewidth = 2, label = "N(0, 1)")
    plt.legend(loc = "upper left", fontsize = 8)
    plt.show()
    
Tests = [ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9, ex10, ex11, ex12, ex13, ex14, ex15, ex16, ex17]
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Please configure your test by Run -> Configure"
    else:
        for i in range(len(Tests) + 1):
            if sys.argv[1] == "test_" + str(i):
                Tests[i]()
                break
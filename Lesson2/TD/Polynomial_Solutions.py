# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:48:04 2017

@author: ndoannguyen
"""

import math, cmath, random
EPSILON = 0.001
TOLERANCE = 0.0001
MAX_ITERATIONS = 1000

import numpy as np

#------------------------------------------------------------------------------        
def simplify(coefficients):
    """
        Simplify a list by remove zeros on right hand side
        Ex: simplify([1, 2, 3, 0, 0, 0]) = [1, 2, 3]
        Input: A list
        Output: A list
    """
    current_degree = -1
    for i in range(len(coefficients)):
        if (abs(coefficients[i]) > EPSILON):
            current_degree = i
    return coefficients[:current_degree + 1]

def getNextApproximations(f, x):
    x0, x1, x2 = x[0], x[1], x[2]
    c = f(x2)
    
    LeftMatrix = np.array([[(x0 - x2)**2, x0 - x2], [(x1 - x2)**2, x1 - x2]])
    RightMatrix = np.array([f(x0) - c, f(x1) - c])
    Solution = np.linalg.solve(LeftMatrix, RightMatrix)

    a, b = Solution[0], Solution[1]
    delta = b**2 - 4*a*c
    if abs(b + cmath.sqrt(delta)) > abs(b - cmath.sqrt(delta)):
        x3 = x2 + (-2*c)/(b + cmath.sqrt(delta))
    else:
        x3 = x2 + (-2*c)/(b - cmath.sqrt(delta))
    return x1, x2, x3
    
class PolynomialError(Exception):
    def __init__(self, notification):
        self.__notification = notification
    def __str__(self):
        return repr(self.__notification)

class Polynomial:
    
    def __init__(self, coefficients):
        """
            Exercise 1: 
            Initialize a polynomial
        """
        self.__coefficients = simplify(coefficients)
    
    def getCoefficients(self):
        """
            Exercise 1:
            Get coefficients of a polynomial as a list
        """
        return self.__coefficients
    
    def getDegree(self):
        """
            Exercise 2:
            Get degree of a polynomial
        """
        return len(self.__coefficients) - 1
    
    def add(self, P):
        """
            Exercise 3:
            Add 2 polynomials
        """
        min_deg = min(self.getDegree(), P.getDegree())
        coefs_of_sum = []
        for i in range(min_deg + 1):
            coefs_of_sum.append(self.__coefficients[i] + P.__coefficients[i])
        if self.getDegree() > min_deg:
            for i in range(min_deg + 1, self.getDegree() + 1):
                coefs_of_sum.append(self.__coefficients[i])
        elif P.getDegree() > min_deg:
            for i in range(min_deg + 1, P.getDegree() + 1):
                coefs_of_sum.append(P.__coefficients[i])
        return Polynomial(coefs_of_sum)
    
    def getOpposite(self):
        """
            Exercise 4
            Get opposite of a polynomial
        """
        return Polynomial([-x for x in self.__coefficients])

    def substract(self, P):
        """
            Exercise 4
            Substract self and P
        """
        return self.add(P.getOpposite())
    
    def multiply(self, P):
        """
            Exercise 5
            Multiply self and P            
        """
        if self.getDegree() == -1 or P.getDegree() == -1:
            return Polynomial([])
            
        coefs_of_product = [0] * (self.getDegree() + P.getDegree() + 1)
        for i in range(self.getDegree() + 1):
            for j in range(P.getDegree() + 1):
                coefs_of_product[i + j] += self.__coefficients[i] * P.__coefficients[j]
        return Polynomial(coefs_of_product)
    
    def power(self, a):
        """
            Exercise 6
            Power of a polynomial
        """
        if self.getDegree() == -1:
            return Polynomial([])
        
        if a == 0:
            return Polynomial([1])
        
        root = self.power(a/2)
        if a % 2 == 0:
            return root.multiply(root)
        
        else:
            return root.multiply(root).multiply(self)
    
    def divide(self, P):
        """
            Exercise 7
            Divide self by P
        """
        if P.getDegree() == -1:
            raise PolynomialError("Impossible to divide by 0")
            return
            
        remainder = Polynomial(self.__coefficients)
        
        if self.getDegree() < P.getDegree():
            return Polynomial([]), remainder
        
        coefs_quotient = [0] * (self.getDegree() - P.getDegree() + 1)
        quotient = Polynomial([])
        
        while remainder.getDegree() >= P.getDegree():
            coefs_quotient[remainder.getDegree() - P.getDegree()] = float(remainder.__coefficients[remainder.getDegree()]) / P.__coefficients[P.getDegree()]
            quotient = Polynomial(coefs_quotient)
            remainder = self.substract(P.multiply(quotient))

        return quotient, remainder
    
    def isDivisor(self, P):
        """
            Exercise 8
            Check if self is divisor of P
        """
        if self.getDegree() == -1:
            return False
        remainder = P.divide(self)[1]
        return remainder.getDegree() == -1
    
    def getGcd(self, P):
        """
            Exercise 9
            Get great common divisors of self and P
        """
        if self.getDegree() == -1 and P.getDegree() == -1:
            raise PolynomialError("Impossible to get gcd of 2 zero polynomial")
        a = Polynomial(self.__coefficients)
        b = Polynomial(P.__coefficients)
        temp = None
        while b.getDegree() >= 0:
            temp = a.divide(b)[1]
            a = b
            b = temp
        return a.divide(Polynomial([a.__coefficients[a.getDegree()]]))[0]
    
    def evaluate(self, x):
        """
            Exercise 1
        """
        S = 0
        for i in range(self.getDegree() + 1):
            S += self.__coefficients[i] * x**i
        return S
    
    def getRoots(self):
        """
            Exercise 11 & 12
            Get all complex roots of self
        """
        # This part is for exercise 11
        # return []
        
        # This part is for exercise 12
        if self.getDegree() == 0:
            return []
        if self.getDegree() == 1:
            return LinearPolynomial(self.getCoefficients()).getRoots()
        if self.getDegree() == 2:
            return QuadraticPolynomial(self.getCoefficients()).getRoots()
        else:
            current_polynomial = Polynomial(self.getCoefficients())
            roots = []
            
            while current_polynomial.__coefficients[0] == 0:
                roots.append(0)
                current_polynomial.__coefficients = current_polynomial.__coefficients[1:]
                
            while current_polynomial.getDegree() > 2:

                #Initialization
                x = (random.random(), random.random(), random.random())
                while abs(current_polynomial.evaluate(x[2])) > EPSILON:
                    x = (random.random(), random.random(), random.random())
                    nb_iters = 0
                    while (abs(current_polynomial.evaluate(x[2])) > EPSILON or abs(x[2] - x[1]) > TOLERANCE) and nb_iters < MAX_ITERATIONS:
                        x = getNextApproximations(current_polynomial.evaluate, x)
                        nb_iters += 1

                roots.append(x[2])
                
                if abs(x[2].imag) < TOLERANCE:
                    current_polynomial = current_polynomial.divide(Polynomial([-x[2].real, 1]))[0]
                else:
                    roots.append(x[2].conjugate())
                    current_polynomial = current_polynomial.divide(Polynomial([abs(x[2])**2, -2*x[2].real, 1]))[0]
                    
            roots += current_polynomial.getRoots()
            
            for i in range(len(roots)):
                roots[i] = round(roots[i].real, 4) + round(roots[i].imag, 4) * 1j
                if roots[i].imag == 0:
                    roots[i] = roots[i].real
                
            return sorted(roots, key = lambda x: (x.real, x.imag))
    
    def getDerivative(self):
        """
            Exercise 13
        """
        if self.getDegree() <= 0:
            return Polynomial([])
        else:
            deriv_coefs = [0] * (self.getDegree())
            for i in range(1, self.getDegree() + 1):
                deriv_coefs[i - 1] = i * self.__coefficients[i]
        return Polynomial(deriv_coefs)
    
    def getHigherDerivative(self, n):
        """
            Exercise 13
        """
        if n == 1:
            return self.getDerivative()
        return self.getHigherDerivative(self, n-1).getDerivative()
    
    def getRealRoots(self):
        roots = self.getRoots()
        return map(lambda x: x.real, filter(lambda x: abs(x.imag) < EPSILON, roots))
    
    def getStopPoints(self):
        """
            Exercise 13
        """
        deriv = self.getDerivative()
        return sorted(set(deriv.getRealRoots()))
    
    def getFirstNonNullDerivative(self, stoppoint):
        current_polynomial = Polynomial(self.__coefficients)
        res = 0
        value = 0
        for n in range(1, self.getDegree() + 1):
            current_polynomial = current_polynomial.getDerivative()
            value = current_polynomial.evaluate(stoppoint)
            if abs(value) > EPSILON:
                res = n
                break
        return res, value
        
    def getLocalMin(self):
        stoppoints = self.getStopPoints()
        return filter(lambda x: self.getFirstNonNullDerivative(x)[0]%2 == 0 and self.getFirstNonNullDerivative(x)[1] > 0, stoppoints)
        
    def getLocalMax(self):    
        stoppoints = self.getStopPoints()
        return filter(lambda x: self.getFirstNonNullDerivative(x)[0]%2 == 0 and self.getFirstNonNullDerivative(x)[1] < 0, stoppoints)       

#------------------------------------------------------------------------------

class LinearPolynomial(Polynomial):
    def __init__(self, coefficients):
        """
            Exercise 10
        """
        Polynomial.__init__(self, coefficients)
        if self.getDegree() != 1:
            raise PolynomialError("Not a linear polynomial.")
    
    def getRoots(self):
        """
            Exercise 11
            Get roots of a linear polynomial
        """
        return [float(-self.getCoefficients()[0])/self.getCoefficients()[1]]

#------------------------------------------------------------------------------

class QuadraticPolynomial(Polynomial):
    def __init__(self, coefficients):
        """
            Exercise 10
        """
        Polynomial.__init__(self, coefficients)
        if self.getDegree() != 2:
            raise PolynomialError("Not a quadratic polynomial.")
    
    def getRoots(self):
        """
            Exercise 11
            Get roots of a quadratic polynomial
        """
        a, b, c = self.getCoefficients()[2], self.getCoefficients()[1], self.getCoefficients()[0]
        delta = b**2 - 4*a*c
        if delta >= 0:
            roots = sorted([(-b - math.sqrt(delta))/(2*a), (-b + math.sqrt(delta))/(2*a)])
        else:
            roots = sorted([(-b - math.sqrt(-delta)*1j)/(2*a), (-b + math.sqrt(-delta)*1j)/(2*a)], key=lambda x: (x.real, x.imag))
        return roots    

#------------------------------------------------------------------------------

class CubicPolynomial(Polynomial):
    def __init__(self, coefficients):
        """
            Exercise 10
        """
        Polynomial.__init__(self, coefficients)
        if self.getDegree() != 3:
            raise PolynomialError("Not a cubic polynomial.")
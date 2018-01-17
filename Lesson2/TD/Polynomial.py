# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:48:04 2017

@author: ndoannguyen
"""
    
class PolynomialError(Exception):
    def __init__(self, notification):
        self.__notification = notification
    def __str__(self):
        return repr(self.__notification)

class Polynomial:
    
    def __init__(self, coefficients):
        """
            Exercise 1: 
            Initialize a polynomial (self)
        """
        #TODO
    
    def __str__(self):
        """
            Exercise 1
            Print a polynomial in beautiful form.
        """
        #TODO, NOTEST

    
    def getCoefficients(self):
        """
            Exercise 1:
            Get coefficients of a polynomial (self) as a list
        """
        #TODO

    
    def getDegree(self):
        """
            Exercise 2:
            Get degree of a polynomial (self)
        """
        #TODO

    
    def add(self, P):
        """
            Exercise 3:
            Add 2 polynomials: self and P
        """
        #TODO
        

    def substract(self, P):
        """
            Exercise 4
            Substract self and P
        """
        #TODO
        
    
    def multiply(self, P):
        """
            Exercise 5
            Multiply self and P            
        """
        #TODO
        
    
    def power(self, a):
        """
            Exercise 6
            a_th power of a polynomial (self)
        """
        #TODO
        
    
    def divide(self, P):
        """
            Exercise 7
            Divide self by P, return (quotient, remainder)
        """
        #TODO
        
    
    def isDivisor(self, P):
        """
            Exercise 8
            Check if self is divisor of P
        """
        #TODO
        
    
    def getGcd(self, P):
        """
            Exercise 9
            Get great common divisors of self and P
        """
        #TODO
        
    
    def getRoots(self):
        """
            Exercise 11 & 12
            Get all complex roots of self
        """
        #TODO

        
    def getLocalMin(self):
        """
            Exercise 13
            Get all (real) local minima of a polynomial (self)
        """
        #TODO
        
        
    def getLocalMax(self):    
        """
            Exercise 13
            Get all (real) local maxima of a polynomial (self)
        """
        #TODO
        
        
#------------------------------------------------------------------------------

class LinearPolynomial(Polynomial):
    def __init__(self, coefficients):
        """
            Exercise 10
        """
        Polynomial.__init__(self, coefficients)
        #NOT FINISHED. TODO
        
    
    def getRoots(self):
        """
            Exercise 11
            Get roots of a linear polynomial
        """
        #TODO
        

#------------------------------------------------------------------------------

class QuadraticPolynomial(Polynomial):
    def __init__(self, coefficients):
        """
            Exercise 10
        """
        Polynomial.__init__(self, coefficients)
        #NOT FINISHED. TODO
    
    
    def getRoots(self):
        """
            Exercise 11
            Get roots of a quadratic polynomial
        """
        #TODO
        
#------------------------------------------------------------------------------

class CubicPolynomial(Polynomial):
    def __init__(self, coefficients):
        """
            Exercise 10
        """
        Polynomial.__init__(self, coefficients)
        #NOT FINISHED. TODO
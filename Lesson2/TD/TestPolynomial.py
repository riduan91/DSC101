# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 22:05:09 2017

@author: ndoannguyen
"""

from Polynomial_Solutions import Polynomial, PolynomialError, LinearPolynomial, QuadraticPolynomial, CubicPolynomial
import sys

TOLERANCE = 0.01

def compareOK(P, lQ):
    coefs = P.getCoefficients()
    if len(coefs) != len(lQ):
        print "Expected: ", lQ
        print "Received: ", coefs
        return False
    errors = sum(abs(coefs[i] - lQ[i]) for i in range(len(coefs)))
    if errors > TOLERANCE:
        print "Expected: ", lQ
        print "Received: ", coefs
        return False
    return True
    

P = [[0, 0], [1, 0, 0], [-4.5, 2.5, 0, 0, 3], [1, 2, 4.5, 0, 0, 0]]
Q = [[1, 2, 3], [2, -1, 3], [4.5, -2.5, 0, 0, -3], [2, -1, -4.5]]

R = [[1, 1], [1, 1], [], [-1], [-2]]
power = [2, 5, 100, 20, 11]

D1 = [[3, 2, 1], [-1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1], [7, 3, 3, 2], [1, 1, 1]]
E = [[1, 1, 1], [-2, 2], [5, -5], [0.5, -2], [2, 3, 1, 1]]

F = [[-16, 0, 0, 0, 0, 0, 0, 0, 1], [1, -1, 0, 0, 0, 1]]
G = [[0, 0, 0, 0, 4, 0, 0, 2, -1], [0, 0, 0, 1], [0, 0, 0, 0, -1], [1, 2, -3, 1, -4, 1]]

Test1Result = [[], [1], [-4.5, 2.5, 0, 0, 3], [1, 2, 4.5]]
Test2Result = [-1, 0, 4, 2]
Test3Result = [[1, 2, 3], [3, -1, 3], [], [3, 1]]
Test4Result = [[-1, -2, -3], [-1, 1, -3], [-9, 5, 0, 0, 6], [-1, 3, 9]]
Test5Result = [[], [2, -1, 3], [-20.25, 22.5, -6.25, 0, 27, -15, 0, 0, -9], [2, 3, 2.5, -13.5, -20.25]]
Test6Result = [[1, 2, 1], [1, 5, 10, 10, 5, 1], [], [1], [-2048]]
Test7ResultA = [[1], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2], [-1.9375, -1.75, -1], []]
Test7ResultB = [[2, 1], [], [2], [3.98438*2], [1, 1, 1]]
Test8Result = [False, True, False, False, False]
Test9Result = [[1], [-1, 1], [1], [1], [1]]
Test10Result = [2, 1, 1, 1, 3]
Test11Result = [[-0.5 - 0.86602j, -0.5 + 0.86602j], [1], [1], [0.25], []]
Test12Result = [[-1.414, -1-1j, -1+1j, -1.414j, 1.414j, 1-1j, 1+1j, 1.414], [-1.1673, -0.1812 - 1.0839j, -0.1812 + 1.0839j, 0.7648 - 0.3524j, 0.7648 + 0.3524j]]
Test13ResultA = [[0], [], [], [3.1178]]
Test13ResultB = [[-0.9094, 2.0], [], [0], [0.3097]]

def test0():
    print "Test 0 OK"
        
def test1():
    """
        Test for initialization and getCoefficients()
    """
    for lp, res in zip(P, Test1Result):
        p = Polynomial(lp)
        print "Initializing ", lp
        if not compareOK(p, res):
            print "Test 1 not OK"
            return
        else:
            print "OK"
    print "Test 1 OK"
    return
    
def test2():
    """
        Test for degree
    """
    for lp, res in zip(P, Test2Result):
        p = Polynomial(lp)
        deg = p.getDegree()
        print "Finding degree of ", lp
        if deg != res:
            print "Expected: " + str(res)
            print "Received: " + str(deg)
            print "Test 2 not OK"
            return
        else:
            print "OK"
    print "Test 2 OK"
    return
    
def test3():
    """
        Test for addition
    """
    for lp, lq, res in zip(P, Q, Test3Result):
        p = Polynomial(lp)
        q = Polynomial(lq)
        print lp, " + ", lq
        r = p.add(q)
        if not compareOK(r, res):
            print "Test 3 not OK"
            return
        else:
            print "OK"
    print "Test 3 OK"
    return

def test4():
    """
        Test for substraction
    """
    for lp, lq, res in zip(P, Q, Test4Result):
        p = Polynomial(lp)
        q = Polynomial(lq)
        print lp, " - ", lq
        r = p.substract(q)
        if not compareOK(r, res):
            print "Test 4 not OK"
            return
        else:
            print "OK"
    print "Test 4 OK"
    return

def test5():
    """
        Test for multiplication
    """
    for lp, lq, res in zip(P, Q, Test5Result):
        p = Polynomial(lp)
        q = Polynomial(lq)
        print lp, " * ", lq
        r = p.multiply(q)
        if not compareOK(r, res):
            print "Test 5 not OK"
            return
        else:
            print "OK"
    print "Test 5 OK"
    return

def test6():
    """
        Test for power
    """
    for lp, lq, res in zip(R, power, Test6Result):
        p = Polynomial(lp)
        print lp, " ^ ", lq
        r = p.power(lq)
        if not compareOK(r, res):
            print "Test 6 not OK"
            return
        else:
            print "OK"
    print "Test 6 OK"
    return
    
def test7():
    """
        Test for division
    """
    p = Polynomial([1])
    q = Polynomial([0])
    print "[1] / []"
    try:
        q.divide(q)
    except PolynomialError:
        print "OK"
        pass
    for lp, lq, res1, res2 in zip(D1, E, Test7ResultA, Test7ResultB):
        p = Polynomial(lp)
        q = Polynomial(lq)
        print lp, " / ", lq
        r1, r2 = p.divide(q)
        if not compareOK(r1, res1) or not compareOK(r2, res2):
            print "Test 7 not OK"
            return
        else:
            print "OK"
    print "Test 7 OK"
    return

def test8():
    """
        Test for divisibility
    """
    for lp, lq, res in zip(E, D1, Test8Result):
        p = Polynomial(lp)
        q = Polynomial(lq)
        print lp, " is divisible by ", lq, "?"
        if p.isDivisor(q) != res:
            print "Expected: ", res
            print "Received: ", p.isDivisor(q)
            print "Test 8 not OK"
            return
        else:
            print "OK"
    print "Test 8 OK"
    return

def test9():
    """
        Test gcd
    """
    for lp, lq, res in zip(E, D1, Test9Result):
        p = Polynomial(lp)
        q = Polynomial(lq)
        print "gcd(", lp, ", ", lq, ")"
        r = p.getGcd(q)
        if not compareOK(r, res):
            print "Test 9 not OK"
            return
        else:
            print "OK"
    print "Test 9 OK"
    return

def test10():
    """
        Test subclass
    """
    for lp, res in zip(E, Test10Result):
        
        print lp
            
        if res == 1:
            try:
                p = LinearPolynomial(lp)
            except:
                print "Class LinearPolynomial not OK"
                print "Test 10 not OK"
                return
            try:
                p = QuadraticPolynomial(lp)
            except PolynomialError:
                pass
            try:
                p = CubicPolynomial(lp)
            except PolynomialError:
                pass
                    
            if p.getDegree() != 1:
                print "Class LinearPolynomial not OK"
                print "Test 10 not OK"
                return
            print "OK"
        
        elif res == 2:
            try:
                p = QuadraticPolynomial(lp)
            except:
                print "Class QuadraticPolynomial not OK"
                print "Test 10 not OK"
                return
            try:
                p = LinearPolynomial(lp)
            except PolynomialError:
                pass
            try:
                p = CubicPolynomial(lp)
            except PolynomialError:
                pass
            if p.getDegree() != 2:
                print "Class LinearPolynomial not OK"
                print "Test 10 not OK"
                return
            print "OK"
        
        elif res == 3:
            try:
                p = CubicPolynomial(lp)
            except:
                print "Class CubicPolynomial not OK"
                print "Test 10 not OK"
                return
            try:
                p = LinearPolynomial(lp)
            except PolynomialError:
                pass
            try:
                p = QuadraticPolynomial(lp)
            except PolynomialError:
                pass
            if p.getDegree() != 3:
                print "Class LinearPolynomial not OK"
                print "Test 10 not OK"
                return
            print "OK"           
        
        else:
            try:
                p = CubicPolynomial(lp)
            except:
                pass
            try:
                p = LinearPolynomial(lp)
            except PolynomialError:
                pass
            try:
                p = QuadraticPolynomial(lp)
            except PolynomialError:
                pass
    
    print "Test 10 OK"
    return

def test11():
    """
        Test getRoots
    """
    for lp, res in zip(E, Test11Result):
        p = Polynomial(lp)
        if p.getDegree() == 1:
            p = LinearPolynomial(lp)
        elif p.getDegree() == 2:
            p = QuadraticPolynomial(lp)
        elif p.getDegree() == 3:
            p = CubicPolynomial(lp)
        print "Roots of ", lp
        r = Polynomial(p.getRoots())
        if not compareOK(r, res):
            print "Test 11 not OK"
            return
        else:
            print "OK"
    print "Test 11 OK"
    return
    
def test12():
    """
        Test getRoots
    """
    for lp, res in zip(F, Test12Result):
        p = Polynomial(lp)
        print "Roots of ", lp
        r = Polynomial(p.getRoots())
        if not compareOK(r, res):
            print "Test 12 not OK"
            return
        else:
            print "OK"
    print "Test 12 OK"
    return

def test13():
    """
        Test for local extrema
    """

    for lp, res1, res2 in zip(G, Test13ResultA, Test13ResultB):
        p = Polynomial(lp)
        print "Extrema of ", lp
        r1 = p.getLocalMin()
        for i in range(len(r1)):
            r1[i] = round(r1[i], 4)
        
        r2 = p.getLocalMax()
        for i in range(len(r2)):
            r2[i] = round(r2[i], 4)
        if r1 != res1 or r2 != res2:
            print "Expected: ", res1, res2
            print "Received: ", r1, r2
            print "Test 13 not OK"
            return
        else:
            print "OK"
    print "Test 13 OK"
    return

Tests = [test0, test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12, test13]

#MAIN FUNCTION
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Please configure your test by Run -> Configure"
    else:
        for i in range(len(Tests) + 1):
            if sys.argv[1] == "test_" + str(i):
                Tests[i]()
                break
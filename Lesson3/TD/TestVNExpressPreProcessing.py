# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 15:44:24 2018

@author: ndoannguyen
"""

import sys, os, shutil
from VNExpressPreProcessing import prepareDataFolder

def test0():
    print("Test 0 OK.")

def test1():
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("Test 1 not OK. Please reinstall beautifulsoup4.")
    else:
        print("Test 1 OK.")

def test2():
    if (os.path.isfile("clrscr")): 
        os.remove("clrscr")
    elif (os.path.isdir("clrscr")):
        shutil.rmtree("clrscr")

    try:
        prepareDataFolder("clrscr")
    except Exception as e:
        print("Test 2 not OK.")
        print(e)
    else:
        if not (os.path.isdir("clrscr")):
            print("Test 2 not OK.")
            print("The folder has not been created.")
        else:
            os.rmdir("clrscr")
            print("Test 2 OK.")
    

Tests = [test0, test1, test2]

#MAIN FUNCTION
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Please configure your test by Run -> Configure"
    else:
        for i in range(len(Tests) + 1):
            if sys.argv[1] == "test_" + str(i):
                Tests[i]()
                break
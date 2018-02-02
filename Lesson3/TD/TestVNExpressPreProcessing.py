# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 15:44:24 2018

@author: ndoannguyen
"""

import sys, os, shutil
from VNExpressPreProcessing import prepareDataFolder, dayToTimestamp, downloadFirstTitlePage

#CATEGORIES
THOI_SU = 1001005 
GIA_DINH = 1002966 
SUC_KHOE = 1003750 
THE_GIOI = 1001002 
KINH_DOANH = 1003159 
GIAI_TRI = 1002691 
THE_THAO = 1002565 
PHAP_LUAT = 1001007 
GIAO_DUC = 1003497 
DU_LICH = 1003231 
KHOA_HOC = 1001009 
SO_HOA = 1002592 
XE = 1001006 
CONG_DONG = 1001012 
TAM_SU = 1001014

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
    
def test3():
    s = dayToTimestamp("21/01/2018")
    if not isinstance(s, int) and not isinstance(s, long):
        print("Please convert your result to type int.")
        print("Test 3 not OK")
    else:
        if s != 1516489200:
            print("Test 3 not OK. 21/01/2018 should be converted to 1516489200.")
        else:
            print("Test 3 OK.")

def test4():
    s = downloadFirstTitlePage(THOI_SU, "01/01/2018", "02/01/2018")
    if s.find("Tài xế trả tiền xu") <= 0:
        print("Page not successfully downloaded. Test 4 not OK")
    else:
        print("Test 4 OK")

def test5():
    pass


Tests = [test0, test1, test2, test3, test4, test5]

#MAIN FUNCTION
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Please configure your test by Run -> Configure"
    else:
        for i in range(len(Tests) + 1):
            if sys.argv[1] == "test_" + str(i):
                Tests[i]()
                break
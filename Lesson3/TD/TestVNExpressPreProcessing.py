# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 15:44:24 2018

@author: ndoannguyen
"""

import sys, os, shutil
import pandas as pd
from VNExpressPreProcessing_Solution import prepareDataFolder, dayToTimestamp, downloadFirstTitlePage, getLinksFromTitlePage, downloadTitlePage, \
    saveLinksFromTitlePages, downloadArticle, getComponents, saveArticles, readContent, addAuthorColumn, getSimpleWordFrequency

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
    print(s.find("Tài xế trả tiền xu"))
    if s.find("Tài xế trả tiền xu") <= 0:
        print("Page not successfully downloaded. Test 4 not OK")
    else:
        print("Test 4 OK")

def test5():
    s = downloadFirstTitlePage(KHOA_HOC, "01/01/2018", "31/01/2018")
    u = getLinksFromTitlePage(s)
    if 'https://vnexpress.net/tin-tuc/khoa-hoc/hoi-dap/day-la-cu-gi-3706123.html' in u:
        print("Test 5 OK")
    else:
        print("Test 5 not OK.")

def test6():
    s = downloadTitlePage(11, KHOA_HOC, "01/01/2018", "31/01/2018")
    u = getLinksFromTitlePage(s)
    if 'https://vnexpress.net/tin-tuc/khoa-hoc/giai-ma/vien-da-chua-kim-cuong-ngoai-hanh-tinh-3696712.html' in u:
        print("Test 6 OK")
    else:
        print("Test 6 not OK.")

def test7():
    prepareDataFolder("Data")
    saveLinksFromTitlePages(KHOA_HOC, "01/01/2018", "05/01/2018", "Science_Jan2018_Titles.txt", "Data")
    if not os.path.isdir("Data") or not os.path.isfile("Data/Science_Jan2018_Titles.txt"):
        print("Test 7 not OK. Please make sure the file is in the folder.")
    else:
        pass

def test8():
    s = downloadArticle("https://vnexpress.net/tin-tuc/khoa-hoc/hoi-dap/can-bao-nhieu-son-de-phu-het-be-mat-trai-dat-3692631.html")
    if s.find("để phủ hết") > 0:
        print("Test 8 OK.")
    else:
        print("Test 8 not OK. Review your download function.")

def test9():
    s = downloadArticle("https://vnexpress.net/tin-tuc/khoa-hoc/nhen-tho-san-gia-chet-van-bi-ong-bap-cay-truy-giet-3694170.html")
    r = getComponents(s)
    if r[0].find("Nh") < 0 or r[1].find("Chi") < 0 or r[2][4].find(" Hoa") < 0:
        print("Test 9 not OK.")
    elif r[0].find("\t") >= 0 or r[1].find("\t") >= 0 or r[2][4].find("\t") >= 0:
        print("Test 9 not OK. Your strings are not very clean. Please remove all \\t character")
    else:
        print("Test 9 OK.")

def test10():
    prepareDataFolder("Data")
    saveLinksFromTitlePages(KHOA_HOC, "01/01/2018", "02/01/2018", "Science_Jan2018_Titles.txt", "Data")
    saveArticles("Science_Jan2018_Titles.txt", "Science_Jan2018_Articles.txt", "Data")
    f = open("Data/Science_Jan2018_Articles.txt").readlines()
    if len(f) >= 7:
        print("Test 10 OK.")
    else:
        print("Test 10 not OK.")

def test11():
    if not os.path.isfile("Data/Science_Jan2018_Articles.txt"):
        prepareDataFolder("Data")
        saveLinksFromTitlePages(KHOA_HOC, "01/01/2018", "02/01/2018", "Science_Jan2018_Titles.txt", "Data")
        saveArticles("Science_Jan2018_Titles.txt", "Science_Jan2018_Articles.txt", "Data")
    f = open("Data/Science_Jan2018_Articles.txt").readlines()
    if len(f) < 7:
        prepareDataFolder("Data")
        saveLinksFromTitlePages(KHOA_HOC, "01/01/2018", "02/01/2018", "Science_Jan2018_Titles.txt", "Data")
        saveArticles("Science_Jan2018_Titles.txt", "Science_Jan2018_Articles.txt", "Data")
    r = readContent("Science_Jan2018_Articles.txt", "Data")
    if len(r.iloc[0, 0]) > 0 or  len(r.iloc[0, 1]) > 0 or  len(r.iloc[0, 2]) > 0:
        print("Test 11 OK")
    else:
        print("Test 11 not OK")

def test12():
    if not os.path.isfile("Data/Science_Jan2018_Articles.txt"):
        prepareDataFolder("Data")
        saveLinksFromTitlePages(KHOA_HOC, "01/01/2018", "02/01/2018", "Science_Jan2018_Titles.txt", "Data")
        saveArticles("Science_Jan2018_Titles.txt", "Science_Jan2018_Articles.txt", "Data")
    f = open("Data/Science_Jan2018_Articles.txt").readlines()
    if len(f) < 7:
        prepareDataFolder("Data")
        saveLinksFromTitlePages(KHOA_HOC, "01/01/2018", "02/01/2018", "Science_Jan2018_Titles.txt", "Data")
        saveArticles("Science_Jan2018_Titles.txt", "Science_Jan2018_Articles.txt", "Data")
    r = readContent("Science_Jan2018_Articles.txt", "Data")
    s = addAuthorColumn(r)
    tester = False
    for e in s.loc[:, "author"]:
        if e.find("Phương Hoa") >= 0:
            tester = True
    if tester:
        print("Test 12 OK.")
    else:
        print("Test 12 not OK.")

def test13():
    if not os.path.isfile("Data/Science_Jan2018_Articles.txt"):
        prepareDataFolder("Data")
        saveLinksFromTitlePages(KHOA_HOC, "01/01/2018", "02/01/2018", "Science_Jan2018_Titles.txt", "Data")
        saveArticles("Science_Jan2018_Titles.txt", "Science_Jan2018_Articles.txt", "Data")
    f = open("Data/Science_Jan2018_Articles.txt").readlines()
    if len(f) < 7:
        prepareDataFolder("Data")
        saveLinksFromTitlePages(KHOA_HOC, "01/01/2018", "02/01/2018", "Science_Jan2018_Titles.txt", "Data")
        saveArticles("Science_Jan2018_Titles.txt", "Science_Jan2018_Articles.txt", "Data")
    r = readContent("Science_Jan2018_Articles.txt", "Data")
    if getSimpleWordFrequency(r)["và"] >= 1:
        print("Test 13 OK.")
    else:
        print("Test 13 not OK.")

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
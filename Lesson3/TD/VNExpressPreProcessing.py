# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 15:42:25 2018

@author: ndoannguyen
"""

import os, time, urllib

# Exercise 1
from bs4 import BeautifulSoup

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

BASE_URL = "http://vnexpress.net/category/day/"

def prepareDataFolder(data_folder):
    # Exercise 2
    # TODO
    if (os.path.isfile(data_folder)):
        print("A file named '" + data_folder + "' already existed." )
    elif (os.path.isdir(data_folder)):
        print("Folder '" + data_folder + "' already existed." )
    else:
        os.mkdir(data_folder)

def dayToTimestamp(day):
    # Exercise 3
    # TODO
    return int(time.mktime(time.strptime(day, "%d/%m/%Y")))

def downloadFirstTitlePage(category, fromdate, todate):
    # Exercise 4
    # TODO
    url = "%s?cateid=%d&fromdate=%d&todate=%d&allcate=%d|%d|" \
        % (BASE_URL, category, dayToTimestamp(fromdate), dayToTimestamp(todate), category, category)
    return urllib.urlopen(url).read()

def 
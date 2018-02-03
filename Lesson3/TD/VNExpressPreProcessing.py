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

def dayToTimestamp(day):
    # Exercise 3
    # TODO

def downloadFirstTitlePage(category, fromdate, todate):
    # Exercise 4
    # TODO

def getLinksFromTitlePage(page_content):
    # Exercise 5
    # To be modify
    tree = BeautifulSoup(page_content, "lxml")
    articles = tree.find('section', 'sidebar_1').find_all('h3', 'title_news')
    for article in articles:
        href = article.find('a')['href']
        print(href)
    
def downloadTitlePage(index, category, fromdate, todate):
    # Exercise 6
    # TODO


def saveLinksFromTitlePages(category, fromdate, todate, title_file, data_folder):
    # Exercise 7
    # TODO


def downloadArticle(link):
    # Exercise 8
    # TODO
    

def getComponents(article):
    # Exercise 9
    # To be modify
    tree = BeautifulSoup(article, "lxml")
    sidebar = tree.find('section', 'sidebar_1')
    title_news = sidebar.find('h1', 'title_news_detail').contents[0]
    #title_news = clean(title_news)
    short_intro = sidebar.find('h2', 'description').contents[0]
    #short_intro = clean(short_intro)
    inside_content = sidebar.find_all('p', 'Normal')
    print title_news
    print short_intro
    print inside_content


def saveArticles(title_file, content_file, data_folder):
    # Exercise 10
    # TODO

def readContent(content_file, data_folder):
    # Exercise 11
    # TODO

def addAuthorColumn(articles_table):
    # Exercise 12
    # TODO

def getSimpleWordFrequency(articles_table):
    # Exercise 13
    # TODO
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 15:42:25 2018

@author: ndoannguyen
"""

import os, time, urllib, re
import pandas as pd

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

def getLinksFromTitlePage(page_content):
    # Exercise 5
    # To be modify
    url_list = []
    tree = BeautifulSoup(page_content, "lxml")
    articles = tree.find('section', 'sidebar_1').find_all('h3', 'title_news')
    for article in articles:
        href = article.find('a')['href']
        url_list.append(href)
    return url_list
    
def downloadTitlePage(index, category, fromdate, todate):
    # Exercise 6
    # TODO
    url = "%spage/%d.html?cateid=%d&fromdate=%d&todate=%d&allcate=%d|%d|" \
        % (BASE_URL, index, category, dayToTimestamp(fromdate), dayToTimestamp(todate), category, category)
    return urllib.urlopen(url).read()

def saveLinksFromTitlePages(category, fromdate, todate, title_file, data_folder):
    # Exercise 7
    # TODO
    index = 1
    result = []
    while True:
        print("Downloading page %d" % index)
        s = downloadTitlePage(index, category, fromdate, todate)
        u = getLinksFromTitlePage(s)
        if len(u) == 0:
            break
        else:
            result += u
            index += 1
    myfile = open(data_folder + "/" + title_file, 'w')
    myfile.write("\n".join(result))
    myfile.close()

def downloadArticle(link):
    # Exercise 8
    # To do
    return urllib.urlopen(link).read()

def getComponents(article):
    # Exercise 9
    # To be modify
    tree = BeautifulSoup(article, "lxml")
    sidebar = tree.find('section', 'sidebar_1')
    title_news = sidebar.find('h1', 'title_news_detail').contents[0]
    title_news = clean(title_news)
    short_intro = sidebar.find('h2', 'description').contents[0]
    short_intro = clean(short_intro)
    inside_content = sidebar.find_all('p', 'Normal')
    """
    print title_news
    print short_intro
    print inside_content
    """
    contents = []
    for content in inside_content:
        quasi_paragraphs = content.contents
        strings = []
        for quasi_paragraph in quasi_paragraphs:
            if type(quasi_paragraph) != 'str':
                quasi_paragraph = quasi_paragraph.string
                if quasi_paragraph == None:
                    continue
            strings.append(quasi_paragraph)
        string = " ".join(strings)
        string = clean(string)
        if len(string) > 0:
            contents.append(string)
    return title_news, short_intro, contents 

def clean(string):
    pattern = re.compile(r'[\n\t\r]')
    string = pattern.sub("", string)
    string = re.sub(' +', ' ', string)
    if len(string) > 0 and string[0] == ' ':
        string = string[1:]
    if len(string) > 0 and string[-1] == ' ':
        string = string[:-1]
    return string

def saveArticles(title_file, content_file, data_folder):
    # Exercise 10
    # TODO
    f = open(data_folder + "/" + title_file, 'r')
    titles = f.read().split("\n")
    g = open(data_folder + "/" + content_file, 'w')
    for title in titles:
        try:
            print("Downloading %s", title)
            article = downloadArticle(title)
            title_news, short_intro, contents = getComponents(article)
            res = ""
            if title_news != None and short_intro != None and contents != None and len(contents) > 0:
                res = title_news + "\t\t"
                res += short_intro + "\t\t"
                res += "\t".join(contents)
                res += "\n"
            g.write(res.encode("utf-8"))
        except:
            pass
    g.close()
    f.close()

def readContent(content_file, data_folder):
    # Exercise 11
    # TODO
    return pd.read_csv(data_folder + "/" + content_file, header = None, sep="\t\t", engine="python")

def getAuthor(content):
    author = content[content.rfind("\t") + 1:]
    if len(author) >= 20:
        author = ""
    return author

def addAuthorColumn(articles_table):
    # Exercise 12
    # TODO
    articles_table.insert(len(articles_table.columns), "author", map(lambda res: getAuthor(res), articles_table.iloc[:, 2]))
    return articles_table

def getSimpleWordFrequency(articles_table):
    # Exercise 13
    # TODO
    word_freq = {}
    for data in articles_table.iloc[:, 2]:
        # A very naive method
        data = data.replace(".", "").replace(",", "").replace(":", "").replace(";", "").replace("!", "").replace("?", "").replace("”", "").replace("\"", "").replace("'", "").replace("\t", "")
        data = data.replace("0", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "").replace("6", "").replace("7", "").replace("8", "").replace("9", "")
        words = data.split(" ")
        for word in words:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    del(word_freq[""])
    return word_freq


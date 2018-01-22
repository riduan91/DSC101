# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 18:43:30 2017

@author: ndoannguyen
"""

import os, urllib, re
from bs4 import BeautifulSoup
import time
from multiprocessing import Pool

DATA_DIR = "../../FrequencyData/"
TMP_DIR = "tmp"

CAT_ID = {          "THOI_SU":          1001005, 
                    "GIA_DINH":         1002966, 
                    "SUC_KHOE":         1003750, 
                    "THE_GIOI":         1001002, 
                    "KINH_DOANH":       1003159, 
                    "GIAI_TRI":         1002691, 
                    "THE_THAO":         1002565, 
                    "PHAP_LUAT":        1001007, 
                    "GIAO_DUC":         1003497, 
                    "DU_LICH":          1003231, 
                    "KHOA_HOC":         1001009, 
                    "SO_HOA":           1002592, 
                    "XE":               1001006, 
                    "CONG_DONG":        1001012, 
                    "TAM_SU":           1001014
        }

URL_MODEL = {}
FROMDATE = 1500069600
TODATE = 1500153144
BASE_URL = "http://vnexpress.net/category/day/"
VIDEO_PREFIX = "video.vnexpress.net"

CAT_URL = {}

def prepare_stockage():
    listdir = os.listdir(DATA_DIR)
    if TMP_DIR not in listdir:
        os.mkdir(DATA_DIR + TMP_DIR)

def download_page(url):
    return urllib.urlopen(url)

def get_article_url(fromdate, todate, categories):
    article_urls = []
    if categories == "all":
        for cat_name in CAT_ID.keys():
            article_urls += get_article_url(fromdate, todate, cat_name)
    else:
        cat_name = categories
        print("Category %s" % cat_name)
        has_next = True
        cat_url = "%s?cateid=%d&fromdate=%d&todate=%d&allcate=%d|%d|" % (BASE_URL, CAT_ID[cat_name], fromdate, todate, CAT_ID[cat_name], CAT_ID[cat_name])          
        page_counter = 0        
        while has_next:
            page_counter += 1
            print("Searching in page %d: %s" % (page_counter, cat_url))
            f = download_page(cat_url).read()
            tree = BeautifulSoup(f)
            articles = tree.find('section', 'sidebar_1').find_all('h3', 'title_news')
            for article in articles:
                href = article.find('a')['href']
                if href not in article_urls and href.find(VIDEO_PREFIX) == -1:
                    article_urls.append(href)
            next_page = tree.find('a', 'next')
            has_next = next_page != None
            if has_next:
                cat_url = "http://vnexpress.net" + next_page['href']
    return article_urls

def store_article_url(time_string_1, time_string_2, filename, categories = "all"):
    print("Searching for links of all articles between %s and %s, category %s..." % (time_string_1, time_string_2, categories))
    fromdate = int(time.mktime(time.strptime(time_string_1, "%d/%m/%y")))
    todate = int(time.mktime(time.strptime(time_string_2, "%d/%m/%y"))) + 24 * 60 * 60
    articles = get_article_url(fromdate, todate, categories)
    print("Saving links into %s." % (DATA_DIR + filename))
    f = open(DATA_DIR + filename, 'w')
    f.write("\n".join(articles))
    f.close()

def get_article_content(article):
    f = download_page(article).read()
    tree = BeautifulSoup(f)
    sidebar = tree.find('section', 'sidebar_1')
    title_news = sidebar.find('h1', 'title_news_detail').contents[0]
    title_news = clean_title(title_news)
    short_intro = sidebar.find('h2', 'description').contents[0]
    short_intro = clean_short_intro(short_intro)
    normal = sidebar.find_all('p', 'Normal')
    contents = []
    for content in normal:
        quasi_paragraphs = content.contents
        strings = []
        for quasi_paragraph in quasi_paragraphs:
            if type(quasi_paragraph) != 'str':
                quasi_paragraph = quasi_paragraph.string
                if quasi_paragraph == None:
                    continue
            strings.append(quasi_paragraph)
        string = " ".join(strings)
        string = clean_content(string)
        if len(string) > 0:
            contents.append(string)
    return title_news, short_intro, contents        

def store_articles_content(titlefile, datafile):
    f = open(DATA_DIR + titlefile, 'r')
    titles = f.read().split("\n")
    g = open(DATA_DIR + datafile, 'w')
    p = Pool(8)
    contents = p.map(partially_store_articles_content, titles)
    g.write("\n".join(contents).encode('utf-8'))   
    g.close()
    f.close()

def partially_store_articles_content(title):
    try:
        title_news, short_intro, contents = get_article_content(title)
        res = ""
        if title_news != None:
            res = title_news + "\t\t"
            res += short_intro + "\t\t"
            for content in contents:
                res += content + "\t"
        return res
    except:
        pass
        return ""
    
def clean_title(string):
    pattern = re.compile(r'[\n\t\r]')
    string = pattern.sub("", string)
    string = re.sub(' +', ' ', string)
    if string[0] == ' ':
        string = string[1:]
    if string[-1] == ' ':
        string = string[:-1]
    return string

def clean_short_intro(string):
    return clean_title(string)
   
def clean_content(string):
    pattern = re.compile(r'[\n\t\r]')
    string = pattern.sub("", string)
    string = re.sub(' +', ' ', string)
    if len(string) > 0 and string[0] == ' ':
        string = string[1:]
    if len(string) > 0 and string[-1] == ' ':
        string = string[:-1]
    return string

#prepare_stockage()

"""
store_article_url(1500069600, 1500153144, "articles_1507.txt")
store_articles_content("articles_1507.txt" , "data_1507.txt")
store_article_url(1499983200, 1500066744, "articles_1407.txt")
store_articles_content("articles_1407.txt" , "data_1407.txt")
"""

#store_article_url("01/01/17", "31/01/17", "SportTitles_Jan2017.txt", "Thể thao")
#store_articles_content("SportTitles_Jan2017.txt", "Sport_Jan2017.txt")
begin = ["01/01/17", "01/02/17", "01/03/17", "01/04/17", "01/05/17", "01/06/17", "01/07/17", "01/08/17", "01/09/17", "01/10/17", "01/11/17", "01/12/17"]
end = ["31/01/17", "28/02/17", "31/03/17", "30/04/17", "31/05/17", "30/06/17", "31/07/17", "31/08/17", "30/09/17", "31/10/17", "30/11/17", "31/12/17"]
CatE = "KHOA_HOC"
cate = "Science"
mth = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

if __name__ == '__main__':
    for i in range(0, 12):
        t1 = time.time()
        store_article_url(begin[i], end[i], cate + "Titles_" + mth[i] + "2017.txt", CatE)
        t2 = time.time()
        print("Step 1 took " + str(t2 - t1) + " seconds.")
        store_articles_content(cate + "Titles_" + mth[i] + "2017.txt", cate + "_" + mth[i] + "2017.txt")
        print("Step 2 took " + str(time.time() - t2) + " seconds.")
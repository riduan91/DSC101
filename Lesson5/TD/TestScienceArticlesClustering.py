# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 16:06:45 2018

@author: ndoannguyen
"""

import sys, pandas, os
from ScienceArticlesClustering import concatenateDataFiles, paragraphToSentences, sentenceToSegments

RAW_DATA_FOLDER = "RawData/"
ALL_DATA_FOLDER = "FullData/"
ALL_DATA_FILE = "FullData/Science2017.txt"
WORDLIST_FILE = "FullData/VietnameseWords.txt"

def test0():
    print "Test 0 OK."

def test1():
    concatenateDataFiles(RAW_DATA_FOLDER, ALL_DATA_FOLDER, ALL_DATA_FILE)
    s = pandas.read_csv(ALL_DATA_FILE, sep="\t\t", header=None)
    if len(s) >= 4000:
        for i in range(len(s)):
            if len(s.iloc[i, :]) != 3:
                print("Test 1 not OK.")
                return
        print("Test 1 OK.")
        return
    else:
        print("Test 1 not OK.")
        return

def test2():
    if not os.path.isfile(ALL_DATA_FILE):
        concatenateDataFiles(RAW_DATA_FOLDER, ALL_DATA_FOLDER, ALL_DATA_FILE)
    else:
        s = pandas.read_csv(ALL_DATA_FILE, sep="\t\t", header=None).iloc[1, 2].split("\t")[0]
        r = paragraphToSentences(s)
        if not isinstance(r, list):
            print("Test 2 not OK.")
            return
        for u in r:
            if ". " in u:
                print("Test 2 not OK.")
                return
        print("Test 2 OK.")

def test3():
    if not os.path.isfile(ALL_DATA_FILE):
        concatenateDataFiles(RAW_DATA_FOLDER, ALL_DATA_FOLDER, ALL_DATA_FILE)
    else:
        s = pandas.read_csv(ALL_DATA_FILE, sep="\t\t", header=None).iloc[1, 2].split("\t")[0]
        r = paragraphToSentences(s)[1]
        u = sentenceToSegments(r)
        if not isinstance(u, list):
            print("Test 3 not OK.")
            return
        for u in r:
            if ", " in u:
                print("Test 3 not OK.")
                return
        print("Test 3 OK.")


Tests = [test0, test1, test2, test3]

#MAIN FUNCTION
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Please configure your test by Run -> Configure"
    else:
        for i in range(len(Tests) + 1):
            if sys.argv[1] == "test_" + str(i):
                Tests[i]()
                break
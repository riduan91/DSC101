# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 16:06:45 2018

@author: ndoannguyen
"""

import sys, pandas, os
from ScienceArticlesClustering import concatenateDataFiles, paragraphToSentences, sentenceToSegments, segmentToUnits, createWordSet, unitsToWords

RAW_DATA_FOLDER = "RawData/"
ALL_DATA_FOLDER = "FullData/"
ALL_DATA_FILE = "FullData/Science2017.txt"
WORDLIST_FILE = "VietnameseDictionary/WordList.txt"
FREQUENCY_FILE = "VietnameseDictionary/FrequencyByArticle.txt"

def test0():
    print "Test 0 OK."

def test1():
    concatenateDataFiles(RAW_DATA_FOLDER, ALL_DATA_FOLDER, ALL_DATA_FILE)
    my_file = open(ALL_DATA_FILE)
    s = my_file.read().split("\n")
    s = [line.split("\t\t") for line in s]
    print(len(s))
    if len(s) >= 3000:
        for i in range(len(s)):
            if len(s[i]) != 3:
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
        for v in r:
            if ", " in v:
                print("Test 3 not OK.")
                return
        print("Test 3 OK.")

def test4():
    if not os.path.isfile(ALL_DATA_FILE):
        concatenateDataFiles(RAW_DATA_FOLDER, ALL_DATA_FOLDER, ALL_DATA_FILE)
    else:
        s = pandas.read_csv(ALL_DATA_FILE, sep="\t\t", header=None).iloc[1, 2].split("\t")[0]
        r = paragraphToSentences(s)[1]
        u = sentenceToSegments(r)[0]
        v = segmentToUnits(u)
        if not isinstance(v, list):
            print("Test 4 not OK.")
            return
        for w in v:
            if " " in w:
                print("Test 4 not OK.")
                return
        print("Test 4 OK.")

def test5():
    word_set = createWordSet(WORDLIST_FILE)
    if not isinstance(word_set, set) or len(word_set) < 29000 or "nhân dân" not in word_set:
        print("Test 5 not OK")
        return
    print("Test 5 OK.")

def test6():
    word_set = createWordSet("VietnameseDictionary/WordList.txt")
    res = unitsToWords(["Lionel", "Messi", "là", "một", "Cầu", "thủ", "nổi", "tiếng", "Thế", "giới"], word_set)
    if "Lionel Messi" not in res:
        print("Test 6 not OK. Attention to proper nouns")
        return
    if "cầu thủ" not in res:
        print("Test 6 not OK. Attention to capitalization")
        return
    if "một" not in res:
        print("Test 6 not OK. Attention to 1-syllable words.")
        return
    print("Test 6 OK")
    
        

Tests = [test0, test1, test2, test3, test4, test5, test6]

#MAIN FUNCTION
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Please configure your test by Run -> Configure"
    else:
        for i in range(len(Tests) + 1):
            if sys.argv[1] == "test_" + str(i):
                Tests[i]()
                break
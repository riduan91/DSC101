# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 23:40:21 2018

@author: ndoannguyen
"""

import sys
from HisMCQ_Solution import *

DATAFILE = "QCM.csv"

def test0():
    try:
        myfile = open(DATAFILE)
        myfile.close()
    except:
        print("Please move ", DATAFILE, " to your current folder.")
    print("Test 0 OK")

def test1():
    checker = 1
    text = readQuestionFileAsString(DATAFILE)
    if len(text) not in [16812, 13258] :
        print("Function readQuestionFileAsString() not OK.")
        checker = 0
    
    lines = readQuestionFileAsLines(DATAFILE)
    if len(lines) != 77 or len(lines[34]) not in [103, 85]:
        print("Function readQuestionFileAsLines() not OK.")
        checker = 0
    
    lines = readQuestionFileAsCleanLines(DATAFILE)
    if len(lines) != 77 or len(lines[34]) not in [102, 84]:
        print("Function readQuestionFileAsCleanLines() not OK.")
        checker = 0
    
    if checker == 1:
        print("Test 1 OK")
    else:
        print("Test 1 not OK")

def test2():
    checker = 1
    lines = readQuestionFileAsCleanLines(DATAFILE)
    questions = parseQuestionsAsListOfList(lines)
    if questions[14][3] != "Dòng nào sau đây không nêu đúng hiểm hoạ ngoại xâm của Trung Quốc tương ứng với triều đại trị vì?" or questions[27][6] != "Vạn Xuân":
        print("parseQuestionsAsListOfList() not OK")
        checker = 0
    
    if checker == 1:
        print("Test 2 OK")
    else:
        print("Test 2 not OK")    

def test3():
    checker = 1
    lines = readQuestionFileAsCleanLines(DATAFILE)
    questions = parseQuestionsAsListOfList(lines)
    answer1 = answer(questions, "Dòng nào sau đây không nêu đúng hiểm hoạ ngoại xâm của Trung Quốc tương ứng với triều đại trị vì?")
    if answer1 != "Thời Đường, quân Tây Hạ":
        print("answer() not OK")
        checker = 0
    
    if checker == 1:
        print("Test 3 OK")
    else:
        print("Test 3 not OK")    


def test4():
    checker = 1
    lines = readQuestionFileAsCleanLines(DATAFILE)
    questions = parseQuestionsAsDictionary(lines)
    if len(questions) != 77 or int(questions[hash("Quốc hiệu nước ta dưới thời Lý Nam Đế là gì?")][2]) != 49:
        print("parseQuestionsAsDictionary() not OK")
        checker = 0
    
    if checker == 1:
        print("Test 4 OK")
    else:
        print("Test 4 not OK")    

def test5():
    checker = 1
    lines = readQuestionFileAsCleanLines(DATAFILE)
    questions = parseQuestionsAsDictionary(lines)
    answer1 = answer_2(questions, "Dòng nào sau đây không nêu đúng hiểm hoạ ngoại xâm của Trung Quốc tương ứng với triều đại trị vì?")
    if answer1 != "Thời Đường, quân Tây Hạ" or not isinstance(questions, dict):
        print("answer_2() not OK")
        checker = 0
    
    if checker == 1:
        print("Test 5 OK")
    else:
        print("Test 5 not OK")  
        
def test6():
    checker = 1
    lines = readQuestionFileAsCleanLines(DATAFILE)
    questions = parseQuestionsAsListOfList(lines)
    S = searchQuestionsContainingWord(questions, "Trần")
    
    if len(S) != 5 or int(S[0][2]) != 26:
        print("searchQuestionsContainingWord() not OK")
        checker = 0
    
    if checker == 1:
        print("Test 6 OK")
    else:
        print("Test 6 not OK") 

def test7():
    checker = 1
    lines = readQuestionFileAsCleanLines(DATAFILE)
    questions = parseQuestionsAsListOfList(lines)
    modify(questions[74], 10, "AAA")
    
    if questions[74][10] != "AAA" or questions[74][0] != "VN":
        print("modify() not OK")
        checker = 0
    
    addTag(questions[74], "Mytag")
    addTag(questions[74], "Mytag2")

    if questions[74][10] != "AAA|Mytag|Mytag2":
        print("addTag() not OK")
        checker = 0
    
    if checker == 1:
        print("Test 7 OK")
    else:
        print("Test 7 not OK") 

def test8():
    checker = 1
    lines = readQuestionFileAsCleanLines("QCM.csv")
    questions = parseQuestionsAsListOfList(lines)
    addTag(questions[49], "A")
    addTag(questions[49], "B")
    addTag(questions[49], "C")
    saveDatabaseToFile(questions, "QCM2.csv")
    newquestions = parseQuestionsAsListOfList(readQuestionFileAsCleanLines("QCM2.csv"))
    
    if newquestions[49][10] != "A|B|C":
        print("saveDatabaseToFile() not OK")
        checker = 0
    
    if checker == 1:
        print("Test 8 OK")
    else:
        print("Test 8 not OK") 

def test9():
    checker = 1
    lines = readQuestionFileAsCleanLines("QCM.csv")
    questions = parseQuestionsAsListOfList(lines)
    S = set()
    for i in range(10):
        S.add(generateRandomQuestion(questions)[2])

    if len(S) <= 1:
        print("generateRandomQuestion() not OK")
        checker = 0
    
    if checker == 1:
        print("Test 9 OK")
    else:
        print("Test 9 not OK") 

def test10():
    checker = 1
    lines = readQuestionFileAsCleanLines("QCM.csv")
    questions = parseQuestionsAsListOfList(lines)
    S = set()
    for i in range(2):
        L = generateRandomQuestionList(questions, 20)
        if not isinstance(L, tuple) or not isinstance(L[0], list) or not isinstance(L[1], list) or not isinstance(L[0][0], int) or len(L[1][9]) != 12:
            print("generateRandomQuestionList() not OK")
            checker = 0
        S.update(L[0])

    if len(S) <= 20:
        print("generateRandomQuestionList() not OK")
        checker = 0
    
    if checker == 1:
        print("Test 10 OK")
    else:
        print("Test 10 not OK")

def test11():
    checker = 1
    lines = readQuestionFileAsCleanLines("QCM.csv")
    questions = parseQuestionsAsListOfList(lines)

    if not isCorrectAnswer(questions[11], "C") or isCorrectAnswer(questions[18], "D"):
        print("isCorrectAnswer() not OK")
        checker = 0
    if checker == 1:
        print("Test 11 OK")
    else:
        print("Test 11 not OK")

def test12():
    checker = 1
    A = [1, 2, 3, 4, 5, 6, 7]
    B = [3, 4, 5, 6, 7, 8, 9]

    if not isGoodChoice(A, B, 5) or isGoodChoice(A, B, 4):
        print("isGoodChoice() not OK")
        checker = 0
    if checker == 1:
        print("Test 12 OK")
    else:
        print("Test 12 not OK")
        
Tests = [test0, test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12]

#MAIN FUNCTION
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Please configure your test by Run -> Configure"
    else:
        for i in range(len(Tests) + 1):
            if sys.argv[1] == "test_" + str(i):
                Tests[i]()
                break
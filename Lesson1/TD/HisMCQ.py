# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 23:36:31 2019

@author: ndoannguyen
"""

ZONE = 0
PERIOD = 1
INDEX = 2
CONTENT = 3
OPTION_A = 4
OPTION_B = 5
OPTION_C = 6
OPTION_D = 7
CORRECTION = 8
LEVEL = 9
TAGS = 10
EXPLANATION = 11

DATAFILE = "QCM.csv"
NEWDATAFILE = "NewQCM.csv"
TAG_DELIMITOR = "|"

#---------------------PART 1---------------------------------

def readQuestionFileAsString(filename):
    """
        Exercise 1
    """
    #TODO

def readQuestionFileAsLines(filename):
    """
        Exercise 1
    """
    #TODO

def readQuestionFileAsCleanLines(filename):
    """
        Exercise 1
    """
    #TODO

def parseQuestionsAsListOfList(lines):
    """
        Exercise 2
    """
    #TODO

def answer(questions_list, question_content):
    """
        Exercise 3
    """
    #TODO
    
def parseQuestionsAsDictionary(lines):
    """
        Exercise 4
    """
    #TODO    

def answer_2(questions_dictionary, question_content):
    """
        Exercise 5
    """
    #TODO
    
#---------------------PART 2---------------------------------
    
def searchQuestionsContainingWord(questions_list, keyword):
    """    
        Exercise 6
    """
    #TODO
    
def modify(question, column_index, text):
    """
        Exercise 7
    """
    #TODO

def addTag(question, newtag):
    """
        Exercise 7
    """
    #TODO
    
def saveDatabaseToFile(questions_list, newfilename):
    """
        Exercise 8
    """
    #TODO

def generateRandomQuestion(questions_list):
    """
        Exercise 9
    """
    #TODO

def generateRandomQuestionList(questions_list, nb_questions):
    """
        Exercise 10
    """ 
    #TODO
    
def isCorrectAnswer(question, answer):
    """
        Exercise 11
    """
    #TODO
    
    
#---------------------PART 3---------------------------------

def isGoodChoice(indices_1, indices_2, ceil):
    """
        Exercise 12
    """
    #TODO

def generateHistoryTest():
    """
        Do not modify this function, except for Python 3 syntax adaptation
    """
    nb_questions = raw_input("Please choose the number of question: ")
    lines = readQuestionFileAsLines(DATAFILE)
    questions_list_raw = parseQuestionsAsListOfList(lines)  
    questions_list = generateRandomQuestionList(questions_list_raw, int(nb_questions))[1]
    score = 0
    for question in questions_list:
        print(question[CONTENT])
        print("A. %s" % question[OPTION_A])
        print("B. %s" % question[OPTION_B])
        print("C. %s" % question[OPTION_C])
        print("D. %s" % question[OPTION_D])
        answer = raw_input("Please answer by typing A, B, C or D: ")
        if isCorrectAnswer(question, answer):
            score += 1
    print("Your score is %d / %d" % (score, int(nb_questions)))
    print("Game finished.")
    
def generateRepeatedHistoryTest():
    """
        Exercise 13
    """

#---------------------PART 4---------------------------------

def generateTagUpdateApplication():
    """
        Do not modify this function, except for Python 3 syntax adaptation
    """
    lines = readQuestionFileAsLines(DATAFILE)
    questions_list = parseQuestionsAsListOfList(lines)
    keyword = raw_input("Which keyword do you want to search? Please type in Vietnamese without accent: ")
    keyword 
    concerned_questions = searchQuestionsContainingWord(questions_list, keyword)
    print("%d questions found." % len(concerned_questions))
    if len(concerned_questions):
        newtag = raw_input("What is the hashtag you would like to use?: ")
    for question in concerned_questions:
        print(question[CONTENT])
        want_to_tag = raw_input("Do you want to tag this question with the hashtag %s? Press 'Y' if yes, any other key if no: " % newtag)
        if want_to_tag == 'Y':
            addTag(question, newtag)
        else:
            print("Next question: ")
    print("No more question!")
    saveDatabaseToFile(questions_list, NEWDATAFILE)
    print("Process finished!")
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 23:36:31 2018

@author: ndoannguyen
"""
import random

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

def readQuestionFileAsString(filename):
    """
        Exercise 1
    """
    myfile = open(filename, 'r')
    text = myfile.read()
    myfile.close()
    return text

def readQuestionFileAsLines(filename):
    """
        Exercise 1
    """
    myfile = open(filename, 'r')
    lines = myfile.readlines()
    myfile.close()
    return lines

def readQuestionFileAsCleanLines(filename):
    """
        Exercise 1
    """
    text = readQuestionFileAsString(filename)
    return text.split("\n")

def parseQuestionsAsListOfList(lines):
    """
        Exercise 2
    """
    questions_list = []
    for line in lines:
        question = line.split("\t")
        questions_list.append(question)
    return questions_list

ANSWER_INDEX = {"A": OPTION_A, "B": OPTION_B, "C": OPTION_C, "D": OPTION_D}

def answer(questions_list, question_content):
    """
        Exercise 3
    """
    for question in questions_list:
        if question[CONTENT] == question_content:
            return question[ANSWER_INDEX[question[CORRECTION]]]
    
def parseQuestionsAsDictionary(lines):
    """
        Exercise 4
    """
    questions_dictionary = {}
    for line in lines:
        question = line.split("\t")
        questions_dictionary[hash(question[CONTENT])] = question
    return questions_dictionary

def get(questions_dictionary, question_content):
    """
        Exercise 4
    """
    return questions_dictionary[hash(question_content)]
    

def answer_2(questions_dictionary, question_content):
    """
        Exercise 5
    """
    question = questions_dictionary[hash(question_content)]
    return question[ANSWER_INDEX[question[CORRECTION]]]
    
def searchQuestionsContainingWord(questions_list, keyword):
    """    
        Exercise 6
    """
    return filter(lambda question: question[CONTENT].find(keyword) >= 0, questions_list)
    
def modify(questions_list, question_index, column_index, text):
    """
        Exercise 7
    """
    questions_list[question_index][column_index] = text

def modify_2(questions_dictionary, question_content, column_index, text):
    """
        Exercise 8
    """
    question = questions_dictionary[hash(question_content)]
    if column_index == CONTENT:
        questions_dictionary[hash(question[CONTENT])] = None
        questions_dictionary[hash(text)] = question
        question[column_index] = text
    else:
        question[column_index] = text

def generateRandomQuestion(questions_list):
    """
        Exercise 9
    """
    import random
    random_index = random.randint(0, len(questions_list) - 1)
    return questions_list[random_index]
    
def generateRandomQuestion_2(questions_dictionary):
    """
        Exercise 10
    """
    import random
    random_index = random.randint(0, len(questions_dictionary) - 1)
    return questions_dictionary[questions_dictionary.keys()[random_index]]
    
def getCorrection(question):
    """
        Exercise 11
    """
    return question[CORRECTION]

def isCorrectAnswer(question, answer):
    """
        Exercise 11
    """
    return question[CORRECTION] == answer
    
    
#---------------------PART 2 - A MULTIPLE CHOICE GAME---------------------------------

def generateRandomQuestionList(filename, nb_questions):
    """
        Exercise 12
    """
    lines = readQuestionFileAsLines(filename)
    questions_list = parseQuestionsAsListOfList(lines)  
    my_shuffle = range(len(questions_list))
    random.shuffle(my_shuffle)
    chosen_questions_list = []
    for i in range(nb_questions):
        chosen_questions_list.append(questions_list[my_shuffle[i]])
    return my_shuffle[:nb_questions], chosen_questions_list

def isGoodChoice(indices_1, indices_2, ceil):
    """
        Exercise 13
    """
    return len(set(indices_1) & set(indices_2)) <= ceil


def generateHistoryTest():
    """
        Do not modify this function
    """
    nb_questions = raw_input("Please choose the number of question: ")
    questions_list = generateRandomQuestionList(DATAFILE, int(nb_questions))[1]
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
        Exercise 14
    """
    all_indices = set([])
    nb_questions = raw_input("Please choose the number of question: ")
    want_to_continue = True
    max_score = 0
    database_expired = False
    while want_to_continue and not database_expired: 
        questions_list_with_indices = generateRandomQuestionList(DATAFILE, int(nb_questions))
        indices, questions_list = questions_list_with_indices[0], questions_list_with_indices[1]
        
        nb_tries = 0
        while nb_tries < 50 and not isGoodChoice(indices, all_indices, int(nb_questions)/3):
            questions_list_with_indices = generateRandomQuestionList(DATAFILE, int(nb_questions))
            indices, questions_list = questions_list_with_indices[0], questions_list_with_indices[1]
            nb_tries += 1
        
        if nb_tries == 50:
            print("Sorry, you seem to have answered almost all the questions in the database.")
            print("Your highest score in an attempt is %d / %d" % (max_score, int(nb_questions)))
            print("Game finished.")
            database_expired = True
            return
        
        else:
            score = 0
            all_indices = all_indices | set(indices)
            for question in questions_list:
                print(question[CONTENT])
                print("A. %s" % question[OPTION_A])
                print("B. %s" % question[OPTION_B])
                print("C. %s" % question[OPTION_C])
                print("D. %s" % question[OPTION_D])
                answer = raw_input("Please answer by typing A, B, C or D: ")
                if isCorrectAnswer(question, answer):
                    score += 1
            if score > max_score:
                max_score = score
            print("Your score this time is %d / %d" % (score, int(nb_questions)))
            print("Number of questions has been used in the database: %d." % len(all_indices))
        
        continue_key = raw_input("Do you want to replay? Press 'Y' if yes, any other key if no: ")
        if continue_key == 'Y':
            want_to_continue = True
        else:
            print("Your highest score in an attempt is %d / %d" % (max_score, int(nb_questions)))
            print("Game finished.")
            want_to_continue = False
            return

#---------------------PART 3 - APPLICATION TO ADD TAGS---------------------------------

def addTag(question, newtag):
    """
        Exercise 15
    """
    current_tags = question[TAGS].split(TAG_DELIMITOR)
    if newtag not in current_tags:
        current_tags.append(newtag)
        if current_tags[0] == '-':
            current_tags = current_tags[1:]
    question[TAGS] = TAG_DELIMITOR.join(current_tags)

def saveDatabaseToFile(questions_list, newfilename):
    """
        Exercise 16
    """
    myfile = open(newfilename, 'w')
    for question in questions_list:
        myfile.write("\t".join(question))
        myfile.write("\n")
    myfile.close()

def generateTagUpdateApplication():
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
            
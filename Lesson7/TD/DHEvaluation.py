# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:34:28 2018

@author: ndoannguyen
"""

import numpy as np
import cv2
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

import scipy.io

RAW_DATA = "RawForm/"
SCORE_DATA = "ScoreData/"
LABEL_DATA = "LabelData/Labels.csv"
MNIST_DATA = "MNIST/mldata/mnist-original"
MNIST_DATA_TRANSFORMED = "MNIST/mldata/mnist-transformed"
MIN_MARK = 0
MAX_MARK = 10

class EvaluationForm:
    BLACK = 0
    BLACK_WHITE_THRESHOLD = 168
    WHITE = 255
    
    MARGIN_UP = 600
    MARGIN_DOWN = 100
    MARGIN_LEFT = 1140
    MARGIN_RIGHT = 300
    ROW_GROUP_DISTANCE = 10
    COLUMN_GROUP_DISTANCE = 10


    EDGE_LOW_VALUE = 100
    EDGE_HIGH_VALUE = 800
    MIN_VOTE_HORIZONTAL = 400
    MIN_VOTE_VERTICAL = 300
    
    CELL_MARGIN = 5
    CRITERIA = ["A", "B", "C", "D"]
    
    SCORE_WIDTH = 0
    SMALL_SIZE = 28
    
    def __init__(self, source_image):
        # Exercise 2
        # TODO
    
    def getImage(self):
        return self.img
    
    def getHeight(self):
        return self.img.shape[0]
    
    def getWidth(self):
        return self.img.shape[1]
    
    def getEdges(self):
        # Exercise 3
        # TODO
    
    def saveImage(self, destination_file):
        cv2.imwrite(destination_file, self.img)
    
    def getHorizontalLines(self):
        # Exercise 4
        # TODO
        
    def getVerticalLines(self):
        # Exercise 4
        # TODO
        
    def getHorizontalLineGroups(self):
        # Exercise 5
        # TODO

    def getCleanHorizontalLineGroups(self):
        # Exercise 6
        # TODO
         
    def getVerticalLineGroups(self):
        # Exercise 5
        # TODO

    def getCleanVerticalLineGroups(self):
        # Exercise 6
        # TODO
    
    def getCellLeftEdges(self):
        # Exercise 7
        # TODO
        
    def getCellRightEdges(self):
        # Exercise 7
        # TODO
        
    def getCellTopEdges(self):
        # Exercise 7
        # TODO
        
    def getCellBottomEdges(self):
        # Exercise 7
        # TODO

    def saveCells(self):
        # Exercise 8
        # TODO

MIN_BLACK = 64

def MyPerceptron():
    return None
    # Exercise 15
    # TODO
        
def MyLDA():
    return None
    # Exercise 15
    # TODO
    
def MyQDA():
    return None
    # Exercise 15
    # TODO
    
def MyLogisticRegression():
    return None
    # Exercise 15
    # TODO
    
def MyGaussianNaiveBayes():
    return None
    # Exercise 15
    # TODO
    
def MyKNN5():
    return None
    # Exercise 15
    # TODO
    
def MyKNN1():
    return None
    # Exercise 15
    # TODO
    
def MySVM():
    return None
    # Exercise 15
    # TODO
    
def MyLinearRegression():
    return None
    # Exercise 15
    # TODO

MODELS = [MyPerceptron, MyLDA, MyQDA, MyLogisticRegression, MyGaussianNaiveBayes, MyKNN5, MyKNN1, MySVM, MyLinearRegression]
MODEL_NAMES = ["Perceptron", "LDA", "QDA", "LogisticRegression", "GaussianNaiveBayes", "KNN5", "KNN1", "SVM", "LinearRegression"]

class Mark:    
    def __init__(self, X_folder=None, y_file = None):
        # Exercise 9
        # TODO        
        
    def getX(self):
        # Exercise 10
        # TODO    
    
    def getNames(self):
        # Exercise 10
        # TODO    
    
    def gety(self):
        # Exercise 10
        # TODO
        
    def draw(self, index):
        # Exercise 11
        # TODO
    
    def drawAfterFiltering(self, index):
        # Exercise 14
        # TODO
        
    def getFilteredX(self):
        # Exercise 12
        # TODO  
    
    def getFilteredNames(self):
        # Exercise 12
        # TODO  
    
    def getFilteredy(self):
        # Exercise 12
        # TODO     

    def filterData(self, criterion = "FilterByQuality", args = None):
        # Exercise 12, 13
        # TODO        

    def getTrainTestScore(self, model, i, j):
        # Exercise 16
        # TODO 
    
    def massiveTrainTest(self, models, max_mark = MAX_MARK):
        # Exercise 17
        # TODO

    def getWrongCase(self, model, i, j):
        # Exercise 18
        # TODO  
        
    def loadMatData(self, filename):
        # Exercise 19
        # TODO  
        
    def loadTransformedMatData(self, filename):
        # Exercise 21
        # TODO
        

def transformMatData(inputfile, outputfile):
    mat = scipy.io.loadmat(inputfile)
    X = mat["data"].T
    copyX = X[:,:]
    for i, x in enumerate(X):
        thres, binary_x = cv2.threshold(x, EvaluationForm.BLACK_WHITE_THRESHOLD, EvaluationForm.WHITE, cv2.THRESH_BINARY)
        binary_x = binary_x/255
        binary_x = binary_x.reshape((28, 28))
        min_row, max_row, min_col, max_col = EvaluationForm.getRealBound(1 - binary_x)
        binary_x = binary_x[min_row: max_row+1, min_col: max_col+1]
        binary_x = cv2.resize(binary_x, (EvaluationForm.SMALL_SIZE, EvaluationForm.SMALL_SIZE), interpolation = cv2.INTER_LINEAR)
        copyX[i] = binary_x.flatten() 
    scipy.io.savemat(outputfile, {"data": copyX.T, "label": mat["label"]})

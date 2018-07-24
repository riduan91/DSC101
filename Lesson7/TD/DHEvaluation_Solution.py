# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:34:28 2018

@author: ndoannguyen
"""

import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
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
        intermediate_img = cv2.imread(source_image, cv2.IMREAD_GRAYSCALE)
        self.ret, self.img = cv2.threshold(intermediate_img, self.BLACK_WHITE_THRESHOLD, self.WHITE, cv2.THRESH_BINARY)
        self.name = source_image.replace(RAW_DATA, "")
    
    def getImage(self):
        return self.img
    
    def getHeight(self):
        return self.img.shape[0]
    
    def getWidth(self):
        return self.img.shape[1]
    
    def getEdges(self):
        self.edges = cv2.Canny(self.img, self.EDGE_LOW_VALUE, self.EDGE_HIGH_VALUE)
        return self.edges
    
    def saveImage(self, destination_file):
        cv2.imwrite(destination_file, self.img)
    
    def getHorizontalLines(self):
        if 'edges' not in self.__dict__:
            self.getEdges()
        lines = cv2.HoughLinesP(self.edges, 1, np.pi/2, self.MIN_VOTE_HORIZONTAL)
        self.horizontalLines = lines[:]
        return sorted(list(map(lambda x: x[1], list(filter(lambda x: x[1] == x[3] and x[1] > self.MARGIN_UP and x[1] < self.getHeight() - self.MARGIN_DOWN , lines[:, 0, :])))))
    
    def getVerticalLines(self):
        if 'edges' not in self.__dict__:
            self.getEdges()
        lines = cv2.HoughLinesP(self.edges, 1, np.pi/2, self.MIN_VOTE_VERTICAL)
        self.verticalLines = lines[:]
        return sorted(list(map(lambda x: x[0], list(filter(lambda x: x[0] == x[2] and x[0] > self.MARGIN_LEFT and x[0] < self.getWidth() - self.MARGIN_RIGHT, lines[:, 0, :])))))
    
    def getHorizontalLineGroups(self):
        edges = self.getHorizontalLines()
        edge_groups = []
        if len(edges) > 0:
            edge_groups.append([edges[0]])
        current_group = 0
        for i in range(1, len(edges)):
            if edges[i] - edges[i-1] < self.ROW_GROUP_DISTANCE:
                edge_groups[current_group].append(edges[i])
            else:
                edge_groups.append([edges[i]])
                current_group += 1
        self.horizontalLineGroups = edge_groups
        return edge_groups

    def getCleanHorizontalLineGroups(self):
        edge_groups = list(filter(lambda x: len(x) >= 3, self.getHorizontalLineGroups()))
        self.horizontalLineGroups = edge_groups
        return edge_groups
        
        
    def getVerticalLineGroups(self):
        edges = self.getVerticalLines()
        edge_groups = []
        if len(edges) > 0:
            edge_groups.append([edges[0]])
        current_group = 0
        for i in range(1, len(edges)):
            if edges[i] - edges[i-1] < self.COLUMN_GROUP_DISTANCE:
                edge_groups[current_group].append(edges[i])
            else:
                edge_groups.append([edges[i]])
                current_group += 1
        self.verticalLineGroups = edge_groups
        return edge_groups

    def getCleanVerticalLineGroups(self):
        edge_groups = list(filter(lambda x: len(x) >= 3, self.getVerticalLineGroups()))
        self.verticalLineGroups = edge_groups
        return edge_groups
    
    def getCellLeftEdges(self):
        if 'verticalLineGroups' not in self.__dict__:
            self.getCleanVerticalLineGroups()
        return [e[-1] for e in self.verticalLineGroups[:-1]]
        
    def getCellRightEdges(self):
        if 'verticalLineGroups' not in self.__dict__:
            self.getCleanVerticalLineGroups()
        return [e[0] for e in self.verticalLineGroups[1:]]
        
    def getCellTopEdges(self):
        if 'horizontalLineGroups' not in self.__dict__:
            self.getCleanHorizontalLineGroups()
        return [e[-1] for e in self.horizontalLineGroups[:-1]]
        
    def getCellBottomEdges(self):
        if 'horizontalLineGroups' not in self.__dict__:
            self.getCleanHorizontalLineGroups()
        return [e[0] for e in self.horizontalLineGroups[1:]]
    

    def saveCells(self):
        left = self.getCellLeftEdges()
        right = self.getCellRightEdges()
        top = self.getCellTopEdges()
        bottom = self.getCellBottomEdges()
        s = self.CELL_MARGIN
        for i in range(len(top)):
            for j in range(len(left)):
                cell = self.img[int(top[i]) + s: int(bottom[i]) - s, int(left[j]) + s: int(right[j]) - s]
                min_row, max_row, min_col, max_col = EvaluationForm.getRealBound(cell)
                if max_row - min_row < self.SCORE_WIDTH:
                    continue
                cell = cell[min_row: max_row + 1, min_col: max_col + 1]
                cell = cv2.resize(cell, (self.SMALL_SIZE, self.SMALL_SIZE), interpolation = cv2.INTER_LINEAR)
                cv2.imwrite('%s%s_%d-%s.jpg' % (SCORE_DATA, self.name.replace(".jpg", ""), i, self.CRITERIA[j]), cell)
        return
    
    @staticmethod
    def getRealBound(X):
        min_row = X.shape[0] - 1
        max_row = 0
        min_col = X.shape[1] - 1
        max_col = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i,j] == 0:
                    if min_row > i:
                        min_row = i
                    if min_col > j:
                        min_col = j
                    if max_row < i:
                        max_row = i
                    if max_col < j:
                        max_col = j
        return min_row, max_row, min_col, max_col

MIN_BLACK = 64

def MyPerceptron():
    return Perceptron()

def MyLDA():
    return LinearDiscriminantAnalysis()

def MyQDA():
    return QuadraticDiscriminantAnalysis()

def MyLogisticRegression():
    return LogisticRegression(C = 1.0)

def MyGaussianNaiveBayes():
    return GaussianNB()

def MyKNN5():
    return KNeighborsClassifier(n_neighbors=5, weights = 'uniform')

def MyKNN1():
    return KNeighborsClassifier(n_neighbors=1, weights = 'uniform')

def MySVM():
    return SVC(C = 1.0, kernel = 'linear')

def MyLinearRegression():
    return LinearRegression()

class Mark:    
    def __init__(self, X_folder=None, y_file = None):
        if X_folder == None:
            self.X = None
            self.y = None
            self.names = None
            return
        N = len(list(filter(lambda x: x.endswith(".jpg"), os.listdir(X_folder))))
        self.names = [""] * N 
        X = [""] * N 
        for i, imgfile in enumerate(os.listdir(X_folder)):
            if not imgfile.endswith(".jpg"):
                continue
            intermediate_img = cv2.imread(X_folder + imgfile, cv2.IMREAD_GRAYSCALE)
            ret, img = cv2.threshold(intermediate_img, EvaluationForm.BLACK_WHITE_THRESHOLD, EvaluationForm.WHITE, cv2.THRESH_BINARY)
            self.names[i] = imgfile
            img = ((255 - img.flatten())/255).astype(int)
            X[i] = img
        self.X = np.array(X) 
        if y_file != None:
            self.ready(y_file)
        else:
            self.y = None
    
    def ready(self, filename):
        lines = pd.read_csv(filename, header=None)
        y_dict = {}
        for i, j in zip(lines.iloc[:, 0].values, lines.iloc[:, 1].values):
            y_dict[i] = j
        y = np.zeros((len(self.names)))
        for i, name in enumerate(self.names):
            if name in y_dict:
                y[i] = y_dict[name]
            else:
                y[i] = -1
        self.y = y
        
    def getX(self):
        return self.X
    
    def getNames(self):
        return self.names
    
    def gety(self):
        if 'y' not in self.__dict__:
            return []
        return self.y

    def draw(self, index):
        reshape = self.getX()[index].reshape((28, 28))
        S = "-" * 30 + "\n"
        for i in range(reshape.shape[0]):
            S += "|"
            for j in range(reshape.shape[1]):
                if reshape[i, j] == 1:
                    S += "X"
                else:
                    S += " "
            S += "|\n"
        S += ("-" * 30)
        print(S)
    
    def drawAfterFiltering(self, index):
        reshape = self.getFilteredX()[index].reshape((28, 28))
        S = "-" * 30 + "\n"
        for i in range(reshape.shape[0]):
            S += "|"
            for j in range(reshape.shape[1]):
                if reshape[i, j] == 1:
                    S += "X"
                else:
                    S += " "
            S += "|\n"
        S += ("-" * 30)
        print(S)
    
    def criterion1(x, y, name, args = None):
        return sum(x) >= MIN_BLACK        
    
    def criterion2(x, y, name, args = None):
        check = False
        for school in args:
            if name.startswith(school):
                check = True
        return check and sum(x) >= MIN_BLACK
    
    def criterion3(x, y, name, args = None):
        check = False
        for score in args:
            if y == score:
                check = True
        return check and sum(x) >= MIN_BLACK
        
    CRITERIA = {"FilterByQuality": criterion1,
                "FilterByQualityAndSchool": criterion2,
                "FilterByQualityAndScore": criterion3
                }
    
    def filterData(self, criterion = "FilterByQuality", args = None):
        self.X_filtered = np.array([self.X[i] for i in range(len(self.names)) if self.CRITERIA[criterion](self.X[i], self.y[i], self.names[i], args)])
        self.names_filtered = np.array([self.names[i] for i in range(len(self.names)) if self.CRITERIA[criterion](self.X[i], self.y[i], self.names[i], args)])
        if 'y' in self.__dict__ and len(self.y) == len(self.names):
            self.y_filtered = np.array([self.y[i] for i in range(len(self.y)) if self.CRITERIA[criterion](self.X[i], self.y[i], self.names[i], args)])
        
    def getFilteredX(self):
        if 'X_filtered' not in self.__dict__:
            return self.X
        return self.X_filtered
    
    def getFilteredNames(self):
        if 'names_filtered' not in self.__dict__:
            return self.names
        return self.names_filtered
    
    def getFilteredy(self):
        if 'y_filtered' not in self.__dict__:
            return self.y
        return self.y_filtered    

    def getTrainTestScore(self, model, i, j):
        self.filterData(criterion = "FilterByQualityAndScore", args = [i, j])
        X = self.getFilteredX()
        y = self.getFilteredy()
        rate = np.mean(cross_val_score(model(), X, y, cv=3))
        return rate
    
    def getWrongCase(self, model, i, j):
        model = model()
        S = set([])
        self.filterData("FilterByQualityAndScore", [i, j])
        X = self.getFilteredX()
        y = self.getFilteredy()
        names = self.getFilteredNames()
        N = len(X)
        X_train, X_test, y_train, y_test, names_train, names_test = X[:N//2], X[N//2:], y[:N//2], y[N//2:], names[:N//2], names[N//2:]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        for name, y1, y2 in zip(names_test, y_test, y_pred):
            if y1 != y2:
                S.add(name)
        
        model.fit(X_test, y_test)
        y_pred = model.predict(X_train)
        for name, y1, y2 in zip(names_train, y_train, y_pred):
            if y1 != y2:
                S.add(name)
        return S    
    
    def massiveTrainTest(self, models, max_mark = MAX_MARK):
        results = np.zeros((max_mark+1, max_mark+1, len(models)))
        for i in range(0, max_mark+1):
            for j in range(i+1, max_mark+1):
                print("Classifying %d and %d" % (i, j))
                for e, model in enumerate(models):
                    tts = self.getTrainTestScore(model, i, j)
                    results[i, j, e] = tts
                    results[j, i, e] = tts
        return results
        
    def loadMatData(self, filename):
        mat = scipy.io.loadmat(filename)
        self.X = mat["data"].T
        thres, binary_X = cv2.threshold(self.X, EvaluationForm.BLACK_WHITE_THRESHOLD, EvaluationForm.WHITE, cv2.THRESH_BINARY) 
        self.X = binary_X/255
        self.y = mat["label"][0]
        self.names = np.array(["mnist_" + str(i) for i in range(len(self.X))])
        
    def loadTransformedMatData(self, filename):
        mat = scipy.io.loadmat(filename)
        self.X = mat["data"].T
        self.y = mat["label"][0]
        self.names = np.array(["mnist_" + str(i) for i in range(len(self.X))])
        

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
        
MODELS = [MyPerceptron, MyLDA, MyQDA, MyLogisticRegression, MyGaussianNaiveBayes, MyKNN5, MyKNN1, MySVM, MyLinearRegression]
MODEL_NAMES = ["Perceptron", "LDA", "QDA", "LogisticRegression", "GaussianNaiveBayes", "KNN5", "KNN1", "SVM", "LinearRegression"]

mnist_mark = Mark()
mnist_mark.loadMatData(MNIST_DATA)

donghanh_mark = Mark(SCORE_DATA, LABEL_DATA)

group_identifier = []
scores = []

mnist_mark = Mark()
mnist_mark.loadTransformedMatData(MNIST_DATA_TRANSFORMED)

donghanh_mark = Mark(SCORE_DATA, LABEL_DATA)

group_identifier = []
scores = []

for i in range(10):
    for j in range(i+1, 10):
        print(i, " and ", j)
        group_identifier.append("%d vs %d" % (i, j))
        mnist_mark.filterData(criterion = "FilterByQualityAndScore", args = [i, j])
        donghanh_mark.filterData(criterion = "FilterByQualityAndScore", args = [i, j])
        model = MyLogisticRegression().fit(mnist_mark.getFilteredX(), mnist_mark.getFilteredy())
        score =  model.score(donghanh_mark.getFilteredX(), donghanh_mark.getFilteredy())
        scores.append(score)

data = pd.DataFrame(scores, index = group_identifier, columns = ["LogisticRegression"])
print(data)
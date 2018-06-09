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
from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

import scipy.io

RAW_DATA = "ComplexRawForm/"
SCORE_DATA = "ComplexScoreData/"
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
        # Fix in exercise 31
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

def MyLogisticRegression_Multiclass_OVR():
    # Exercise 24
    return LogisticRegressionCV(multi_class = 'ovr')

def MyLogisticRegression_Multiclass_MTN():
    # Exercise 24
    return LogisticRegressionCV(multi_class = 'multinomial')

def MyLDA_Multiclass():
    # Exercise 24
    return LinearDiscriminantAnalysis(store_covariance = True)

def MyKNN5_Multiclass():
    # Exercise 24
    return KNeighborsClassifier(n_neighbors=5, weights = 'uniform')

def MyKNN1_Multiclass():
    # Exercise 24
    return KNeighborsClassifier(n_neighbors=1, weights = 'uniform')

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

    def getTrainTestScore(self, model, i = None, j = None, scoring='accuracy'):
        # Exercise 25
        if i is not None and j is not None:
            self.filterData(criterion = "FilterByQualityAndScore", args = [i, j])
        else:
            self.filterData(criterion = "FilterByQuality")
        X = self.getFilteredX()
        y = self.getFilteredy()
        rate = np.mean(cross_val_score(model, X, y, cv=3, scoring=scoring))
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
        
    def saveModelParameters(self, model, model_name, parameters_file):
        # Exercise 27
        f = open(parameters_file, 'w')
        self.filterData(criterion = "FilterByQuality")
        f.write(model_name)
        f.write("\n")
        model = model()
        if model_name in ["LogisticRegression_OVR", "LogisticRegression_Multinomial"]:
            model.fit(self.getFilteredX(), self.getFilteredy())
            f.write("coef_")
            f.write("\n")
            f.write("\n".join([", ".join([str(model.coef_[i, j]) for j in range(len(model.coef_[i]))]) for i in range(len(model.coef_))]))
            f.write("\n")
            f.write("intercept_")
            f.write("\n")
            f.write("\n ".join([str(model.intercept_[i]) for i in range(len(model.intercept_))]))
            f.write("\n")
            f.write("C_")
            f.write("\n")
            f.write("\n ".join([str(model.C_[i]) for i in range(len(model.C_))]))
            f.write("\n")
            f.write("classes_")
            f.write("\n")
            f.write("\n ".join([str(model.classes_[i]) for i in range(len(model.classes_))]))
            f.write("\n")
        elif model_name in ["LDA_Multinomial"]:
            model.fit(self.getFilteredX(), self.getFilteredy())
            f.write("coef_")
            f.write("\n")
            f.write("\n".join([", ".join([str(model.coef_[i, j]) for j in range(len(model.coef_[i]))]) for i in range(len(model.coef_))]))
            f.write("\n")
            f.write("intercept_")
            f.write("\n")
            f.write("\n".join([str(model.intercept_[i]) for i in range(len(model.intercept_))]))
            f.write("\n")
            f.write("priors_")
            f.write("\n")
            f.write("\n".join([str(model.priors_[i]) for i in range(len(model.priors_))]))
            f.write("\n")
            f.write("means_")
            f.write("\n")
            f.write("\n".join([", ".join([str(model.means_[i, j]) for j in range(len(model.means_[i]))]) for i in range(len(model.means_))]))
            f.write("\n")
            f.write("covariance_")
            f.write("\n")
            f.write("\n".join([", ".join([str(model.covariance_[i, j]) for j in range(len(model.covariance_[i]))]) for i in range(len(model.covariance_))]))
            f.write("\n")
            f.write("classes_")
            f.write("\n")
            f.write("\n".join([str(model.classes_[i]) for i in range(len(model.classes_))]))
            f.write("\n")
            
        f.close()
    
    def loadModelParameters(self, parameters_file):
        # Exercise 28
        f = open(parameters_file, 'r')
        name = f.readline().replace("\n", "")
        nb_classes = 11
        nb_features = 28*28
        if name in ["LogisticRegression_OVR", "LogisticRegression_Multinomial"] :
            if name == "LogisticRegression_OVR":
                model = MyLogisticRegression_Multiclass_OVR()
            else:
                model = MyLogisticRegression_Multiclass_MTN()
            coef_ = np.zeros((nb_classes, nb_features))
            intercept_ = np.zeros((nb_classes))
            C_ = np.zeros((nb_classes))
            classes_ = np.zeros((nb_classes))
            f.readline()
            for i in range(nb_classes):
                line = np.array(f.readline().replace("\n", "").split(", ")).astype(float)
                coef_[i] = line
            f.readline()
            for i in range(nb_classes):
                line = float(f.readline().replace("\n", ""))
                intercept_[i] = line
            f.readline()
            for i in range(nb_classes):
                line = float(f.readline().replace("\n", ""))
                C_[i] = line
            f.readline()
            for i in range(nb_classes):
                line = float(f.readline().replace("\n", ""))
                classes_[i] = line
            model.coef_ = coef_
            model.intercept_ = intercept_
            model.C_ = C_
            model.classes_ = classes_
        elif name in ["LDA_Multinomial"]:
            model = MyLDA_Multiclass()
            coef_ = np.zeros((nb_classes, nb_features))
            intercept_ = np.zeros((nb_classes))
            priors_ = np.zeros((nb_classes))
            means_ = np.zeros((nb_classes, nb_features))
            covariance_ = np.zeros((nb_features, nb_features))
            classes_ = np.zeros((nb_classes))
            f.readline()
            for i in range(nb_classes):
                line = np.array(f.readline().replace("\n", "").split(", ")).astype(float)
                coef_[i] = line
            f.readline()
            for i in range(nb_classes):
                line = float(f.readline().replace("\n", ""))
                intercept_[i] = line
            f.readline()
            for i in range(nb_classes):
                line = float(f.readline().replace("\n", ""))
                priors_[i] = line
            f.readline()
            for i in range(nb_classes):
                line = np.array(f.readline().replace("\n", "").split(", ")).astype(float)
                means_[i] = line
            f.readline()
            for i in range(nb_features):
                line = np.array(f.readline().replace("\n", "").split(", ")).astype(float)
                covariance_[i] = line
            f.readline()
            for i in range(nb_classes):
                line = float(f.readline().replace("\n", ""))
                classes_[i] = line
            model.coef_ = coef_
            model.intercept_ = intercept_
            model.priors_ = priors_
            model.means_ = means_
            model.covariance_ = covariance_
            model.classes_ = classes_    
        elif name == "KNN5":
            model = MyKNN5_Multiclass()
        elif name == "KNN1":
            model = MyKNN1_Multiclass()
        f.close()
        return model
    
    def predict(self, model):
        # Exercise 29
        return model.predict(self.getX())
    
    def predictAsDict(self, model):
        # Exercise 30
        names = self.getNames()
        predictions = self.predict(model)
        res = {}
        for name, pred in zip(names, predictions):
            res[name] = pred
        return res

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
    
class ComplexImage():
    BLACK_WHITE_THRESHOLD = 168
    WHITE = 255
    DECIMAL_POINT_MIN_SIZE = 20
    DECIMAL_POINT_MAX_SIZE = 300
    SMALL_SIZE = 28
    
    def __init__(self, source_image):
        # Exercise 33
        intermediate_img = cv2.imread(source_image, cv2.IMREAD_GRAYSCALE)
        self.ret, self.img = cv2.threshold(intermediate_img, self.BLACK_WHITE_THRESHOLD, self.WHITE, cv2.THRESH_BINARY)
        self.name = source_image.replace(SCORE_DATA, "")
    
    def getImage(self):
        # Exercise 33
        return self.img
    
    def getHeight(self):
        # Exercise 33
        return self.img.shape[0]
    
    def getWidth(self):
        # Exercise 33
        return self.img.shape[1]

    def draw(self):
        # Exercise 33
        S = "-" * self.getWidth() + "\n"
        for i in range(self.getHeight()):
            S += "|"
            for j in range(self.getWidth()):
                if self.img[i, j] == 0:
                    S += "X"
                else:
                    S += " "
            S += "|\n"
        S += ("-" * self.getWidth())
        print(S)
    
    def getProjection(self):
        # Exercise 34
        res = np.ones((self.getWidth()))
        for i in range(self.getWidth()):
            if sum(self.WHITE - self.img[:,i]) == 0:
                res[i] = 0
        segments = []
        for i in range(self.getWidth()):
            if res[i] == 1 and (i == 0 or res[i-1] == 0):
                segments.append([i])
            if res[i] == 0 and i > 0 and res[i-1] == 1:
                segments[-1].append(i)
        if len(segments[-1]) == 1:
            segments[-1].append(self.getWidth())
        return segments
    
    def getSmallerImages(self):
        # Exercise 35
        smaller_images = []
        segments = self.getProjection()
        for segment in segments:
            img = self.img[:, segment[0] : segment[1]]
            min_row, max_row, min_col, max_col = EvaluationForm.getRealBound(img)
            if max_row - min_row < 0:
                continue
            img = img[min_row: max_row + 1, min_col: max_col + 1]
            smaller_images.append(img)
        return smaller_images
    
    @staticmethod
    def isDecimalPoint(img):
        # Exercise 36
        return np.sum(ComplexImage.WHITE - img) > ComplexImage.DECIMAL_POINT_MIN_SIZE * ComplexImage.WHITE and np.sum(ComplexImage.WHITE - img) < ComplexImage.DECIMAL_POINT_MAX_SIZE * ComplexImage.WHITE
    
    @staticmethod
    def isDigit(img):
        # Exercise 36
        return np.sum(ComplexImage.WHITE - img) >= ComplexImage.DECIMAL_POINT_MAX_SIZE * ComplexImage.WHITE
    
    
    def saveSmallerImages(self):
        # Exercise 37
        imgs = self.getSmallerImages()
        imgs = list(filter(lambda x : ComplexImage.isDecimalPoint(x) or ComplexImage.isDigit(x), imgs))
        point_pos = len(imgs)
        for i, img in enumerate(imgs):
            if ComplexImage.isDecimalPoint(img):
                point_pos = i
                break    
        
        for i, img in enumerate(imgs):
            if not ComplexImage.isDecimalPoint(img):
                new_img = cv2.resize(img, (self.SMALL_SIZE, self.SMALL_SIZE), interpolation = cv2.INTER_LINEAR)
                digit_pos = 0
                if i < point_pos:
                    digit_pos = point_pos - i - 1
                else:
                    digit_pos = point_pos - i
                cv2.imwrite('%s%s_Position_%d.jpg' % (SCORE_DATA_SPLITTED, self.name.replace(".jpg", ""), digit_pos), new_img)
        return
            
        
MODELS = [MyPerceptron, MyLDA, MyQDA, MyLogisticRegression, MyGaussianNaiveBayes, MyKNN5, MyKNN1, MySVM, MyLinearRegression]
MODEL_NAMES = ["Perceptron", "LDA", "QDA", "LogisticRegression", "GaussianNaiveBayes", "KNN5", "KNN1", "SVM", "LinearRegression"]
MULTICLASS_MODELS = [MyLogisticRegression_Multiclass_OVR, MyLogisticRegression_Multiclass_MTN, MyLDA_Multiclass, MyKNN5_Multiclass, MyKNN1_Multiclass]
MULTICLASS_MODEL_NAMES = ["LogisticRegression_OVR", "LogisticRegression_Multinomial", "LDA_Multinomial", "KNN5", "KNN1"]
SCORINGS = ["accuracy", "f1_micro", "f1_macro", "precision_micro", "precision_macro", "recall_micro", "recall_macro"]

SELECTED_MULTICLASS_MODELS = [MyLogisticRegression_Multiclass_MTN, MyKNN1_Multiclass]
SELECTED_MULTICLASS_MODEL_NAMES = ["LogisticRegression_Multinomial", "KNN1"]

LOGISTIC_REGRESSION_OVR_PARAMS = "SavedModels/LogisticRegression_OVR.txt"
LOGISTIC_REGRESSION_MTN_PARAMS = "SavedModels/LogisticRegression_MTN.txt"
LDA_MULTINOMIAL_PARAMS = "SavedModels/LDA_Multinomial.txt" 
KNN5_PARAMS = "SavedModels/KNN5.txt" 
KNN1_PARAMS = "SavedModels/KNN1.txt" 

SCORE_DATA_SPLITTED = "ComplexScoreData_Splitted/"


def readEvaluationFormAsDict(image_folder, model_file):
    # Exercise 30
    if not os.path.isdir(SCORE_DATA):
        os.mkdir(SCORE_DATA)
    for imgfile in os.listdir(image_folder):
        form = EvaluationForm(image_folder + imgfile)
        form.saveCells()
    mark = Mark(X_folder = SCORE_DATA)
    model = mark.loadModelParameters(model_file)
    return mark.predictAsDict(model)

def readComplexEvaluationFormAsDict(image_folder, model_file):
    # Exercise 38
    if not os.path.isdir(SCORE_DATA):
        os.mkdir(SCORE_DATA)
    if not os.path.isdir(SCORE_DATA_SPLITTED):
        os.mkdir(SCORE_DATA_SPLITTED)    
    for imgfile in os.listdir(image_folder):
        form = EvaluationForm(image_folder + imgfile)
        form.SMALL_SIZE = 100
        form.saveCells()
    for imgfile in os.listdir(SCORE_DATA):
        complexImgFile = ComplexImage(SCORE_DATA + imgfile)
        complexImgFile.saveSmallerImages()
    mark = Mark(X_folder = SCORE_DATA_SPLITTED)
    model = mark.loadModelParameters(model_file)
    raw_dict = mark.predictAsDict(model)
    new_dict = {}
    for k in raw_dict:
        real_key = k.replace(".jpg", "").split("_Position_")[0] + ".jpg"
        position = int(k.replace(".jpg", "").split("_Position_")[1])
        if real_key not in new_dict:
            new_dict[real_key] = raw_dict[k] * 10.0 ** position
        else:
            new_dict[real_key] += raw_dict[k] * 10.0 ** position
    return new_dict
    
def test_26():
    donghanh_mark = Mark(SCORE_DATA, LABEL_DATA)
    results = np.zeros((len(MULTICLASS_MODELS), len(SCORINGS)))
    for i, model, modelname in zip(range(len(MULTICLASS_MODELS)), MULTICLASS_MODELS, MULTICLASS_MODEL_NAMES):
        for j, scoring in enumerate(SCORINGS):
            print("Evaluation of model %s, using %s" % (modelname, scoring))
            results[i, j] = donghanh_mark.getTrainTestScore(model(), scoring=scoring)
    return results

"""
[[0.91278021 0.90550712 0.91278021 0.91909033 0.91278021 0.89827087]
 [0.93223133 0.92457184 0.93223133 0.93413081 0.93223133 0.92138473]
 [0.8045219  0.78971981 0.8045219  0.80204618 0.8045219  0.78664076]
 [0.94788586 0.94117794 0.94788586 0.9506969  0.94788586 0.94003042]
 [0.95832615 0.9557867  0.95832615 0.96319252 0.95832615 0.95205785]]
"""
#print(test_26())

#TEST 28
"""
donghanh_mark = Mark(SCORE_DATA, LABEL_DATA)
donghanh_mark.filterData()
#donghanh_mark.loadModelParameters(LOGISTIC_REGRESSION_OVR_PARAMS)
#donghanh_mark.loadModelParameters(LDA_MULTINOMIAL_PARAMS)
donghanh_mark.saveModelParameters(MyLogisticRegression_Multiclass_OVR, "LogisticRegression_OVR", LOGISTIC_REGRESSION_OVR_PARAMS)
donghanh_mark.saveModelParameters(MyLogisticRegression_Multiclass_MTN, "LogisticRegression_Multinomial", LOGISTIC_REGRESSION_MTN_PARAMS)
donghanh_mark.saveModelParameters(MyLDA_Multiclass, "LDA_Multinomial", LDA_MULTINOMIAL_PARAMS)
donghanh_mark.saveModelParameters(MyKNN5_Multiclass, "KNN5", KNN5_PARAMS)
donghanh_mark.saveModelParameters(MyKNN1_Multiclass, "KNN1", KNN1_PARAMS)
"""

""" #TEST 29
donghanh_mark = Mark(SCORE_DATA, LABEL_DATA)
#donghanh_mark.filterData()
model = donghanh_mark.loadModelParameters(LOGISTIC_REGRESSION_OVR_PARAMS)
print(donghanh_mark.predictAsDict(model))
"""

#TEST 31
"""
RAW_DATA = "RawFormTest/"
SCORE_DATA = "ScoreDataTest/"
LOGISTIC_REGRESSION_OVR_PARAMS = "SavedModels/LogisticRegression_OVR.txt"
result = readEvaluationFormAsDict(RAW_DATA, LOGISTIC_REGRESSION_OVR_PARAMS)
print(result)
"""

# TEST 32
"""
RAW_DATA = "ComplexRawForm/"
SCORE_DATA = "ComplexScoreData/"
my_form = EvaluationForm("ComplexRawForm/BKHCM_3B.jpg")
my_form.SMALL_SIZE = 100
my_form.saveCells()
"""

# TEST 33, 34
"""
my_image = ComplexMark("ComplexScoreData/BKHCM_3B_0-A.jpg")
print(my_image.getProjection())
"""

# TEST 35
"""
my_image = ComplexMark("ComplexScoreData/BKHCM_3B_0-A.jpg")
imgs = my_image.getSmallerImages()
"""

"""
# TEST 36
RAW_DATA = "ComplexRawForm/"
SCORE_DATA = "ComplexScoreData/"
SCORE_DATA_SPLITTED = "ComplexScoreData_Splitted/"
my_image = ComplexMark("ComplexScoreData/BKHCM_3B_0-A.jpg")
my_image.saveSmallerImages()
"""

# TEST 38
"""
RAW_DATA = "ComplexRawForm/"
SCORE_DATA = "ComplexScoreData/"
SCORE_DATA_SPLITTED = "ComplexScoreData_Splitted/"
print(readComplexEvaluationFormAsDict(RAW_DATA, LOGISTIC_REGRESSION_MTN_PARAMS))
"""
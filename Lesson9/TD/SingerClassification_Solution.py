# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:21:00 2018

@author: ndoannguyen
"""

import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from Constants import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def prepareImageFolder(raw_image_folder, image_folder):
    """
        Exercise 1
        - Create image_folder if not exists
        - Scan raw_image_folder and copy all images from {Singer}/{Image} to image_folder
        - Rename the file like "BaoThy_0.png"
    """
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
    for singer_folder in os.listdir(raw_image_folder):
        for i, image_file in enumerate(os.listdir("%s/%s" % (raw_image_folder, singer_folder))):
            copyfile("%s/%s/%s" % (raw_image_folder, singer_folder, image_file), 
                            "%s/%s_%d%s" % (image_folder, SINGER_NAME_DICTIONARY[singer_folder], i, EXTENSION))
    return

def areTwoEyesHorizontal(left_eye_position, right_eye_position):
    return abs(left_eye_position[1][1] + left_eye_position[1][3]/2 - right_eye_position[1][1] - right_eye_position[1][3]/2 ) <= HORIZONTAL_CHECK
    
def transformImagesToEyesTable(source_folder, destination_folder, destination_data_file, index_list):
    """
        Exercise 9
        - Initialize the DataSet
        - Mkdir if not exists
    """
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)
    data_file = open(destination_data_file, 'w')
    file_list = os.listdir(source_folder)
    count = 0
    for index in index_list:
        image_file = file_list[index]
        face = Face("%s/%s" % (source_folder, image_file))
        if len(face.eye_positions) == 2 and areTwoEyesHorizontal(face.eye_positions[0], face.eye_positions[1]):
            data_file.write(",".join(face.normalized_eyes[0].flatten().astype("str")))
            data_file.write(",")
            data_file.write(",".join(face.normalized_eyes[1].flatten().astype("str")))
            data_file.write(",")
            singer_name = face.image_name[: face.image_name.find("_")]
            data_file.write(str(SINGER_INDEX_DICTIONARY[singer_name]))
            data_file.write(",")
            data_file.write(face.image_name)
            data_file.write("\n")
        count += 1
        if (count % 10 == 0):
            print("%d files processed." % count)
    data_file.close()

def transformImagesToFacesTable(source_folder, destination_folder, destination_data_file, index_list):
    """
        Exercise 9
        - Initialize the DataSet
        - Mkdir if not exists
    """
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)
    data_file = open(destination_data_file, 'w')
    file_list = os.listdir(source_folder)
    count = 0
    for index in index_list:
        image_file = file_list[index]
        face = Face("%s/%s" % (source_folder, image_file))
        if len(face.normalized_faces) >= 1:
            data_file.write(",".join(face.normalized_faces[0].flatten().astype("str")))
            data_file.write(",")
            singer_name = face.image_name[: face.image_name.find("_")]
            data_file.write(str(SINGER_INDEX_DICTIONARY[singer_name]))
            data_file.write(",")
            data_file.write(face.image_name)
            data_file.write("\n")
        count += 1
        if (count % 10 == 0):
            print("%d files processed." % count)
    data_file.close()

class Face:
    """
        Attribute:
        - image_path: path of the image
        - image_name: name of the image
        - image: numpy array representing the image
        - gray_image: numpy array representing the gray image
        - face_positions: list of (x, y, w, h) representing the position of the face
        - faces: list of array representing the face (should be of length 1 in this TD)
        - normalized_faces: resize to 32 x 32
        
        ClassAttribute
        - cascade_path: the cascade_path
    """
    EXTENSION = EXTENSION
    HAARCASCADE_FRONTALFACE = HAARCASCADE_FRONTALFACE_DEFAULT
    HAARCASCADE_EYE = HAARCASCADE_EYE_DEFAULT
    
    HAAR_SCALE_FACTOR = 1.1
    HAAR_MIN_NEIGHBORS = 1
    HAAR_MIN_SIZE = (1, 1)
    
    COLOR_RED = (255, 0, 0)
    BLACK_WHITE_THRESHOLD = 168
    COLOR_WHITE = 255
    FACE_NORMALIZED_SIZE = 64
    EYE_NORMALIZED_SIZE = 32
    
    def __init__(self, image_path):
        """
            Exercise 2 + 5
            - Initialize the image
        """
        self.image_path = image_path
        filename_position = image_path.rfind("/") + 1

        self.image_name = image_path[filename_position:].replace(Face.EXTENSION, "")
        self.image = cv2.imread(image_path)
        self.gray_image = cv2.imread(image_path, 0)
        # Exercise 5
        self.normalized_faces = []
        self.normalizeFaces()
        self.normalized_eyes = []
        self.normalizeEyes()
        
    def draw(self, mode = "full", index = 0):
        """
            Exercise 3 + 4
            - Show an image: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
            - mode is "full" or "face"
        """
        
        # Exercise 3
        if mode == "full_color":
            plt.imshow(self.image, cmap = 'hsv')
            return
        if mode == "full_gray":
            plt.imshow(self.gray_image, cmap = 'gray')
            return
        # Exercise 4
        if mode == "face":
            if not hasattr(self, "faces"):
                self.detectFaces()
            if index >= len(self.faces):
                print("There are only %d face(s) in the image" % len(self.faces))
                return
            plt.imshow(self.faces[index], cmap = 'gray')
        # Exercise 4
        if mode == "face_marked":
            if not hasattr(self, "face_positions"):
                self.detectFaces()
            img = self.gray_image
            if index >= len(self.face_positions):
                print("There are only %d face(s) in the image" % len(self.face_positions))
                return
            x, y, w, h = self.face_positions[index]
            cv2.rectangle(img, (x, y), (x+w, y+h), Face.COLOR_RED, 1)
            plt.imshow(img, cmap = 'gray')
        # Exercise 5
        if mode == "normalized_face":
            if not hasattr(self, "normalized_faces"):
                self.normalizeFaces()
            if index >= len(self.faces):
                print("There are only %d face(s) in the image" % len(self.normalized_faces))
                return
            plt.imshow(self.normalized_faces[index], cmap = 'gray')  
        # Exercise 6
        if mode == "eye":
            if not hasattr(self, "eyes"):
                self.detectEyes()
            if index >= len(self.eyes):
                print("There are only %d eye(s) in the image" % len(self.eyes))
                return
            plt.imshow(self.eyes[index][1], cmap = 'gray')
        if mode == "eye_marked":
            if not hasattr(self, "eyes"):
                self.detectEyes()
            if index >= len(self.eyes):
                print("There are only %d eye(s) in the image" % len(self.eyes))
                return
            i = self.eye_positions[index][0]
            img = self.faces[i]
            x, y, w, h = self.eye_positions[index][1]
            cv2.rectangle(img, (x, y), (x+w, y+h), Face.COLOR_RED, 1)
            plt.imshow(img, cmap = 'gray')
        # Exercise 6
        if mode == "normalized_eye":
            if not hasattr(self, "normalized_eyes"):
                self.normalizeEyes()
            if index >= len(self.normalized_eyes):
                print("There are only %d eye(s) in the image" % len(self.normalized_eyes))
                return
            plt.imshow(self.normalized_eyes[index], cmap = 'gray')  
    
    def detectFaces(self):
        """
            Exercise 4
            - Get positions of the face and assign it to self.face_positions
            - https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-detectmultiscale
        """
        faceCascade = cv2.CascadeClassifier(Face.HAARCASCADE_FRONTALFACE)
        self.face_positions = faceCascade.detectMultiScale(self.gray_image, scaleFactor = Face.HAAR_SCALE_FACTOR, minNeighbors = Face.HAAR_MIN_NEIGHBORS, minSize = Face.HAAR_MIN_SIZE)
        self.faces = []
        for (x, y, w, h) in self.face_positions:
            self.faces.append(self.gray_image[y: y + h, x : x + w])
    
    def normalizeFaces(self):
        """
            Exercise 5
            - Detect the face and normalize it to 32 x 32
        """
        if not hasattr(self, "faces"):
            self.detectFaces()
        if len(self.faces) > 0:
            for i, img in enumerate(self.faces):
                self.normalized_faces.append(cv2.resize(img, (Face.FACE_NORMALIZED_SIZE, Face.FACE_NORMALIZED_SIZE), interpolation = cv2.INTER_LINEAR))
    
    def detectEyes(self):
        """
            Exercise 6
            - Detect the eye
        """
        if not hasattr(self, "faces"):
            self.detectFaces()
        self.eyes = []
        self.eye_positions = []
        eyeCascade = cv2.CascadeClassifier(Face.HAARCASCADE_EYE)
        if len(self.faces) > 0:
            for i, img in enumerate(self.faces):
                eyes = eyeCascade.detectMultiScale(img, scaleFactor = Face.HAAR_SCALE_FACTOR, minNeighbors = Face.HAAR_MIN_NEIGHBORS, minSize = Face.HAAR_MIN_SIZE)
                for e in eyes:
                    self.eye_positions += [(i, e)]
        for (i, (x, y, w, h)) in self.eye_positions:
            self.eyes.append((i, self.faces[i][y : y + h, x : x + h]))

    def normalizeEyes(self):
        """
            Exercise 7
            - Detect the eyes and normalize it to 32 x 32
        """
        if not hasattr(self, "eyes"):
            self.detectEyes()
        if len(self.eyes) > 0:
            for (i, img) in self.eyes:
                self.normalized_eyes.append(cv2.resize(img, (Face.EYE_NORMALIZED_SIZE, Face.EYE_NORMALIZED_SIZE), interpolation = cv2.INTER_LINEAR))
        
class DataSet:
    """
        Attributes:
        - X
        - y
        - names
    """
    
    HORIZONTAL_CHECK = 3
    TRAINING_CLASS_SIZE = 100
    DIMENSION = 32*32*2
    SEPARATOR = ","
    SEUIL = 0.0005
    
    def __init__(self, data_file, selected_columns = None):
        """
            Exercise 10
        """
        raw_data = pd.read_csv(data_file, sep = ",", header = None)
        if selected_columns == None:
            self.X = raw_data.iloc[:, :-2].values
        else:
            self.X = raw_data.iloc[:, selected_columns].values
        self.y = raw_data.iloc[:,-2:-1].T.values[0]
        self.names = raw_data.iloc[:,-1:].T.values[0]
    
    def trainTestSplit(self, test_size = 0.5):
        """
            Exercise 11
        """
        self.X_train, self.X_test, self.y_train, self.y_test, self.names_train, self.names_test = train_test_split(self.X, self.y, self.names, test_size = 0.5, random_state = 0)
                
    def train(self, model):
        """
            Exercise 12
        """
        model.fit(self.X_train, self.y_train)
        return model
    
    def predict(self, model):
        """
            Exercise 13
        """
        return model.predict(self.X_test)
        
    def score(self, model):
        """
            Exercise 14
        """
        model.fit(self.X_train, self.y_train)
        return model.score(self.X_test, self.y_test)
    
    def getConfusionMatrix(self, model):
        """
            Exercise 14
        """
        return confusion_matrix(self.y_test, self.predict(model))
    
    def getSignificantFeatures(self, model, seuil = SEUIL):
        """
            Exercise 15
        """
        return [i for i in range(len(model.coef_[0])) if sum([abs(model.coef_[j,i]) for j in range(len(model.coef_))]) > seuil]
    
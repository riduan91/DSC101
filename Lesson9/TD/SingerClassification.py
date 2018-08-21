# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 14:18:43 2018

@author: ndoannguyen
"""

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
        TODO
    """
    pass

def transformImagesToFacesTable(source_folder, destination_folder, destination_data_file, index_list):
    """
        Exercise 8
        - Read images of index_list in source_folder
        - Check if face recognized
        - Store as a table in a csv file in destination_folder
        TODO
    """
    pass  
    
def transformImagesToEyesTable(source_folder, destination_folder, destination_data_file, index_list):
    """
        Exercise 9
        - Read images of index_list in source_folder
        - Check if 2 eyes are valid
        - Store as a table in a csv file in destination_folder
        TODO
    """
    pass

class Face:
    """
        Attribute:
        - image_path: path of the image
        - image_name: name of the image
        - image: numpy array representing the image
        - gray_image: numpy array representing the gray image
        - face_positions: list of (x, y, w, h) representing the position of the face
        - faces: list of array representing the face (should be of length 1 in this TD)
        - normalized_faces: resize to 64 x 64
        - eye_positions: list of (i, (x, y, w, h)) representing the position of the eyes
        - eyes: list of array representing the eyes 
        - normalized_faces: resize to 32 x 32
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
            TODO
        """
        pass
    
    def draw(self, mode = "full", index = 0):
        """
            Exercise 3 + 4 + 5 + 6 + 7
            - Show image of the face/eyes by matplotlib
            TODO
        """
        pass
    
    def detectFaces(self):
        """
            Exercise 4
            - Get positions of the face and assign it to self.face_positions
            - https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-detectmultiscale
            TODO
        """
        pass
    
    def normalizeFaces(self):
        """
            Exercise 5
            - Detect the face and normalize it to 64 x 64
        """
        if not hasattr(self, "faces"):
            self.detectFaces()
        if len(self.faces) > 0:
            for i, img in enumerate(self.faces):
                self.normalized_faces.append(cv2.resize(img, (Face.FACE_NORMALIZED_SIZE, Face.FACE_NORMALIZED_SIZE), interpolation = cv2.INTER_LINEAR))
    
    def detectEyes(self):
        """
            Exercise 6
            - Detect the eyes
            - https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-detectmultiscale
        """
        pass

    def normalizeEyes(self):
        """
            Exercise 7
            - Detect the eyes and normalize it to 32 x 32
            TODO
        """
        pass
        
class DataSet:
    """
        Attributes:
        - X: The vectors (4096 or 2048 or lower dimensions)
        - y: The target (indices of singers)
        - names: The image's names
    """
    
    HORIZONTAL_CHECK = 3
    TRAINING_CLASS_SIZE = 100
    DIMENSION = 32*32*2
    SEPARATOR = ","
    SEUIL = 0.0005
    
    def __init__(self, data_file, selected_columns = None):
        """
            Exercise 10
            - Initialize the data set by construct the attributes X, y, names
            TODO
        """
        pass
    
    def trainTestSplit(self, test_size = 0.5):
        """
            Exercise 11
            - Split X, y, names to training and test parts
            TODO
        """
        pass
                
    def train(self, model):
        """
            Exercise 12
            - Train the initial model by the training set and return the model
            TODO
        """
        pass
    
    def predict(self, model):
        """
            Exercise 13
            - Predict with the trained model
            TODO
        """
        pass
        
    def score(self, model):
        """
            Exercise 14
            - Get the accuracy by the model on the test set
            TODO
        """
        pass
    
    def getConfusionMatrix(self, model):
        """
            Exercise 14
            - Get the confusion matrix on the test set
            TODO
        """
        pass
    
    def getSignificantFeatures(self, model, seuil = SEUIL):
        """
            Exercise 15
            - Get the most important features by
            TODO
        """
        pass
    
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 15:28:46 2018

@author: ndoannguyen
"""

import os, re, sys
import pandas as pd
DEFAULT_ENCODING = 'utf-8'
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing
import sklearn.mixture
from scipy import sparse
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")


def concatenateDataFiles(raw_data_folder, all_data_file):
    # Exercise 1
    # TODO

def paragraphToSentences(paragraph):
    # Exercise 2
    # TODO

def sentenceToSegments(sentence):
    # Exercise 3
    # TODO

def segmentToUnits(segment):
    # Exercise 4
    # TODO

def loadWordSet(wordlist_filename):
    # Exercise 5
    # TODO
    
def unitsToWords(units, word_set):
    # Exercise 6
    # TODO
 
def getWordFrequencyFromArticles(data_file, word_list_file):
    # Exercise 7
    # TODO

def saveWordFrequencyToFile(data_file, word_list_file, frequency_file, global_frequency_file):
    # Exericse 8
    # TODO

def getExplicativeFeatures(global_frequency_file, frequency_file, lowerbound, upperbound, var_lower_bound):
    # Exercise 9
    # TODO

def articlesToSparseVector(frequency_file, features_dict, coordinates_coding_mode = "0-1"):
    # Exercise 10, 16
    # TODO

def getTitles(frequency_file):
    # Exercise 11
    # TODO

def train(vectors, nb_clusters, model = "KMeans"):
    # Exercice 12, 17, 18
    # TODO

def predict(predictive_model, vectors):
    # Exercise 12, 17, 18
    # TODO

def getClusters(titles, prediction):
    # Exercise 13
    # TODO

def getClusterCenters(predictive_model, vectors, prediction):
    # Exercise 14, 17, 18
    # TODO

def getExplicatveFeaturesForEachCluster(predictive_model, vectors, prediction, explicative_features):
    # Exercise 14, (17, 18)
    # TODO
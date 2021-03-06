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
    raw_file_names = os.listdir(raw_data_folder)
    new_file = open(all_data_file, 'w')
    for i, raw_file_name in enumerate(raw_file_names):
        print("Cleaning %s..." % raw_data_folder + "/" + raw_file_name)
        raw_file = open(raw_data_folder + "/" + raw_file_name, 'r')
        s = raw_file.read()
        data = s.split("\n")
        cleaned_data = "\n".join(filter(lambda data: len(data) > 0 and data.count("\t") >= 4 and len(data[data.rfind("\t\t") + 2 :]) > 100, data))
        new_file.write(cleaned_data)
        if i < len(raw_file_names) - 1:
            new_file.write("\n")
        raw_file.close()
    new_file.close()

def paragraphToSentences(paragraph):
    # Exercise 2
    # TODO
    text = re.sub(' +', ' ', paragraph)
    text = text.replace('\xc2\xa0', '')
    text = text.replace(". ",".<stop>").replace("? ","?<stop>").replace("! ","!<stop>")
    sentences = text.split("<stop>")
    return sentences

def sentenceToSegments(sentence):
    # Exercise 3
    # TODO
    text = re.sub(' +', ' ', sentence)
    text = text.replace(", ",",<stop>").replace(": ",":<stop>").replace("; ",";<stop>").replace("- ",";<stop>").replace(") ",")<stop>").replace(" (","<stop>(")    
    segments = text.split("<stop>")
    return segments

def clean(segment):
    pattern = re.compile(r'[,;$\:\^\=\+\-\"\'\(\)\/\@\*\&\%\“\.{1,}\?\!\d]')
    text = re.sub(pattern, '', segment)
    text = re.sub(' +', ' ', text)
    if len(text) > 0 and text[-1] == " ":
        text = text[:-1]
    if len(text) > 0 and text[0] == " ":
        text = text[1:]
    return text

def segmentToUnits(segment):
    # Exercise 4
    # TODO
    segment = clean(segment)
    units = filter(lambda x: x != "", segment.split(" "))
    return units

def loadWordSet(wordlist_filename):
    # Exercise 5
    # TODO
    word_set = set([])
    wordlist_file = open(wordlist_filename)
    lines = wordlist_file.read().split("\n")
    for line in lines:
        word_set.add(line)
    return word_set
    
def unitsToWords(units, word_set):
    # Exercise 6
    # TODO
    for i in range(len(units)):
        try:
            if units[i].decode('utf-8')[0].isupper() and units[i] in word_set\
            and (i == len(units) - 1 or not units[i + 1].decode('utf-8')[0].isupper()) and \
            (i == 0 or not units[i - 1].decode('utf-8')[0].isupper()):
                units[i] = units[i].decode('utf-8').lower().encode('utf-8')
        except:
            pass
       
    words = []
    current_position = 0
    while current_position < len(units):
        old_position = current_position
        for j in range(min(4, len(units) - current_position), 0, -1):
            current_units = units[current_position: current_position + j]
            trying_word = " ".join(current_units)
            if isCapitalized(current_units):
                words.append(trying_word)
                current_position += j
                break                
            if trying_word in word_set:
                words.append(trying_word)
                current_position += j
                break
        if current_position == old_position:
            words.append(units[current_position])
            current_position += 1
    
    return words

def isCapitalized(units):
    for unit in units:
        try:
            if not unit.decode('utf-8')[0].isupper():
                return False
        except:
            return False
    return True
 
def getWordFrequencyFromArticles(data_file, word_list_file):
    # Exercise 7
    # TODO
    word_set = loadWordSet(word_list_file)
    frequency_list = []
    data = pd.read_csv(data_file, sep="\t\t", header=None)
    short_intro = data.iloc[:, 1]
    content = data.iloc[:, 2]
    for i, s in enumerate(zip(short_intro, content)):
        frequency_for_article = {}
        paragraphs = s[0].split("\t") + s[1].split("\t")[:-1]
        for p in paragraphs:
            sentences = paragraphToSentences(p)
            for st in sentences:
                segments = sentenceToSegments(st)
                for sg in segments:
                    units = segmentToUnits(sg)                        
                    words = unitsToWords(units, word_set)
                    for w in words:
                        if w not in frequency_for_article:
                            frequency_for_article[w] = 1
                        else:
                            frequency_for_article[w] += 1
        frequency_list.append(frequency_for_article)
    return frequency_list

def saveWordFrequencyToFile(data_file, word_list_file, frequency_file, global_frequency_file):
    # Exericse 8
    # TODO
    data = pd.read_csv(data_file, sep="\t\t", header=None)
    titles = data.iloc[:, 0]
    frequency_list = getWordFrequencyFromArticles(data_file, word_list_file)
    f = open(frequency_file, 'w')
    for i, (title, frequency_dict) in enumerate(zip(titles, frequency_list)):
        f.write(title)
        f.write("\t\t")
        f.write("\t".join([item + ":" + str(frequency_dict[item]) for item in frequency_dict]))
        if i < len(titles) - 1:
            f.write("\n")
    f.close()
    
    global_frequency = getGlobalFrequency(frequency_list)
    g = open(global_frequency_file, 'w')
    sorted_keys = sorted(global_frequency.keys(), key = lambda word: -global_frequency[word])
    g.write("\n".join([word + ":" + str(global_frequency[word]) for word in sorted_keys]))
    g.close()

def getGlobalFrequency(frequency_list):
    global_frequency = {}
    for l in frequency_list:
        for word in l:
            if word not in global_frequency:
                global_frequency[word] = 1
            else:
                global_frequency[word] += 1
    return global_frequency

def getExplicativeFeatures(global_frequency_file, frequency_file, lowerbound, upperbound, var_lower_bound):
    # Exercise 9
    data = pd.read_csv(global_frequency_file, sep=":", header=None)
    # features_list = data[(data[1] >= lowerbound) & (data[1] <= upperbound)
    features_list = data[(data[1] >= lowerbound) & (data[1] <= upperbound) ][0]
    features = {}
    for i, feature in enumerate(features_list):
        features[feature] = i
    features_statistics = bagOfWordToNaiveVectors(frequency_file, features)
    var = features_statistics.var(axis = 1)
    stretch_features_keys = filter(lambda x: var[features[x]] >= var_lower_bound, features)
    stretch_features = {}
    for i, k in enumerate(stretch_features_keys):
        stretch_features[k] = i
    return stretch_features

def bagOfWordToNaiveVectors(frequency_file, features_dict):
    datafile = open(frequency_file, 'r')
    data = datafile.read().split("\n")
    datafile.close()
    features_statistics = np.zeros((len(features_dict), len(data)))
    for i, line in enumerate(data):
        content = line.split("\t\t")[1]
        words_and_freqs = content.split("\t")
        for waf in words_and_freqs:
            s = waf.split(":")
            word, freq = s[0], int(s[1])
            if word in features_dict:
                features_statistics[features_dict[word], i] = freq
    return features_statistics

def articlesToSparseVector(frequency_file, features_dict, coordinates_coding_mode = "0-1"):
    # Exercise 10
    datafile = open(frequency_file, 'r')
    data = datafile.read().split("\n")
    datafile.close()
    row_position = []
    column_position = []
    values = []
    titles = [""] * len(data)
    for i, line in enumerate(data):
        title_and_content = line.split("\t\t")
        title = title_and_content[0]
        titles[i] = title
        content = title_and_content[1]
        words_and_freqs = content.split("\t")
        for waf in words_and_freqs:
            s = waf.split(":")
            word = s[0]
            freq = int(s[1])
            if word in features_dict:
                row_position.append(i)
                column_position.append(features_dict[word])
                values.append(freqToCoordinates(freq, coordinates_coding_mode))
    return sparse.coo_matrix((values, (row_position, column_position)), shape=(len(data), len(features_dict)))

def getTitles(frequency_file):
    #Exercise 11
    data = pd.read_csv(frequency_file, sep="\t\t", header = None)
    return list(data[0])

def freqToCoordinates(freq, coordinates_coding_mode = "0-1"):
    # freq is a positive integer
    if coordinates_coding_mode == "0-1":
        return 1
    if coordinates_coding_mode == "sqrt":
        return 1 + np.sqrt(freq)
    if coordinates_coding_mode == "log":
        return np.log(1 + freq)
    return 1

def train(vectors, nb_clusters, model = "KMeans"):
    # Exercice 12, 17, 18
    if model == "KMeans":
        kmeans = KMeans(n_clusters=nb_clusters, random_state=0)
        kmeans.fit(vectors)
        kmeans.model_name = "KMeans"
        return kmeans
    if model == "KMeans_Cosine":
        kmeans = KMeans(n_clusters=nb_clusters, random_state=0)
        vectors = preprocessing.normalize(vectors)
        kmeans.fit(vectors)
        kmeans.model_name = "KMeans_Cosine"
        return kmeans
    if model == "GMM":
        gmm = sklearn.mixture.GMM(n_components = nb_clusters, init_params = 'kmeans')
        gmm.fit(vectors.toarray())
        gmm.model_name = "GMM"
        return gmm
    if model == "Hierarchical_Euclidean":
        hier = AgglomerativeClustering(n_clusters = nb_clusters, affinity = "euclidean")
        hier.model_name = "Hierarchical_Euclidean"
        return hier 
    if model == "Hierarchical_Cosine":
        hier = AgglomerativeClustering(n_clusters = nb_clusters, affinity = "cosine", linkage = "complete")
        hier.model_name = "Hierarchical_Cosine"
        return hier    
    return None

def predict(predictive_model, vectors):
    # Exercise 12, 17, 18
    if predictive_model.model_name == "KMeans":
        return predictive_model.predict(vectors)
    if predictive_model.model_name == "KMeans_Cosine":
        vectors = preprocessing.normalize(vectors)
        return predictive_model.predict(vectors)
    if predictive_model.model_name == "GMM":
        return predictive_model.predict(vectors.toarray())
    if predictive_model.model_name in ["Hierarchical_Euclidean", "Hierarchical_Cosine"]:
        return predictive_model.fit_predict(vectors.toarray())
        return None

def getClusterCenters(predictive_model, vectors, prediction):
    # Exercise 14, 17, 18
    # TODO
    if predictive_model.model_name == "KMeans":
        return predictive_model.cluster_centers_ 
    if predictive_model.model_name == "KMeans_Cosine":
        return predictive_model.cluster_centers_ 
    elif predictive_model.model_name == "GMM":
        return predictive_model.means_
    elif predictive_model.model_name == "Hierarchical_Euclidean":
        vectors = vectors.toarray();
        nb_clusters = max(prediction) + 1
        centers = [ np.average([vectors[j, :] for j in range(len(vectors)) if prediction[j] == i], axis=0) for i in range(nb_clusters) ]
        return np.array(centers)
    elif predictive_model.model_name == "Hierarchical_Cosine":
        vectors = vectors.toarray();
        nb_clusters = max(prediction) + 1
        centers = [ np.average([vectors[j, :] for j in range(len(vectors)) if prediction[j] == i], axis=0) for i in range(nb_clusters) ]
        return np.array(centers)
    else:
        return None

def getClusters(titles, prediction):
    # Exercise 13
    # TODO
    nb_clusters = max(prediction) + 1
    clusters = []
    for i in range(nb_clusters):
        clusters.append([])
    for i in range(len(prediction)):
        clusters[prediction[i]].append(titles[i])
    return clusters

def getExplicatveFeaturesForEachCluster(predictive_model, vectors, prediction, explicative_features):
    # Exercise 14
    # TODO
    centers = getClusterCenters(predictive_model, vectors, prediction)
    nb_clusters = len(centers)
    A = []
    for i in range(nb_clusters):
        A.append(sorted(zip(explicative_features.keys(), [centers[i, explicative_features[k]] for k in explicative_features.keys()]), key = lambda x: -x[1]))
    return A
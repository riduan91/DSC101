from ScienceArticlesClustering import *

"""
raw_data_folder = "RawData/"
all_data_folder = "FullData/"
all_data_file = "FullData/Science2017.txt"
word_list_file = "VietnameseDictionary2/WordList.txt"
frequency_file = "VietnameseDictionary/FrequencyByArticle.txt"
global_frequency_file = "VietnameseDictionary/GlobalFrequency.txt"
"""

raw_data_folder = "RawData2/"
all_data_folder = "FullData2/"
all_data_file = "FullData2/Sport2017.txt"
word_list_file = "VietnameseDictionary2/WordList.txt"
frequency_file = "VietnameseDictionary2/FrequencyBySportArticle.txt"
global_frequency_file = "VietnameseDictionary2/GlobalSportFrequency.txt"

nb_cluster = 4
lowerbound = 200
upperbound = 1000
var_lower_bound = 0.5

print("Making a large data file.")
concatenateDataFiles(raw_data_folder, all_data_folder, all_data_file)

print("Saving word frequency to file.")
saveWordFrequencyToFile(all_data_file, word_list_file, frequency_file, global_frequency_file)

print("Extract titles from articles.")
titles = articleToTitles(frequency_file)

print("Getting features.")
features = getExplicativeFeatures(global_frequency_file, frequency_file, lowerbound, upperbound, var_lower_bound)

print("Extract titles from articles.")
vectors = articlesToSparseVector(frequency_file, features)        
        
print("Transform to vector OK")
print("Training")
kmeans = modelTrainedByKMeans(vectors, nb_cluster)

print("Predicting")
pred = predictByKMeans(kmeans, vectors)

print("Printing centers")
centers = getClusterCenters(kmeans)

print("Priting clusters")
clusters = addArticleTitlesToCluster(titles, pred)

print("Importance of clusters")
A = getExplicatveFeaturesForClusters(kmeans, features)

inertia = kmeans.inertia_

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 05 16:15:31 2018

@author: ndoannguyen
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

DATA_FOLDER = "Data/"
VNINDEX_FILE = DATA_FOLDER + "VNIndex.csv"

vnindex_columns = ["Ticker", "Date", "Open", "High", "Low", "Close", "Vol"]
data = pd.read_csv(VNINDEX_FILE, skiprows = 1, names = vnindex_columns)

ROW = 0
COLUMN = 1

#Exercise 1
data_1 = data.drop(labels = ["Ticker", "Date"], axis = COLUMN)
data_1 = data_1.drop(len(data) - 1, axis = ROW)
target_1 = data["Open"].drop(0)

LRModel1 = LinearRegression()
LRModel1.fit(data_1, target_1)
prediction_1 = LRModel1.predict(data_1)

score_1 = LRModel1.score(data_1, target_1)
#print mean_squared_error(target_1, prediction_1)
print r2_score(target_1, prediction_1)

coefs_1 = LRModel1.coef_, LRModel1.intercept_


#Exercise 2
data_2 = data.drop(labels = ["Ticker", "Date"], axis = COLUMN)
data_2 = data_2.drop(len(data) - 1, axis = ROW)
target_2 = data["Close"].drop(0)

LRModel2 = LinearRegression()
LRModel2.fit(data_2, target_2)
prediction_2 = LRModel2.predict(data_2)

score_2 = LRModel2.score(data_2, target_2) #score = R2_score
#print mean_squared_error(target_2, prediction_2)
print r2_score(target_2, prediction_2)

coefs_2 = LRModel2.coef_, LRModel2.intercept_


#Exercise 3
data_3 = data.drop(labels = ["Ticker", "Date"], axis = COLUMN)
data_3 = data_3.drop(len(data) - 1, axis = ROW)
target_3 = (data["Close"] - data["Open"]).drop(0)

LRModel3 = LinearRegression()
LRModel3.fit(data_3, target_3)
prediction_3 = LRModel3.predict(data_3)

score_3 = LRModel3.score(data_3, target_3) #score = R2_score
#print mean_squared_error(target_3, prediction_3)
print r2_score(target_3, prediction_3)

coefs_3 = LRModel3.coef_, LRModel3.intercept_
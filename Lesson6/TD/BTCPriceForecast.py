# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:08:21 2018

@author: ndoannguyen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split #replaced by model_selection in sklearn 0.19+
from bs4 import BeautifulSoup
import statsmodels.formula.api as sm
from datetime import datetime
import time
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
#import stockstats as sts //Uncommented when installed

SHIFT_NUMBER = 7

COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3', 'rsv_9', 'kdjk_9', 'kdjj_9', 'macd', 'macds', 'macdh', 'rs_6', 'rsi_6', 'rs_12', 'rsi_12', 'wr_6', 'wr_10', 'cci', 'cci_20', 'tr', 'atr', 'dma', 'high_delta', 'um', 'low_delta', 'dm', 'pdm', 'pdm_14', 'pdi_14', 'mdm_14', 'mdi_14', 'dx_14', 'adx', 'adxr', 'trix', 'change', 'vr', 'vr_6_sma']
REFINED_COLUMNS = [ 'cr-ma3', 'kdjk_9', 'macds', 'macdh', 'rs_6', 'rsi_6', 'wr_6', 'atr', 'dma', 'high_delta', 'um', 'dm', 'pdm_14', 'mdm_14', 'mdi_14', 'trix', 'change']

PRICE_FILE = "Data/BTCPrice.csv"
EVENT_FILE = "Data/BTCNews.csv"
TREND_FILE = "Data/BTCTrend.csv"
EVENT_PAGE = "https://99bitcoins.com/price-chart-history/"

# COLUMNS
TIMESTAMP = "Timestamp"
OPEN = "Open"
HIGH = "High"
LOW = "Low"
CLOSE = "Close"
VOLUME_BTC = "Volume (BTC)"
VOLUME_CURRENCY = "Volume (Currency)"
WEIGHTED_PRICE = "Weighted Price"

#TEST_SIZE
TEST_SIZE = 0.5
MAX_PREVIOUS_DAYS = 30

#PARAMETERS
ALPHA = np.log(2)

def readData(datafile):
    # Exercise 1
    # TODO

def getTarget(data, shift_number):
    # Exercise 1
    # TODO

def getpRecentPrices(current_price, p):
    # Exercise 2
    # TODO

def buildModel1(X_train, y_train, X_test, y_test):
    # Exercise 2
    # TODO

def getRMSEList(current_price, future_price, max_p):
    # Exercise 3
    # TODO

def backwardEliminationOnPrice(current_price, future_price, max_p, significant_level = 0.01, nb_random_states = 1):
    # Exercise 4
    # TODO

def getPriceDiff(data, shift_number):
    # Exercise 5
    # TODO

def buildModel2(X_train, y_train, X_test, y_test):
    # Exercise 6
    # TODO

def backwardEliminationOnPriceDiff(current_price, future_price_diff, max_p, significant_level = 0.01, nb_random_states = 1):
    # Exercise 6
    # TODO
    
def addTechnicalIndicators(simple_data):
    # DONE
    stock = sts.StockDataFrame(simple_data)
    stock['cr']
    stock['kdjk']
    stock['kdjd']
    stock['kdjj']
    stock['close_10_sma']
    stock['macd']
    stock['boll']
    stock['rsi_6']
    stock['rsi_12']
    stock['wr_6']
    stock['wr_10']
    stock['cci']
    stock['cci_20']
    stock['tr']
    stock['atr']
    stock['dma']
    stock['adxr']
    stock['close_12_ema']
    stock['trix']
    stock['trix_9_sma']
    stock['vr']
    stock['vr_6_sma']
    new_dataframe = pd.DataFrame(stock).loc[:, COLUMNS]
    transformed_dataframe = new_dataframe.iloc[10: len(new_dataframe) - SHIFT_NUMBER] # Bỏ các hàng khuyết dữ liệu
    scaler = StandardScaler()
    scaler.fit(transformed_dataframe)
    return pd.DataFrame(scaler.transform(transformed_dataframe), columns=COLUMNS)


def readAsStockDataFrame(filename):
    # Exercise 7
    # TODO

def buildModel3(X_train, y_train, X_test, y_test):
    # Exercise 8
    # TODO

def backwardEliminationOnTechnicalData(X, y, significant_level = 0.01, nb_random_states = 1):
    # Exercise 9
    # TODO

def getColumnNameFromIndices(indices):
    # Exercise 9
    # DONE
    return [COLUMNS[i-1] for i in indices]

def buildModel4(X_train, y_train, X_test, y_test):
    # Exercise 9
    # TODO

def buildModel5(X_train, y_train, X_test, y_test):
    # Exercise 10
    # TODO

def buildModel6(X_train, y_train, X_test, y_test, alpha):
    # Exercise 11
    # TODO

def indToNameOnPolynomialRegressionDegree2(initial_columns, selected_indices):
    # Exercise 11
    # TODO

def getModelReadableForm(coefs, important_variables_dict):
    # Exercise 11
    # DONE
    S = ""
    for k, V in sorted(important_variables_dict.items(), key = lambda x: -abs(coefs[x[0]])):
        S += " + (%.2f) * %s " % (coefs[k], V)
    return S

def getTCoefficient(price_data, trend_data):
    # Exercise 12
    # TODO

def getHCoefficient(price_data, news_data, alpha):
    # Exercise 12
    # TODO

def getLogTCoefficient(price_data, trend_data):
    # Exercise 12
    # TODO
    
def buildModel7(X_train, y_train, X_test, y_test):
    # Exercise 13
    # TODO

def buildModel8(X_train, y_train, X_test, y_test):
    # Exercise 14
    # TODO

def buildModel9(X_train, y_train, X_test, y_test, alpha):
    # Exercise 14
    # TODO

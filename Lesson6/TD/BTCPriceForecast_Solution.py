# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:35:08 2018

@author: ndoannguyen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split #replaced by model_selection in sklearn 0.19+
import statsmodels.formula.api as sm
from datetime import datetime
import time
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import stockstats as sts

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
    return pd.read_csv(datafile, sep=",")

def getTarget(data, shift_number):
    # Exercise 1
    return np.concatenate((data.loc[:, CLOSE].values[shift_number:], [0] * shift_number))

def getpRecentPrices(current_price, p):
    # Exercise 2
    previous_price = []
    N = len(current_price)
    for i in range(0, p):
        previous_price.append(np.concatenate(([0]*i, current_price[0:N-i])))
    return np.array(previous_price).T

def buildModel1(X_train, y_train, X_test, y_test):
    # Exercise 2
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model, model.coef_, model.intercept_, y_predict, np.sqrt(mean_squared_error(y_test, y_predict)), model.score(X_test, y_test)

def getRMSEList(current_price, future_price, max_p):
    # Exercise 3
    RMSEList = [0] * (max_p)
    for p in range(1, max_p + 1):
        X = getpRecentPrices(current_price, p)
        X = X[p-1: len(X) - SHIFT_NUMBER] #Bỏ các hàng khuyết
        y = future_price[p-1: len(future_price) - SHIFT_NUMBER]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)
        RMSE = buildModel1(X_train, y_train, X_test, y_test)[4]
        RMSEList[p-1] = RMSE
    plt.plot(range(1, max_p + 1), RMSEList)
    return RMSEList

""" #PART 1 of exercise 4
def backwardEliminationOnPrice(current_price, future_price, max_p, significant_level = 0.01):
    # Exercise 4
    X = getpRecentPrices(current_price, max_p)
    X = np.concatenate([np.ones((len(current_price), 1)), X], axis = 1) # Thêm cột hệ số tự do vì OLS cần nó
    X = X[max_p-1: len(X) - SHIFT_NUMBER] #Bỏ các hàng khuyết
    y = future_price[max_p-1: len(future_price) - SHIFT_NUMBER]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)
    selected = range(max_p + 1)
    for p in range(max_p + 1):
        regressor_OLS = sm.OLS(y_train, X_train).fit()
        ind = np.argmax(regressor_OLS.pvalues)
        if regressor_OLS.pvalues[ind] < significant_level:
            break
        X_train = np.delete(X_train, ind, 1)
        X_test = np.delete(X_test, ind, 1)
        selected = np.delete(selected, ind)    
    return selected
"""

def backwardEliminationOnPrice(current_price, future_price, max_p, significant_level = 0.01, nb_random_states = 1):
    X = getpRecentPrices(current_price, max_p)
    X = np.concatenate([np.ones((len(current_price), 1)), X], axis = 1) # Thêm cột hệ số tự do vì OLS cần nó
    X = X[max_p-1: len(X) - SHIFT_NUMBER] #Bỏ các hàng khuyết
    y = future_price[max_p-1: len(future_price) - SHIFT_NUMBER]
    final_selected = set(range(max_p + 1))
    for state in range(nb_random_states):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=state)
        selected = range(max_p + 1)
        for p in range(max_p + 1):
            regressor_OLS = sm.OLS(y_train, X_train).fit()
            ind = np.argmax(regressor_OLS.pvalues)
            if regressor_OLS.pvalues[ind] < significant_level:
                break
            X_train = np.delete(X_train, ind, 1)
            X_test = np.delete(X_test, ind, 1)
            selected = np.delete(selected, ind)   
        final_selected = final_selected.intersection(set(selected))
    return list(final_selected)

def getPriceDiff(data, shift_number):
    # Exercise 5
    return np.concatenate((data.loc[:, CLOSE].values[shift_number:] * 100 / data.loc[:, CLOSE].values[: len(data) - shift_number] - 100, [0] * shift_number))

def buildModel2(X_train, y_train, X_test, y_test):
    # Exercise 6
    return buildModel1(X_train, y_train, X_test, y_test)

def backwardEliminationOnPriceDiff(current_price, future_price_diff, max_p, significant_level = 0.01, nb_random_states = 1):
    # Exercise 6
    return backwardEliminationOnPrice(current_price, future_price_diff, max_p, significant_level, nb_random_states)

""" #Exercise 7, part 1
def readAsStockDataFrame(filename):
    # Exercise 7
    data = pd.read_csv(filename, sep=",", names=["timestamp", "open", "high", "low", "close", "volume", "volume currency", "weighted price"], skiprows = 1)
    return data.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]]
"""

def addTechnicalIndicators(simple_data):
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
    data = pd.read_csv(filename, sep=",", names=["timestamp", "open", "high", "low", "close", "volume", "volume currency", "weighted price"], skiprows = 1)
    data = data.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]]
    return addTechnicalIndicators(data)

def buildModel3(X_train, y_train, X_test, y_test):
    # Exercise 8
    return buildModel1(X_train, y_train, X_test, y_test)

def backwardEliminationOnTechnicalData(X, y, significant_level = 0.01, nb_random_states = 1):
    # Exercise 9
    X = np.concatenate([np.ones((len(X), 1)), X], axis = 1) # Thêm cột hệ số tự do vì OLS cần nó
    y = y[10: len(y) - SHIFT_NUMBER]
    nb_vars = len(X[0])
    final_selected = {}
    for i in range(nb_vars):
        final_selected[i] = 0
    for state in range(nb_random_states):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=state)
        selected = range(nb_vars)
        for p in range(nb_vars):
            regressor_OLS = sm.OLS(y_train, X_train).fit()
            ind = np.argmax(regressor_OLS.pvalues)
            if regressor_OLS.pvalues[ind] < significant_level:
                break
            X_train = np.delete(X_train, ind, 1)
            X_test = np.delete(X_test, ind, 1)
            selected = np.delete(selected, ind)   
        for i in selected:
            final_selected[i] += 1
    return filter(lambda x: final_selected[x] >= nb_random_states * 2./3, final_selected.keys())

def getColumnNameFromIndices(indices):
    return [COLUMNS[i-1] for i in indices]

def buildModel4(X_train, y_train, X_test, y_test):
    # Exercise 9
    return buildModel1(X_train, y_train, X_test, y_test)

def buildModel5(X_train, y_train, X_test, y_test):
    # Exercise 10
    poly = PolynomialFeatures(2)
    X_train_transformed = poly.fit_transform(X_train)
    X_test_transformed = poly.fit_transform(X_test)
    model = LinearRegression(fit_intercept = False)
    model.fit(X_train_transformed, y_train)
    y_predict = model.predict(X_test_transformed)
    return model, model.coef_, model.intercept_, y_predict, np.sqrt(mean_squared_error(y_test, y_predict)), model.score(X_test_transformed, y_test)

def buildModel6(X_train, y_train, X_test, y_test, alpha):
    # Exercise 11
    poly = PolynomialFeatures(2)
    X_train_transformed = poly.fit_transform(X_train)
    X_test_transformed = poly.fit_transform(X_test)
    model = Lasso(fit_intercept = False, max_iter = 100000, alpha = alpha)
    model.fit(X_train_transformed, y_train)
    print "Nb_iterations used: ", model.n_iter_
    y_predict = model.predict(X_test_transformed)
    return model, model.coef_, model.intercept_, y_predict, np.sqrt(mean_squared_error(y_test, y_predict)), model.score(X_test_transformed, y_test)

def indToNameOnPolynomialRegressionDegree2(initial_columns, selected_indices):
    # Exercise 11
    columns = ['1'] + initial_columns
    res = {}
    count = 0
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            if count in selected_indices:
                res[count] = columns[i] + ' * ' + columns[j]
            count += 1
    return res

def getModelReadableForm(coefs, important_variables_dict):
    # Exercise 11
    # FINISHED
    S = ""
    for k, V in sorted(important_variables_dict.items(), key = lambda x: -abs(coefs[x[0]])):
        S += " + (%.2f) * %s " % (coefs[k], V)
    return S

def getTCoefficient(price_data, trend_data):
    # Exercise 12
    result = np.array([0] * len(price_data))
    date_dict = {}
    for j, date in enumerate(trend_data.iloc[:, 0].values):
        timestamp = int(time.mktime(datetime.strptime(date, '%d/%m/%Y').timetuple()))
        date_dict[timestamp] = j
    for i, data_date in enumerate(price_data.iloc[:, 0].values):
        timestamp = int(time.mktime(datetime.strptime(data_date, '%d/%m/%Y %H:%M').timetuple()))
        if timestamp in date_dict:
            ind = date_dict[timestamp]
            result[i] += trend_data.iloc[ind, 1]
    return (result - np.mean(result[1:]))/np.std(result[1:]) #Vì chỉ số đầu tiên bằng 0

def getHCoefficient(price_data, news_data, alpha):
    # Exercise 12
    result = [0] * len(price_data)
    for j, date in enumerate(news_data.iloc[:, 0].values):
        datetime1 = datetime.strptime(date, '%d/%m/%Y')
        for i, data_date in enumerate(price_data.iloc[:, 0].values):
            datetime2 = datetime.strptime(data_date, '%d/%m/%Y %H:%M')
            if datetime2 >= datetime1:
                #result[i] += event_array.iloc[j, 3] * max(1 - (datetime2 - datetime1).days * alpha, 0)
                result[i] += news_data.iloc[j, 3] * np.exp(-alpha * (datetime2 - datetime1).days)
    result = np.array(result)
    return (result - np.mean(result))/np.std(result)

def getLogTCoefficient(price_data, trend_data):
    # Exercise 12
    result = np.array([0.0] * len(price_data))
    date_dict = {}
    for j, date in enumerate(trend_data.iloc[:, 0].values):
        timestamp = int(time.mktime(datetime.strptime(date, '%d/%m/%Y').timetuple()))
        date_dict[timestamp] = j
    for i, data_date in enumerate(price_data.iloc[:, 0].values):
        timestamp = int(time.mktime(datetime.strptime(data_date, '%d/%m/%Y %H:%M').timetuple()))
        if timestamp in date_dict:
            ind = date_dict[timestamp]
            trend = trend_data.iloc[ind, 1]
            if trend > 0:
                result[i] = np.log(trend)
    return (result - np.mean(result[1:]))/np.std(result[1:])

def buildModel7(X_train, y_train, X_test, y_test):
    # Exercise 13
    return buildModel5(X_train, y_train, X_test, y_test)

def buildModel8(X_train, y_train, X_test, y_test):
    # Exercise 14
    return buildModel5(X_train, y_train, X_test, y_test)

def buildModel9(X_train, y_train, X_test, y_test, alpha):
    # Exercise 14
    return buildModel6(X_train, y_train, X_test, y_test, alpha)

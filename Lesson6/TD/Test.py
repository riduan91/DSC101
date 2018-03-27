# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:29:51 2018

@author: ndoannguyen
"""

#TEST

from BTCPriceForecast_Solution import *

# Exercise 1
"""
price_data = readData(PRICE_FILE) # File BTCPrice.csv 
print price_data.head(10)
trend_data = readData(TREND_FILE) # File BTCTrend.csv
print trend_data.head()
target = getTarget(price_data, SHIFT_NUMBER) # Kết quả là một array bắt đầu bởi giá của ngày thứ 7, kết thúc bởi 7 số 0
print pd.DataFrame(getTarget(price_data, SHIFT_NUMBER)).T # Biểu diễn dưới dạng frame để dễ nhìn
"""

# Exericse 2
"""
price_data = readData(PRICE_FILE)
target = getTarget(price_data, SHIFT_NUMBER)
current_price = price_data.loc[:, CLOSE]
X1 = getpRecentPrices(current_price, 10)
"""

"""
price_data = readData(PRICE_FILE)
current_price = price_data.loc[:, CLOSE]
X1 = getpRecentPrices(current_price, 10)
y1 = getTarget(price_data, SHIFT_NUMBER)
X1 = X1[9: len(X1) - SHIFT_NUMBER] #Bỏ các hàng khuyết dữ liệu: 9 hàng đầu và 7 hàng cuối cùng
y1 = y1[9: len(y1) - SHIFT_NUMBER]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=TEST_SIZE, random_state=0)
model1, coefs1, intercept1, y1_predict, RMSE1, score1 = buildModel1(X1_train, y1_train, X1_test, y1_test)
print RMSE1
"""

# Exercise 3
"""
price_data = readData(PRICE_FILE)
current_price = price_data.loc[:, CLOSE]
future_price = getTarget(price_data, SHIFT_NUMBER)
RMSEList = getRMSEList(current_price, future_price, 20)
pd.DataFrame(RMSEList).T # Phần tử 0: RMSE khi chọn p=1, etc.
"""

# Exercise 4
"""
price_data = readData(PRICE_FILE)
current_price = price_data.loc[:, CLOSE]
future_price = getTarget(price_data, SHIFT_NUMBER)
significant_variables = backwardEliminationOnPrice(current_price, future_price, 20)
print significant_variables
"""

"""
price_data = readData(PRICE_FILE)
current_price = price_data.loc[:, CLOSE]
future_price = getTarget(price_data, SHIFT_NUMBER)
significant_variables = backwardEliminationOnPrice(current_price, future_price, 20, significant_level = 0.01, nb_random_states = 20)
print significant_variables
"""

# Exercise 5
"""
price_data = readData(PRICE_FILE) # File BTCPrice.csv 
price_diff = getPriceDiff(price_data, SHIFT_NUMBER) # Kết quả là một array
print pd.DataFrame(price_diff).T # Biểu diễn ở dạng DataFrame
"""

# Exercise 6
"""
price_data = readData(PRICE_FILE)
current_price = price_data.loc[:, CLOSE]
future_price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X2 = getpRecentPrices(current_price, 10)
X2 = X2[9: len(X2) - SHIFT_NUMBER] #Bỏ các hàng khuyết dữ liệu: 9 hàng đầu và 7 hàng cuối cùng
y2 = future_price_diff[9: len(future_price_diff) - SHIFT_NUMBER]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=TEST_SIZE, random_state=0)
model2, coefs2, intercept2, y1_predic2, RMSE1, score2 = buildModel2(X2_train, y2_train, X2_test, y2_test)
print coefs2, intercept2

plt.plot(y2[-200:], label="price_diff")
plt.plot(model2.predict(X2)[-200:], label="predicted_price_diff")
plt.legend()
"""

# Exercise 7
"""
data = readAsStockDataFrame(PRICE_FILE)
print data.head()
"""

# Exercise 8
"""
technical_data = readAsStockDataFrame(PRICE_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X3 = technical_data
y3 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=TEST_SIZE, random_state=0)
model3, coefs3, intercept3, y3_predict, RMSE3, score3 = buildModel1(X3_train, y3_train, X3_test, y3_test)
print coefs3, intercept3
"""

# Exercise 9
"""
technical_data = readAsStockDataFrame(PRICE_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
variable_selection = backwardEliminationOnTechnicalData(technical_data, price_diff, 0.1, 50)
print variable_selection
"""

"""
REFINED_COLUMNS = getColumnNameFromIndices(variable_selection)
print REFINED_COLUMNS
"""

"""
technical_data = readAsStockDataFrame(PRICE_FILE)
REFINED_COLUMNS = getColumnNameFromIndices(variable_selection)
refined_technical_data = technical_data.loc[:, REFINED_COLUMNS]
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X4 = refined_technical_data
y4 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=TEST_SIZE, random_state=0)
model4, coefs4, intercept4, y4_predict, RMSE4, score4 = buildModel4(X4_train, y4_train, X4_test, y4_test)
print coefs4, intercept4
"""

# Exercise 10
"""
technical_data = readAsStockDataFrame(PRICE_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X5 = technical_data[:]
y5 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=TEST_SIZE, random_state=0)
model5, coefs5, intercept5, y5_predict, RMSE5, score5 = buildModel5(X5_train, y5_train, X5_test, y5_test)
"""

# Exercise 11
"""
technical_data = readAsStockDataFrame(PRICE_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X6 = technical_data[:]
y6 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=TEST_SIZE, random_state=0)
alpha6 = 0.1
model6, coefs6, intercept6, y6_predict, RMSE6, score6 = buildModel6(X6_train, y6_train, X6_test, y6_test, alpha6)
print RMSE6
important_variables = filter(lambda i: abs(coefs6[i]) > 1e-2, range(len(model6.coef_)))
print pd.DataFrame(important_variables).T
important_variables_dict = indToNameOnPolynomialRegressionDegree2(COLUMNS, important_variables) # Một từ điển
print pd.DataFrame([important_variables_dict.keys(), important_variables_dict.values()]).T
print getModelReadableForm(model6.coef_, important_variables_dict)

plt.plot(y6[-200:], label="price_diff")
plt.plot(model6.predict(PolynomialFeatures(2).fit_transform(X6)[-200:]), label="predicted_price_diff")
plt.legend()
"""

# Exercise 12
"""
price_data = readData(PRICE_FILE)
trend_data = readData(TREND_FILE)
TCoefficient = getTCoefficient(price_data, trend_data)
print pd.DataFrame(TCoefficient).T 

price_data = readData(PRICE_FILE)
news_data = readData(EVENT_FILE)
HCoefficient = getHCoefficient(price_data, news_data, ALPHA)
print pd.DataFrame(HCoefficient).T # Các giá trị đầu tiên bằng nhau vì không có sự kiện

price_data = readData(PRICE_FILE)
trend_data = readData(TREND_FILE)
logTCoefficient = getLogTCoefficient(price_data, trend_data)
print pd.DataFrame(logTCoefficient).T 
"""

# Exercise 13
"""
price_data = readData(PRICE_FILE)
news_data = readData(EVENT_FILE)
trend_data = readData(TREND_FILE)
TCoefficient = getTCoefficient(price_data, trend_data)
TCoefficient = np.array(TCoefficient).reshape(len(TCoefficient), 1)
HCoefficient = getHCoefficient(price_data, news_data, ALPHA)
HCoefficient = np.array(HCoefficient).reshape(len(HCoefficient), 1)
logTCoefficient = getLogTCoefficient(price_data, trend_data)
logTCoefficient = np.array(logTCoefficient).reshape(len(logTCoefficient), 1)
information_data = np.concatenate([TCoefficient, logTCoefficient, HCoefficient], axis = 1)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X7 = information_data[10: len(price_diff) - SHIFT_NUMBER] #Lấy từ ngày đầu tiên có thông tin
y7 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, test_size=TEST_SIZE, random_state=0)

model7, coefs7, intercept7, y7_predict, RMSE7, score7 = buildModel7(X7_train, y7_train, X7_test, y7_test)
print RMSE7
"""

# Exercise 14
"""
price_data = readData(PRICE_FILE)
news_data = readData(EVENT_FILE)
trend_data = readData(TREND_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)

ALPHA = 0.1

TCoefficient = getTCoefficient(price_data, trend_data)
TCoefficient = np.array(TCoefficient).reshape(len(TCoefficient), 1)
HCoefficient = getHCoefficient(price_data, news_data, ALPHA)
HCoefficient = np.array(HCoefficient).reshape(len(HCoefficient), 1)
logTCoefficient = getLogTCoefficient(price_data, trend_data)
logTCoefficient = np.array(logTCoefficient).reshape(len(logTCoefficient), 1)
information_data = np.concatenate([TCoefficient, logTCoefficient, HCoefficient], axis = 1)
X9_info = information_data[10: len(price_diff) - SHIFT_NUMBER] 

technical_data = readAsStockDataFrame(PRICE_FILE)
price_diff = getPriceDiff(price_data, SHIFT_NUMBER)
X9_tech = technical_data

X9 = np.concatenate([X9_tech, X9_info], axis = 1)

FULL_COLUMNS = COLUMNS + ["T_n", "Log_T_n", "H_n"]

y9 = price_diff[10: len(price_diff) - SHIFT_NUMBER]
X9_train, X9_test, y9_train, y9_test = train_test_split(X9, y9, test_size=TEST_SIZE, random_state=0)

model9, coefs9, intercept9, y9_predict, RMSE9, score9 = buildModel9(X9_train, y9_train, X9_test, y9_test, 0.2)
print RMSE9
"""
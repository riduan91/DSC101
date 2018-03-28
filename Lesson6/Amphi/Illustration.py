# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:21:32 2018

@author: ndoannguyen
"""

import numpy as np
np.random.seed(0)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

TEST_SIZE = 0.2
TRAIN_TEST_SPLIT_RANDOM_STATE = 0

"""
w = np.array([1.5, -2.1, 4, 0, -1.3])
b = 2
sigma = 0.1

N = 10000

X = np.random.uniform(-5, 5, [N, D])
epsilon = np.random.normal(0, sigma, N)
y = X.dot(w) + b + epsilon


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print lr_model.coef_, lr_model.intercept_

y_predict_train = lr_model.predict(X_train)
sigma_hat = mean_squared_error(y_train, y_predict_train)

print sigma_hat

y_predict =  lr_model.predict(X_test)
RMSE = mean_squared_error(y_test, y_predict)

print RMSE

#sklearn.cross_validation.
lr_model = LinearRegression()
# scores = cross_val_score(lr_model, X, y, cv=5)
scores = cross_val_score(lr_model, X, y, scoring=lambda lr_model, X, y: mean_squared_error(y, lr_model.predict(X)), cv=5)

print scores
"""

###############

"""
sigma = 0.1

N = 50

X = np.random.uniform(-1, 1, [N, 1])
epsilon = np.random.normal(0, sigma, N)
y = np.sin(np.pi * X[:,0]) + epsilon

DEGREES = range(1, 25)
train_errs = []
test_errs = []

for degree in DEGREES:
    print("Fitting by a polynomial of degree %d" % degree)
    poly = PolynomialFeatures(degree = degree)
    new_X = poly.fit_transform(X)
    lr_model = LinearRegression(fit_intercept = False)
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=TEST_SIZE, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)
    lr_model.fit(X_train, y_train)
    train_error = mean_squared_error(y_train, lr_model.predict(X_train))
    test_error = mean_squared_error(y_test, lr_model.predict(X_test))
    print("Train error: %f; Test error: %f." % (train_error, test_error))
    train_errs.append(train_error)
    test_errs.append(test_error)

plt.plot(DEGREES, train_errs, 'r', label="Error on training set")
plt.plot(DEGREES, test_errs, 'b', label="Error on test set")   
"""

"""
sigma = 0.1

N = 50

X = np.random.uniform(-1, 1, [N, 1])
epsilon = np.random.normal(0, sigma, N)
y = np.sin(np.pi * X[:,0]) + epsilon

DEGREES = range(1, 20)
mean_scores = []
std_scores = []

for degree in DEGREES:
    print("Fitting by a polynomial of degree %d" % degree)
    poly = PolynomialFeatures(degree = degree)
    new_X = poly.fit_transform(X)
    lr_model = LinearRegression(fit_intercept = False)
    scores = cross_val_score(lr_model, new_X, y, scoring=lambda lr_model, X, y: mean_squared_error(y, lr_model.predict(X)), cv=20)
    print("Mean of scores: %f" % np.mean(scores))
    mean_scores.append(np.mean(scores))
    std_scores.append(np.std(scores))

plt.plot(DEGREES, mean_scores, 'r', label="Mean of scores") 
plt.plot(DEGREES, std_scores, 'b', label="Std dev of scores") 
"""

"""
sigma = 0.1
N = 50

X = np.random.uniform(-1, 1, [N, 1])
epsilon = np.random.normal(0, sigma, N)
y = np.sin(np.pi * X[:,0]) + epsilon

interval = np.linspace(-1, 1, 100)
plt.plot(interval, np.sin(np.pi * interval), 'r', label = "Real function")

#DEGREE = 2
DEGREE = 20
NB_STATES = 10
poly = PolynomialFeatures(degree = DEGREE)
new_X = poly.fit_transform(X)

for i in range(NB_STATES):
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=TEST_SIZE, random_state=i)
    lr_model = LinearRegression(fit_intercept = False)
    lr_model.fit(X_train, y_train)
    interval_transformed = poly.fit_transform(interval.reshape(-1, 1))
    plt.plot(interval, lr_model.predict(PolynomialFeatures(degree = DEGREE).fit_transform(interval.reshape(-1, 1))), 'b')
    #plt.plot(interval[10:90], lr_model.predict(PolynomialFeatures(degree = DEGREE).fit_transform(interval.reshape(-1, 1)))[10:90], 'b')

plt.legend()
"""

"""
sigma = 0.1
N = 50

X = np.random.uniform(-1, 1, [N, 1])
epsilon = np.random.normal(0, sigma, N)
y = np.sin(np.pi * X[:,0]) + epsilon

interval = np.linspace(-1, 1, 100)
plt.plot(interval, np.sin(np.pi * interval), 'r', label = "Real function")

#DEGREE = 2
DEGREE = 8
NB_STATES = 10
poly = PolynomialFeatures(degree = DEGREE)
new_X = poly.fit_transform(X)

for i in range(NB_STATES):
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=TEST_SIZE, random_state=i)
    lr_model = Ridge(fit_intercept = False, alpha = 0.01)
    lr_model.fit(X_train, y_train)
    print lr_model.coef_
    print mean_squared_error(y_test, lr_model.predict(X_test))
    interval_transformed = poly.fit_transform(interval.reshape(-1, 1))
    plt.plot(interval, lr_model.predict(PolynomialFeatures(degree = DEGREE).fit_transform(interval.reshape(-1, 1))), 'b')

plt.legend()
"""

"""
sigma = 0.1
N = 50

X = np.random.uniform(-1, 1, [N, 1])
epsilon = np.random.normal(0, sigma, N)
y = np.sin(np.pi * X[:,0]) + epsilon

interval = np.linspace(-1, 1, 100)
plt.plot(interval, np.sin(np.pi * interval), 'r', label = "Real function")

#DEGREE = 2
DEGREE = 8
NB_STATES = 10
poly = PolynomialFeatures(degree = DEGREE)
new_X = poly.fit_transform(X)

for i in range(NB_STATES):
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=TEST_SIZE, random_state=i)
    lr_model = Lasso(fit_intercept = False, alpha = 0.01)
    lr_model.fit(X_train, y_train)
    print lr_model.coef_
    print mean_squared_error(y_test, lr_model.predict(X_test))
    interval_transformed = poly.fit_transform(interval.reshape(-1, 1))
    plt.plot(interval, lr_model.predict(PolynomialFeatures(degree = DEGREE).fit_transform(interval.reshape(-1, 1))), 'b')

plt.legend()
"""
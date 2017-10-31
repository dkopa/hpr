#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:48:12 2017

@author: hasnat
"""

import itertools

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession()

train = pd.read_csv('input/train.csv')
# DataFrame.head(train)
print('Shape of the train data with all features:', train.shape)

# Select data without the string type
train = train.select_dtypes(exclude=['object'])
print("")
print('Shape of the train data with numerical features:', train.shape)

# Drop a first column ('Id') and replace NA with 0
train.drop('Id',axis = 1, inplace = True)
train.fillna(0,inplace=True)

test = pd.read_csv('input/test.csv')
test = test.select_dtypes(exclude=['object'])
ID = test.Id
test.fillna(0,inplace=True)
test.drop('Id',axis = 1, inplace = True)

print("")
print("List of features contained our dataset:",list(train.columns))

from sklearn.ensemble import IsolationForest

clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", train.shape[0])

col_train = list(train.columns)
col_train_bis = list(train.columns)

col_train_bis.remove('SalePrice')

mat_train = np.matrix(train)
mat_test  = np.matrix(test)
mat_new = np.matrix(train.drop('SalePrice',axis = 1))
mat_y = np.array(train.SalePrice).reshape((1314,1))

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)

prepro = MinMaxScaler()
prepro.fit(mat_train)

prepro_test = MinMaxScaler()
prepro_test.fit(mat_new)

train = pd.DataFrame(prepro.transform(mat_train),columns = col_train)
test  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)

train.head()

X_train = train.iloc[:,0:-1]
y_train = np.log1p(mat_y)

##
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

# Ridge
alpha_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
rmse_ridge = [rmse_cv(Ridge(alpha=alphas)).mean() for alphas in alpha_vals]
print('rmse [Ridge]: ' + str(np.min(pp)))

#myRidge = Ridge(alpha=alpha_vals[np.argmin(rmse_ridge)])
#myRidge.fit(X_train, y_train)
#preds = np.expm1(myRidge.predict(test))

#solution = pd.DataFrame({"id":ID, "SalePrice":preds})
#solution.to_csv("ridge_sol.csv", index = False)

# Lasso
alpha_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
rmse_lasso = [rmse_cv(Lasso(alpha=alphas)).mean() for alphas in alpha_vals]
print('rmse [Lasso]: ' + str(np.min(rmse_lasso)))

# SVC
from sklearn.svm import SVR

c_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
rmse_svr = [rmse_cv(SVR(C=c_val)).mean() for c_val in c_vals]
print('rmse [SVR]: ' + str(np.min(rmse_svr)))

# Linear regression
from sklearn.linear_model import LinearRegression
#lr = LinearRegression()
#lr.fit(X_train, y_train)
rmse_lr = rmse_cv(LinearRegression()).mean()
print('rmse [LR]: ' + str(np.min(rmse_lr)))
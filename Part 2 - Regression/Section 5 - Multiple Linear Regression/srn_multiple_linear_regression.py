#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:08:11 2020

@author: sowrensen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Processing the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# New procedure to use OneHotEncoder directly without using LabelEncoder
#transformer = ColumnTransformer(
#        transformers=[("MLR", OneHotEncoder(), [3])],
#        remainder='passthrough')
#X = transformer.fit_transform(X.tolist())


# Avoiding dummy variable trap, usually the library takes care of it
X = X[:, 1:]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("Score of training set: {:.2f}".format(reg.score(X_train, y_train)))
print("Score of test set: {:.2f}".format(reg.score(X_test, y_test)))

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

# Append a column of 1 shaped (50, 1) to denote as X0 for the constant b0
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
reg_ols = sm.OLS(endog=y, exog=X_opt).fit()
reg_ols.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
reg_ols = sm.OLS(endog=y, exog=X_opt).fit()
reg_ols.summary()

X_opt = X[:, [0, 3, 4, 5]]
reg_ols = sm.OLS(endog=y, exog=X_opt).fit()
reg_ols.summary()

X_opt = X[:, [0, 3, 5]]
reg_ols = sm.OLS(endog=y, exog=X_opt).fit()
reg_ols.summary()

X_opt = X[:, [0, 3]]
reg_ols = sm.OLS(endog=y, exog=X_opt).fit()
reg_ols.summary()
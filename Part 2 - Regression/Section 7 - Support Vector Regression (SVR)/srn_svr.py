#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:36:24 2020

@author: sowrensen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature scaling
# SVR doesn't include feature scaling by default,
# so this has to be done manually
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
# Now, I had to reshape y before feature scaling else it
# will not be able to fit into the StandardScaler, and
# later I am sending it back to it's original shape.
y = sc_y.fit_transform(np.array([y]).reshape(10, 1)).reshape(-1,)

# Fitting the regression model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result with the regression model
y_pred = sc_y.inverse_transform(
        regressor.predict(sc_X.transform(np.array([[6.5]]).reshape(1, 1))))

# Visualizing the regression result
plt.scatter(X, y, marker='*', color='r')
plt.plot(X, regressor.predict(X), color='b')
plt.title("Truth or Bluff (SVR Model)")
plt.xlabel("Positions Level")
plt.ylabel("Salary")
plt.show()

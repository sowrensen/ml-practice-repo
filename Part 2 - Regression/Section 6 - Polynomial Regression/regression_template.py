#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:29:42 2020

@author: sowrensen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# =============================================================================
# # Splitting the dataset into training and test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=0)
# 
# # Feature scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# # Now, I had to reshape y before feature scaling else it
# # will not be able to fit into the StandardScaler, and
# # later I am sending it back to it's original shape.
# y = sc_y.fit_transform(np.array([y]).reshape(10, 1)).reshape(-1,)
# =============================================================================

# Fitting the regression model to the dataset


# Predicting a new result with the regression model
y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))

# Visualizing the regression result
plt.scatter(X, y, color='r')
plt.plot(X, regressor.predict(X), color='b')
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Positions Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the regression result (for higher dimension and smooth curves)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='r')
plt.plot(X_grid, regressor.predict(X_grid), color='b')
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Positions")
plt.ylabel("Salaries")
plt.show()
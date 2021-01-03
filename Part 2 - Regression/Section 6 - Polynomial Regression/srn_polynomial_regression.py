#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:15:57 2020

@author: sowrensen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing linear regression
plt.scatter(X, y, color='r')
plt.plot(X, lin_reg.predict(X), color='b')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Positions")
plt.ylabel("Salaries")
plt.show()

# Visualizing polynomial regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='r')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='b')
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Positions")
plt.ylabel("Salaries")
plt.show()

# Predicting a new result with linear regression
# Not similar to the tutorial due to upgraded version
# of sklearn doesn't support a single value
lin_reg.predict(np.array([6.5]).reshape(1, 1))

# Predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1, 1)))
















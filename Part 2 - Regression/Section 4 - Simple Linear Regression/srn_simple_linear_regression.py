#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:36:35 2020

@author: sowrensen
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import and prepare dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting dataset into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting LinearRegression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set
y_pred = regressor.predict(X_test)
print("Training set score: {:.2f}".format(regressor.score(X_train, y_train)))
print("Test set score: {:.2f}".format(regressor.score(X_test, y_test)))

# Visualizing training set results
plt.scatter(X_train, y_train, color='r')
plt.plot(X_train, regressor.predict(X_train), color='b')
plt.title("Salary vs. Experience (training set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing test set results
plt.scatter(X_test, y_test, color='r')
plt.plot(X_train, regressor.predict(X_train), color='b')
plt.title("Salary vs. Experience (test set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:49:39 2020

@author: sowrensen
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv')
transactions = []
for i in range(0, len(dataset.index)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, len(dataset.columns))])
    
from apyori import apriori
rules = apriori(transactions, 
                min_support=0.003, # Number of purchases of an item in a week / Total transactions (3*7/7500)
                min_confidence=0.2, # Percentage of correct rules 
                min_lift=3, 
                min_length=2)

# Visualizing the rules
result = list(rules)
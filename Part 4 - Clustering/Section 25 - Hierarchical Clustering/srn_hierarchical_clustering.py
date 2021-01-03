#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:38:06 2020

@author: sowrensen
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using dendogram to determine optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Fitting hierarchical clustering to mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualizing the result
colors = ['red', 'blue', 'green', 'magenta', 'cyan']
labels = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']

for i, color, label in zip(range(0, 5), colors, labels):
    plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s=100,
                c=color, label=label)
plt.title("Cluster of clients")
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()
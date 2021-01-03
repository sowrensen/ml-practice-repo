#!usr/bin/env python3
#%%
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
# Import the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
#%%
# Using the elbow method to find the number of clusters
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
#%%
# Plotting the output
plt.plot(range(1, 11), wcss)
plt.title("The elbow method")
plt.xlabel("No. of clusters")
plt.ylabel("WCSS")
plt.show()
#%%
# Applying k-means to the mall dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
#%%
# Visualizing the clusters
# As done in tutorial
# plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], c='red', label="Cluster 1")
# plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], c='blue', label="Cluster 2")
# plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], c='green', label="Cluster 3")
# plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], c='magenta', label="Cluster 4")
# plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], c='cyan', label="Cluster 5")

# Let's make it short and robust
colors = ['red', 'blue', 'green', 'magenta', 'cyan']
labels = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']

for i, color, label in zip(range(0, 5), colors, labels):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100,
                c=color, label=label)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label="Centroids")
plt.title("Cluster of clients")
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()

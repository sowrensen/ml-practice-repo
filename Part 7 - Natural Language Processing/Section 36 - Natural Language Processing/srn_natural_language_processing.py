#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:49:01 2020

@author: sowrensen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the text
import re
import nltk
nltk.download('stopwords', download_dir="/home/sowrensen/Applications/anaconda3/share/nltk_data")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Same types of text is called corpus
corpus = []

for index in range(0, len(dataset['Review'])):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][index])
    review = review.lower().split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the bag of word model
from sklearn.feature_extraction.text import CountVectorizer
# Take 1500 most frequent words
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# From the confusion matrix, it can be seen that the algorithm made 73% correct prediction
# Best algorithms to use for NLP are Naive Bayes, Random Forest, and Decision Tree Classification



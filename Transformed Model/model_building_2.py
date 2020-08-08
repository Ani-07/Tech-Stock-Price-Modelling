# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 16:21:24 2020

@author: Anirudh Raghavan
"""

# Model Building

##############################################################################

# Input 1 - Cleaned and Transformed Features
# Input 2 - Price change labels

# Output - Model to predict stock prices

##############################################################################

# Code

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


##############################################################################

# Step 1 - We shall ensure that the index of our prices data match with our 
# Features

prices  = pd.read_csv("tech_price_data_2.csv")

trans_features = pd.read_csv("sec_feat_transformed.csv")

prices["Year"] = [str(x) for x in prices["Year"]]

prices["Quart"] = [str(x) for x in prices["Quart"]]

prices["Index"] = prices["Ticker"] + prices["Year"] + prices["Quart"]


match_list = []

for i in range(trans_features.shape[0]):
    if prices["Index"][i] == trans_features["Index"][i]:
        score = "Yes"
    else:
        score = "No"
    match_list.append(score)

all(item == "Yes" for item in match_list)

##############################################################################

# Step 2 - Loading Labels and Features and creating a train/ test split

file = open("1_day_change.txt", 'r')
labels = file.readlines()

features  = pd.read_csv("features_proc.csv")

labels = np.array(labels)

#############################################################################

np.any(np.isnan(features))

np.where(np.isnan(features))








X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                                                    test_size=0.2, random_state=42)

#######################################################################################

# Let us first understand the structure of our training labels

tot = sum(y_train == 0) + sum(y_train == 1)
neg = sum(y_train == 0)/tot
pos = sum(y_train == 1)/tot
        
neg
pos

# MODEL BUILDING

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

# In order to measure the performance of our classifer, we shall use a cross
# validation performance score

cross_val_score(gnb, X_train, y_train, cv=5, scoring="accuracy")

# Build Confusion Matrix

confusion_matrix(y_test, y_test)

###############################################################################

# Logistic Regression

log_reg = LogisticRegression(random_state=0, max_iter = 1000)

log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)


confusion_matrix(y_test, y_pred_log)

cross_val_score(log_reg, X_train, y_train, cv=5, scoring="accuracy")


confusion_matrix(y_test, y_pred_log)


##############################################################################

# Boosting

from sklearn.ensemble import AdaBoostClassifier

ada_boost = AdaBoostClassifier(n_estimators=1000, random_state=0, 
                               algorithm='SAMME', learning_rate = 0.5)

ada_boost.fit(X_train, y_train)  

y_pred_ab = ada_boost.predict(X_test)

confusion_matrix(y_test, y_pred_ab)

ada_boost.n_classes_



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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


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

file = open("4_day_change.txt", 'r')
labels = file.readlines()

features  = pd.read_csv("features_proc.csv")

labels = np.array(labels)

print(np.any(np.isnan(features)))

X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                                                    test_size=0.2, random_state=42)

##############################################################################
# MODEL BUILDING
##############################################################################
## Model 1 - Gaussian Naive Bayes

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

nb_conf = confusion_matrix(y_test, y_pred)

###############################################################################

# Model 2 - Logistic Regression

# Fit Model

log_reg = LogisticRegression(random_state=0, max_iter = 2000)

log_reg.fit(X_train, y_train)

# Measure general accuracy of model with cross validation

cross_val_score(log_reg, X_train, y_train, cv=5, scoring="accuracy")

# Predict the response for test dataset
y_pred_log = log_reg.predict(X_test)


# Build Confusion Matrix
log_conf = confusion_matrix(y_test, y_pred_log)

##############################################################################

# Model 3 - Boosting

# Fit Model

ada_boost = AdaBoostClassifier(n_estimators=1000, random_state=0, 
                               algorithm='SAMME', learning_rate = 0.5)

ada_boost.fit(X_train, y_train)  

# Measure general accuracy of model with cross validation

cross_val_score(ada_boost, X_train, y_train, cv=5, scoring="accuracy")

# Predict the response for test dataset

y_pred_ab = ada_boost.predict(X_test)

# Build Confusion Matrix

ab_conf = confusion_matrix(y_test, y_pred_ab)


#############################################################################

# Comparison of three models

# We shall use the following methods to compare the 3 models:
#  1) Precision Rate of predicting class 1 and-1
#  2) Utility function

#############################################################################

# We use precision rate for only class 1 and -1 because we shall make decisions
# only based on those classes and thus we measure the performance of how many
# times we get the classes right

print(classification_report(y_test, y_pred, digits=3))

print(classification_report(y_test, y_pred_ab, digits=3))

print(classification_report(y_test, y_pred_log, digits=3))

#############################################################################

# Utlity Function

# We first create a utility matrix. The utility matrix shall depend on our 
# trading strategy

# We shall assume a trading stragey with stop loss and limit on gains as well

##############################################################################
# Prediction = Class 1

# If our prediction is Class 1, we shall invest money and in case the price 
# increases, we shall sell once the increase exceeds 2% as our prediction is 
# that price will inrease atleast upto 2%

# However, we shall also sell our share if price drops more than 1.5%

# Correct Prediction = +2
# Wrong PRediction (0 or -1) = -1.5

##############################################################################
# Prediction = Class 0

# In such a case, we would take no action and thus we would be left with
# opportunity loss

# Correct Prediction = 0
# Prediction of +1 or -1 = -2

##############################################################################

# Prediction = Class -1

# This would be similar to the first case

# Correct Prediction = +2
# Prediction of +1 or 0 = -1.5

#############################################################################

util_matrix = np.matrix([[2,-1.5,-1.5],[0,0,0],[-1.5,-1.5,2]])

print(np.sum(np.multiply(util_matrix, nb_conf)))

print(np.sum(np.multiply(util_matrix, log_conf)))

print(np.sum(np.multiply(util_matrix, ab_conf)))


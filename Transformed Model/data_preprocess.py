# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:50:04 2020

@author: Anirudh Raghavan
"""

# Data Preprocessing

#############################################################################

# We will explore our transformed feature list and process for model building

# Input: sec_feat_transformed

# Output: feat_cleaned

#############################################################################

# Code

import numpy as np
import pandas as pd

# Description of data

sec_data = pd.read_csv("sec_feat_transformed.csv")

# First understand the dimensions of the data

sec_data.shape

# We have 4718 rows and 141 columns, let us take a look at the column names

sec_data.columns

# Now we will remove the index column

sec_data = sec_data.drop(['Index'], axis=1)

# Now we shall remove the columns which have more than 30% NA rows

sec_data = sec_data.dropna(thresh = sec_data.shape[0]*0.3, axis=1)

for name in sec_data.columns:
    print(name)
    if sum(sec_data[name] == 0) >= sec_data.shape[0]*0.3:
        print("yes")
        sec_data = sec_data.drop([name], axis=1)

col_type = [str(sec_data[sec_data.columns[i]].dtype) for i in range(sec_data.shape[1]-1)]

all(i == "float64" for i in col_type)

# Thus all columns are numerical

for name in sec_data.columns:
    sec_data[name] = sec_data[name].replace(np.nan,sec_data[name].mean())

# We shall now go ahead with normalization

def normalize_data (x, max_x, min_x):
    temp = (x - min_x)/(max_x-min_x)
    return temp

# We then write a for loop to go through each column and then use apply on the each column to 
# compute normalized values and these are then replaced in the column   

for name in sec_data.columns[8:105]:
    print(name)
    sec_data[name] = sec_data[name].apply(normalize_data, 
                                          args = (max(sec_data[name]), 
                                                  min(sec_data[name])))

# First compute a correlation matrix to observe the amount of correlation 
# between features

corr_mat = sec_data.corr()

# Let us see the number of variables with correlation of more than 0.9

N = corr_mat.shape[0]

high_corr = []

for i in range(N):
    for j in range(i,N):
        if corr_mat.iloc[i,j] >= 0.90 or corr_mat.iloc[i,j] <= -0.90:
            tmp = [i,j]
            high_corr.append(tmp)

added = []
removed = []

# We now have pairs of variables with more than 0.9 correlation.
# We will now keep 1 column from each pair of highly correlated pairs in our dataset 
# and remove the others. We will also ensure that there are no duplications.

for pair in high_corr:
    i, j = pair
    if i == j:
        continue
    if i in added or j in added or i in removed or j in removed:
        continue
    else:
        added.append(i)
        removed.append(j)


sec_data_shrunk = sec_data.drop(sec_data.columns[removed], axis = 1)

# This will now become our set of features to work with for training our models

sec_data_shrunk.to_csv("features_proc.csv", index = False)


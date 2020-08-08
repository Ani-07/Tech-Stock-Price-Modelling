# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:50:36 2020

@author: Anirudh Raghavan
"""

# Creating Target Variable

##############################################################################

# Input: Downloaded price data

# Output: Labels for 3 different periods of price movements

##############################################################################

# We will compute the change in share prices in 1 day, 2 day and 4 days

# These change in share prices, will be further labelled based on the following
# criteria:
    
# 1 - if increase is > 2%
# 0 - if increase is >-2% and < 2%
# -1 - if increase is < -2%

##############################################################################

# Code


import pandas as pd

prices  = pd.read_csv("tech_price_data_2.csv")

#############################################################################

def label_creator(x):
    if x > 0.02:
        label = 1
    elif x < -0.02:
        label = -1
    else:
        label = 0
    
    return label


day_1_change = [label_creator(change) for change in prices["1-Day"]]

day_2_change = [label_creator(change) for change in prices["2-Day"]]

day_4_change = [label_creator(change) for change in prices["4-Day"]]

#############################################################################

with open("1_day_change.txt", 'w') as file:
    for item in day_1_change:
        file.write("%s\n" % item)

with open("2_day_change.txt", 'w') as file:
    for item in day_2_change:
        file.write("%s\n" % item)

with open("4_day_change.txt", 'w') as file:
    for item in day_4_change:
        file.write("%s\n" % item)





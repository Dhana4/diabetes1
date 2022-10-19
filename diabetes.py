# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:35:50 2022

@author: admin
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
df = pd.read_csv('diabetes.csv') 
print(df.shape)
df.describe().transpose()
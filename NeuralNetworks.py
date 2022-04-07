# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 07:55:32 2019

@author: Scelitor9
"""

#Neural Networks


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#LogReg
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


#DiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#Decision Trees 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Neural Networks
import keras
from keras.utils import to_categorical

plt.style.use("seaborn")

music=pd.read_csv("C://Users/Scelitor9/Documents/Maths/MAS369/Assignment2/songstrain.csv")
colnames=music.columns
features=["duration","fade_in","fade_out","loudness","mode","tempo", "time_sig", "year", "artist_fam", "artist_pop"]
X=music[features]
y=music['song_pop']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train_copy=X_train
y_train_copy=y_train
X_test_copy=X_test
y_test_copy=y_test

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

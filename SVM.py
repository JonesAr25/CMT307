# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 00:56:00 2019

@author: Scelitor9
"""

#SVMs

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

from sklearn.model_selection import train_test_split
#DiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

#SVMs
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from mpl_toolkits import mplot3d

music=pd.read_csv("C://Users/Scelitor9/Documents/Maths/MAS369/Assignment2/songstrain.csv")
colnames=music.columns
features=["duration","loudness","mode","tempo", "time_sig", "year", "artist_fam", "artist_pop"]


X=music[features].values
y=music['song_pop'].values


X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.05)

model=SVC(kernel="linear",C=1000000)
svmfit=model.fit(X_train,y_train)

y_preds=svmfit.predict(X_test)==y_test

cm_svm1000=metrics.confusion_matrix(y_preds,y_test)

plt.figure(figsize=(6,6))
sns.heatmap(cm_svm_linear,annot=True,fmt="d",linewidths=.5,square=True,cmap="summer")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
all_sample_title=("SVM: Linear Kernel training_set=2500")
plt.title(all_sample_title,size=15)
#plt.savefig('fig12.png')

svc=SVC(kernel="linear")
parameters={"C":[1,100,1000,100000]}
tunesvc=GridSearchCV(svc,parameters,cv=5)
tunesvc.fit(X_train,y_train)

y_preds_linear=tunesvc.predict(X_test)==y_test

cm_svm_linear=metrics.confusion_matrix(y_preds_linear,y_test)

svc=SVC(kernel='rbf')
parameters={"C":[1,10],"gamma":(0.1,0.5,1,2)}
tunesvc=GridSearchCV(svc,parameters,cv=5)
tunesvc.fit(X_train,y_train)

y_preds_rbf=tunesvc.predict(X_test)==y_test

cm_svm_rbf=metrics.confusion_matrix(y_preds_rbf,y_test)

plt.figure(figsize=(6,6))
sns.heatmap(cm_svm_rbf,annot=True,fmt="d",linewidths=.5,square=True,cmap="summer")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
all_sample_title=("SVM: RBF Kernel training_set=5000")
plt.title(all_sample_title,size=15)
#plt.savefig('fig11.png')



plt.figure(figsize=(6,6))
sns.heatmap(cm,annot=True,fmt="d",linewidths=.5,square=True,cmap="summer")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
all_sample_title=("SVM: Linear Kernel training_set=5000")
plt.title(all_sample_title,size=15)
#plt.savefig('fig13.png')









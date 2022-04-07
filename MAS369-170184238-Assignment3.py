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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

#Decision Trees 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#SVMs
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from mpl_toolkits import mplot3d

#Neural Networks
from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier


plt.style.use("seaborn")

music=pd.read_csv("C://Users/Scelitor9/Documents/Maths/MAS369/Assignment2/songstrain.csv")
colnames=music.columns
features=["duration","fade_in","fade_out","loudness","mode","tempo", "time_sig", "year", "artist_fam", "artist_pop"]


#Logistic Regression

X=music[features]
y=music['song_pop']

enc=LabelEncoder()
label_encoder=enc.fit(y)
y=label_encoder.transform(y)    
logreg=LogisticRegression()

#Basic
logreg.fit(X,y)
y_pred=logreg.predict(X)
cm=metrics.confusion_matrix(y,y_pred)        


#split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
logreg.fit(X_train,y_train)
y_test_pred=logreg.predict(X_test)
cm_test=metrics.confusion_matrix(y_test,y_test_pred)
score=logreg.score(X_test,y_test)

plt.figure(figsize=(6,6))
sns.heatmap(cm_test,annot=True,fmt="d",linewidths=.5,square=True,cmap="Blues_r")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
all_sample_title="Accuracy Score: {0}".format(score)
plt.title(all_sample_title,size=15)
#plt.savefig('fig1.png')

TP,FP,FN,TN=cm[0][0],cm[0][1],cm[1][0],cm[1][1]
accuracy,precision,sensitivity,specificity=(TP+TN)/(TP+TN+FP+FN),TP/(TP+FP),TP/(TP+FN),TN/(TN+FP)

y_test_pred_prob=logreg.predict_proba(X_test)[::,1]
fpr,tpr,_=metrics.roc_curve(y_test,y_test_pred_prob)
auc=metrics.roc_auc_score(y_test,y_test_pred_prob)
plt.plot(fpr,tpr,label='auc='+str(auc))
plt.xlabel('Sensitivity')
plt.ylabel('Specificity')
plt.legend(loc=4)
#plt.savefig('fig2.png')



#Discriminant Analysis

#basic


X=music[features].values
y=music['song_pop'].values


lda=LDA(store_covariance=True)
lda.fit(X,y)
y_pred=lda.fit(X,y).predict(X)
y_predprob=lda.fit(X,y).predict_proba(X)

plt.figure(figsize=(6,6))
sns.heatmap(cm,annot=True,fmt="d",linewidths=.5,square=True,cmap="Reds_r")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title('Linear Discriminant Analysis Confusion Matrix')
 #plt.savefig('fig3.png')
    

#5-fold
kf=KFold(n_splits=5,shuffle=True)
kf.get_n_splits(X)
test_preds=[]
for train_index, test_index in kf.split(X):
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]
    test_preds.append(y_test-lda.fit(X_train,y_train).predict(X_test))
test_preds
lda.fit(X_train,y_train)
y_pred=lda.fit(X_train,y_train).predict(X_test)

#score=cross_val_score(estimator=lda,X=X,y=y,cv=kf)

cm_LDA_kf=metrics.confusion_matrix(y_test,y_pred)

TP,FP,FN,TN=cm_LDA_kf[0][0],cm_LDA_kf[0][1],cm_LDA_kf[1][0],cm_LDA_kf[1][1]
accuracy,precision,sensitivity,specificity=(TP+TN)/(TP+TN+FP+FN),TP/(TP+FP),TP/(TP+FN),TN/(TN+FP)

y_test_pred_prob=lda.predict_proba(X_test)[::,1]
fpr,tpr,_=metrics.roc_curve(y_test,y_test_pred_prob)
auc=metrics.roc_auc_score(y_test,y_test_pred_prob)
plt.plot(fpr,tpr,label='auc='+str(auc))
plt.xlabel('Sensitivity')
plt.ylabel('Specificity')
plt.legend(loc=4)
plt.figure(figsize=(6,6))
sns.heatmap(cm_LDA_kf,annot=True,fmt="d",linewidths=.5,square=True,cmap="winter_r")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title('Linear Discriminant Analysis 5-fold Validation')
#plt.savefig('fig9.png')


#QDA

X=music[features].values
y=music['song_pop'].values

qda=QDA(store_covariance=True)
musicQDA=qda.fit(X,y)
qda.fit(X,y)
yhat=qda.predict(X)
cm_qda=metrics.confusion_matrix(yhat,y)

X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.8)
qda.fit(X_train,y_train)
y_preds=qda.predict(X_test)
cm_qda_crossval=metrics.confusion_matrix(y_test,y_preds)

TP,FP,FN,TN=cm_qda_crossval[0][0],cm_qda_crossval[0][1],cm_qda_crossval[1][0],cm_qda_crossval[1][1]
accuracy,precision,sensitivity,specificity=(TP+TN)/(TP+TN+FP+FN),TP/(TP+FP),TP/(TP+FN),TN/(TN+FP)

y_test_pred_prob=qda.predict_proba(X_test)[::,1]
fpr,tpr,_=metrics.roc_curve(y_test,y_test_pred_prob)
auc=metrics.roc_auc_score(y_test,y_test_pred_prob)
plt.plot(fpr,tpr,label='auc='+str(auc))
plt.xlabel('Sensitivity')
plt.ylabel('Specificity')
plt.legend(loc=4)
plt.figure(figsize=(6,6))
sns.heatmap(cm_qda_crossval,annot=True,fmt="d",linewidths=.5,square=True,cmap="winter_r")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")

plt.figure(figsize=(6,6))
sns.heatmap(cm_qda_crossval,annot=True,fmt="d",linewidths=.5,square=True,cmap="YlGn")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title('QDA cross-validated Confusion Matrix')
plt.savefig('fig10.png')

#Decision Trees
X=music[features]
y=music['song_pop']
y1=y>0
treeclf=DecisionTreeClassifier(max_depth=2)
treeclf.fit(X,y1)
treeclf.feature_importances_ 
#export_graphviz(treeclf,out_file="tree.dot",feature_names=features)

yhat=treeclf.predict(X)
yhat
metrics.accuracy_score(y1,yhat)
cm_decision_tree=metrics.confusion_matrix(y1,yhat)

X_train, X_test, y_train, y_test=train_test_split(X,y1,train_size=0.8)
treeclf.fit(X_train,y_train)
yhat=treeclf.predict(X_test)
cm_decisiontree_crossval=metrics.confusion_matrix(y_test,yhat)

#export_graphviz(treeclf,out_file="tree.dot",feature_names=features)
plt.figure(figsize=(6,6))
sns.heatmap(cm_decisiontree_crossval,annot=True,fmt="d",linewidths=.5,square=True,cmap="copper_r")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title('Decision Tree Confusion Matrix')
plt.savefig('fig7.png')

#Random Forests

rfclf=RandomForestClassifier(n_estimators=50)
rfclf.fit(X,y1)
yhat1=rfclf.predict(X)
metrics.accuracy_score(yhat1,y1)
cm_random_forest=metrics.confusion_matrix(y1,yhat1)

plt.figure(figsize=(6,6))
sns.heatmap(cm_random_forest,annot=True,fmt="d",linewidths=.5,square=True,cmap="Greens_r")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title('Random Forest Confusion Matrix')
#plt.savefig('fig4.png')

X_train, X_test, y_train, y_test=train_test_split(X,y1,train_size=0.8)
rfclf.fit(X_train,y_train)
yhat=rfclf.predict(X_test)
cm_random_forest_crossval=metrics.confusion_matrix(y_test,yhat)

plt.figure(figsize=(6,6))
sns.heatmap(cm_random_forest_crossval,annot=True,fmt="d",linewidths=.5,square=True,cmap="Oranges_r")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title('Random Forest Confusion Matrix - With Cross Validation')
#plt.savefig('fig5.png')

metrics.accuracy_score(y_test,yhat)

TP,FP,FN,TN=cm_random_forest_crossval[0][0],cm_random_forest_crossval[0][1],cm_random_forest_crossval[1][0],cm_random_forest_crossval[1][1]
accuracy,precision,sensitivity,specificity=(TP+TN)/(TP+TN+FP+FN),TP/(TP+FP),TP/(TP+FN),TN/(TN+FP)

y_test_pred_prob=rfclf.predict_proba(X_test)[::,1]
fpr,tpr,_=metrics.roc_curve(y_test,y_test_pred_prob)
auc=metrics.roc_auc_score(y_test,y_test_pred_prob)
plt.plot(fpr,tpr,label='auc='+str(auc))
plt.xlabel('Sensitivity')
plt.ylabel('Specificity')
plt.legend(loc=4)
plt.figure(figsize=(6,6))
sns.heatmap(cm_random_forest_crossval,annot=True,fmt="d",linewidths=.5,square=True,cmap="winter_r")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")

#Gradient Boosting
X_train_gb, X_test_gb, y_train_gb, y_test_gb=train_test_split(X,y1,train_size=0.8)
gbc=GradientBoostingClassifier(n_estimators=100, learning_rate=0.025)
gbc.fit(X_train_gb,y_train_gb)
yhat2=gbc.predict(X_test_gb)
cm_gradient_boost_crossval=metrics.confusion_matrix(y_test_gb,yhat2)

plt.figure(figsize=(6,6))
sns.heatmap(cm_gradient_boost_crossval,annot=True,fmt="d",linewidths=.5,square=True,cmap="Purples_r")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title('Gradient Boost Confusion Matrix - With Cross Validation')
#plt.savefig('fig6.png')

metrics.accuracy_score(y_test_gb,yhat2)

TP,FP,FN,TN=cm_gradient_boost_crossval[0][0],cm_gradient_boost_crossval[0][1],cm_gradient_boost_crossval[1][0],cm_gradient_boost_crossval[1][1]
accuracy,precision,sensitivity,specificity=(TP+TN)/(TP+TN+FP+FN),TP/(TP+FP),TP/(TP+FN),TN/(TN+FP)

y_test_pred_prob=gbc.predict_proba(X_test)[::,1]
fpr,tpr,_=metrics.roc_curve(y_test,y_test_pred_prob)
auc=metrics.roc_auc_score(y_test,y_test_pred_prob)
plt.plot(fpr,tpr,label='auc='+str(auc))
plt.xlabel('Sensitivity')
plt.ylabel('Specificity')
plt.legend(loc=4)

#SVMs
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


#Neural Networks

X_train,X_test,y_train,y_test=train_test_split(X,y,
test_size=0.2)
mlp=MLPClassifier(hidden_layer_sizes=(1024,),batch_size=128,max_iter=100)
mlp.fit(X_train,y_train)

y_preds=mlp.predict(X_test)

cm_nn=metrics.confusion_matrix(y_preds,y_test)
mlp.predict_proba(X_test)

plt.figure(figsize=(6,6))
sns.heatmap(cm_nn,annot=True,fmt="d",linewidths=.5,square=True,cmap="plasma")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.title("Neural Networks Cross Validated")
plt.show()
#plt.savefig('fig15')


TP,FP,FN,TN=cm_nn[0][0],cm_nn[0][1],cm_nn[1][0],cm_nn[1][1]
accuracy,precision,sensitivity,specificity=(TP+TN)/(TP+TN+FP+FN),TP/(TP+FP),TP/(TP+FN),TN/(TN+FP)

y_test_pred_prob=mlp.predict_proba(X_test)[::,1]
fpr,tpr,_=metrics.roc_curve(y_test,y_test_pred_prob)
auc=metrics.roc_auc_score(y_test,y_test_pred_prob)
plt.plot(fpr,tpr,label='auc='+str(auc))
plt.xlabel('Sensitivity')
plt.ylabel('Specificity')
plt.legend(loc=4)
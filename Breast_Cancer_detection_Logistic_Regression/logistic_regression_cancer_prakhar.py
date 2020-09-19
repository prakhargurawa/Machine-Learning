# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 17:28:46 2020

@author: prakh

Dataset link: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

Note: Missing data rows are removed

"""

#Import libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import dataset
dataset=pd.read_csv("breast_cancer.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

#Splitting data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Logistic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#Predict the test set
y_pred=classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

#computing the accuracy k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracy=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print("Accuracy: {:.2f} %".format(accuracy.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracy.std()*100))
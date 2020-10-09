# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:26:22 2020

@author: prakh

Play?	Outlook	Temp	Humidity	Windy
no	sunny	hot	high	FALSE
no	sunny	hot	high	TRUE
yes	overcast	hot	high	FALSE
yes	rainy	mild	high	FALSE
yes	rainy	cool	normal	FALSE
no	rainy	cool	normal	TRUE
yes	overcast	cool	normal	TRUE
no	sunny	mild	high	FALSE
yes	sunny	cool	normal	FALSE
yes	rainy	mild	normal	FALSE
yes	sunny	mild	normal	TRUE
yes	overcast	mild	high	TRUE
yes	overcast	hot	normal	FALSE
no	rainy	mild	high	TRUE
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('weathertxt.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:,0].values
print(X)
print(y)


#Encode categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#index which will be applied with one hot encoder
ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2,3])], remainder='passthrough')
X=np.array(ct.fit_transform(X))
print(X)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

#Create classifier
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""
Turn them to numbers, for example for each unique country assingn a unique number (like 1,2,3 and ...)

also you Don't need to use One-Hot Encoding (aka dummy variables) when working with random forest,
because trees don't work like other algorithm (such as linear/logistic regression) and they don't work by distant 
(they work with finding good split for your features) so NO NEED for One-Hot Encoding
"""
# Importing the dataset
dataset = pd.read_csv('weathertxt.csv')
print(dataset)
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:,0].values
print(X)
print(y)

def codeOutlook(n):
    if n=="sunny":
        return 1
    elif n=="overcast":
        return 2
    else:
        return 3
    
def codeTemp(n):
    if n=="hot":
        return 1
    elif n=="mild":
        return 2
    else:
        return 3
    
def codeHumidity(n):
    if n=="high":
        return 1
    elif n=="normal":
        return 2
    else:
        return 3
    
def codeWindy(n):
    if n=="False":
        return 1
    elif n=="True":
        return 2
    else:
        return 3


for x in X:
    print(x[0],x[1],x[2],x[3])
    x[0]=codeOutlook(x[0])
    x[1]=codeTemp(x[1])
    x[2]=codeHumidity(x[2])
    x[3]=codeWindy(x[3])
    print(x[0],x[1],x[2],x[3])
    
    
print(X)
print(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

#Create classifier
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""
Or you could have done Label Encoder
"""

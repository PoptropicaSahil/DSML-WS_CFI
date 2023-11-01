# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 06:23:10 2021

@author: Sahil Girhepuje
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('titanic.csv')

#Let's check the first 5 rows
data.head()
data.shape
data.isnull().sum()

plt.figure(figsize = (10,10))
sns.boxplot(data['Pclass'], data['Age'])
plt.xlabel("Pclass")
plt.ylabel("Age")
plt.show()

def fill_missing_age(row):
    if row['Age'] != row['Age']:     #Checking for null Value for Age
        if row['Pclass'] == 1:
            row['Age'] = 37
        elif row['Pclass'] == 2:
            row['Age'] = 29
        else:
            row['Age'] = 24
    return row

data = data.apply(fill_missing_age, axis = 1)

#Let us now check if our logic worked
data.isnull().sum()


Pclass1 = data[data['Pclass'] == 1]
print("Length of Pclass1 is : ", len(Pclass1))
Pclass1.isnull().sum()

Pclass2 = data[data['Pclass'] == 2]
print("Length of Pclass2 is : ", len(Pclass2))
Pclass2.isnull().sum()

Pclass3 = data[data['Pclass'] == 3]
print("Length of Pclass3 is : ", len(Pclass3))
Pclass3.isnull().sum()

data['Cabin'].fillna("Not Assigned", inplace = True)
data.isnull().sum()

data['Embarked'].value_counts()
data['Embarked'].replace(np.nan, 'S', inplace = True)
#Final check for missing values
data.isnull().sum()

data.drop(['Name', 'Ticket', 'PassengerId'], axis = 1, inplace = True)

from sklearn.preprocessing import LabelEncoder

def le_transform(col):
    le = LabelEncoder()
    col = le.fit_transform(col)
    return col

cols_to_encode = ['Sex', 'Cabin', 'Embarked']

data[cols_to_encode] = data[cols_to_encode].apply(le_transform)

#Let's now look at a sample of the data to see what has happened
data.head()

train = data.drop(['Survived'], axis = 1)
target = data['Survived']

from sklearn.model_selection import train_test_split
X_Train, X_CV, Y_Train, Y_CV = train_test_split(train, target, test_size = 0.2, random_state = 0)

print(X_Train.shape, X_CV.shape)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(verbose = 1, n_jobs = -1)
logreg.fit(X_Train, Y_Train)


from sklearn.metrics import accuracy_score, confusion_matrix
Y_Pred = logreg.predict(X_CV)
print("Accuracy: ", accuracy_score(Y_Pred, Y_CV)*100, "%")

plt.figure(figsize = (8,8))
sns.heatmap(confusion_matrix(Y_Pred, Y_CV), annot = True, fmt = '.3g')
plt.show()
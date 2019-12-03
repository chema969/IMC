#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EJERCICIO9
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Cargar el dataset
data = pd.read_csv('bases/dataset3.csv',header=None)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X,y)

#Partimos el conjunto en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                stratify=y, 
                                                test_size=0.25,random_state=1)
# Entrenar el modelo SVM
svm_model = svm.SVC(kernel='rbf')

Cs = np.logspace(-5, 15, num=11, base=2)
Gs = np.logspace(-15, 8, num=9, base=2)
optimo = GridSearchCV(estimator=svm_model, param_grid=dict(C=Cs,gamma=Gs),
n_jobs=-1,cv=5)
optimo.fit(X_train,y_train)
print(optimo.score(X_test,y_test))

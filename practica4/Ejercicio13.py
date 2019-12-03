#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EJERCICIO13
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Cargar el dataset
data = pd.read_csv('bases/train_nomnist.csv',header=None)
X_train = data.iloc[:,:-1].values
y_train = data.iloc[:,-1].values
data = pd.read_csv('bases/test_nomnist.csv',header=None)
X_test = data.iloc[:,:-1].values
y_test = data.iloc[:,-1].values
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
svm_model = svm.SVC(kernel='rbf')

# Entrenar el modelo SVM
for i in [3,5,10]:
  start_time = time.time()
  Cs = np.logspace(-5, 15, num=11, base=2)
  Gs = np.logspace(-15, 8, num=9, base=2)
  optimo = GridSearchCV(estimator=svm_model, param_grid=dict(C=Cs,gamma=Gs),
  n_jobs=-1,cv=i)
  optimo.fit(X_train,y_train)
  print(optimo.score(X_test,y_test))
  print (time.time() - start_time, "segundos")
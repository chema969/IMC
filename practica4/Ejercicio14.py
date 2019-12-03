#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EJERCICIO14
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Cargar el dataset
data = pd.read_csv('bases/train_spam.csv',header=None)
X_train = data.iloc[:,:-1].values
y_train = data.iloc[:,-1].values
data = pd.read_csv('bases/test_spam.csv',header=None)
X_test = data.iloc[:,:-1].values
y_test = data.iloc[:,-1].values

# Entrenar el modelo SVM
for x in [10e-2,10e-1,1,10]:
  svm_model = svm.SVC(kernel='linear',C=x)
  svm_model.fit(X_train,y_train)
  print(svm_model.score(X_test,y_test))
  
# -*- coding: utf-8 -*-
"""
EJERCICIO17
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
# Cargar el dataset
data = pd.read_csv('bases/train_spam.csv',header=None)
X_train = data.iloc[:,:-1].values
y_train = data.iloc[:,-1].values
data = pd.read_csv('bases/test_spam.csv',header=None)
X_test = data.iloc[:,:-1].values
y_test = data.iloc[:,-1].values

# Entrenar el modelo SVM
svm_model = svm.SVC(kernel='rbf',C=10e-0,gamma='scale')
svm_model.fit(X_train,y_train)
predicted_test=svm_model.predict(X_test)
print(confusion_matrix(y_test,predicted_test))
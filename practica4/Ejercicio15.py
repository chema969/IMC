# -*- coding: utf-8 -*-
"""
EJERCICIO15
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
data = pd.read_csv('bases/vocab.csv',header=None)
vocabulario=np.array(data.values)

for j in range(0,len(vocabulario)):
  vocabulario[j][0]=0
# Entrenar el modelo SVM
svm_model = svm.SVC(kernel='linear',C=10e-2)
svm_model.fit(X_train,y_train)
predicted_test=svm_model.predict(X_test)
print(confusion_matrix(y_test,predicted_test))

for i in range(0,len(predicted_test)):
  if predicted_test[i]!=y_test[i] and y_test[i]==0:
    print("Patron:",i,":")
    for j in range(0,len(X_test[i])):
      if(X_test[i][j]==1):
        print(vocabulario[j][1])
        vocabulario[j][0]=vocabulario[j][0]+1

print("PALABRAS MÁS EQUIVOCADAS")
for j in range(0,len(vocabulario)):
  if(vocabulario[j][0]>3):
    print(vocabulario[j])
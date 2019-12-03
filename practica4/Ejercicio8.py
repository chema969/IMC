#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EJERCICIO8
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# Cargar el dataset
data = pd.read_csv('bases/dataset3.csv',header=None)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X,y)

#Partimos el conjunto en train y test
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                stratify=y, 
                                                test_size=0.25,random_state=1)
# Entrenar el modelo SVM

svm_model = svm.SVC(kernel='rbf',C=2,gamma=200)
svm_model.fit(x_train, y_train)
print(svm_model.score(x_test,y_test))

# Representar los puntos
plt.figure(1)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

# Representar el hiperplano separador
plt.axis('tight')
# Extraer límites
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()

# Crear un grid con todos los puntos y obtener el valor Z devuelto por la SVM
XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
Z = svm_model.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Hacer un plot a color con los resultados
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

plt.show()
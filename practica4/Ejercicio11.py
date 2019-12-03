#EJERCICIO 11

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import math 
# Cargar el dataset
data = pd.read_csv('bases/dataset3.csv',header=None)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X,y)
X=np.array(X)
y=np.array(y)

#Conseguimos los distintos C y gamma
Cs = np.logspace(-5, 15, num=11, base=2)
Gs = np.logspace(-15, 8, num=9, base=2)

#Hacemos k particiones, siendo k=5
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=1)
"""Inicializamos el vector donde vamos a guardar el CCR de los distindos modelos
Las dos primeras columnas de cada fila serán el C y el gamma de esa fila"""
pruebas=np.zeros((len(Cs)*len(Gs),7))
i=2
for train_index, test_index in sss.split(X, y):
  #Separamos en conjunto de entrenamiento y de test
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  j=0
  for c in Cs:
    for gam in Gs:
      #Para cada C y cada gamma entrenamos
      aux = svm.SVC(kernel='rbf',C=c,gamma=gam).fit(X_train,y_train)
      if i==2:
        #Llenamos las dos primeras columnas con c y gamma
        pruebas[j,0]=c
        pruebas[j,1]=gam
      #guardamos el CCR
      pruebas[j,i]=aux.score(X_test,y_test)
      j=j+1
  i=i+1
maximoj=0
maxmedia=np.mean(pruebas[0,2:7])
for x in range(1,(len(Cs)*len(Gs))):
  #Vamos buscando la media mayor entre los distintos conjuntos
  med=np.mean(pruebas[x,2:7])
  if med>maxmedia:
    maxmedia=med
    maximoj=x
#Entrenamos con el optimo
optimo=svm.SVC(kernel='rbf',C=pruebas[maximoj,0],gamma=pruebas[maximoj,1])

#Partimos el conjunto en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                stratify=y, 
                                                test_size=0.25,random_state=1)
optimo.fit(X_train,y_train)
print(optimo.score(X_test,y_test))

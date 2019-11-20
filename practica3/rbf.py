#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:37:04 2016

@author: pagutierrez
"""
import pickle
import sklearn
import sklearn.cluster
import os
import pandas as pd
import numpy as np
import click
import sklearn.metrics
import sklearn.preprocessing
import warnings
import scipy.spatial.distance
import sklearn.linear_model
import sklearn.metrics
from sklearn.model_selection import train_test_split
import time
import math

warnings.filterwarnings('ignore')

@click.command()
@click.option('--train_file', '-t', default=None, required=False,
              help=u'Fichero con los datos de entrenamiento.')
@click.option('--test_file', '-T', default=None, required=False,
              help=u'Fichero con los datos de test.')
@click.option('--classification', '-c', is_flag=True, default=False, show_default=True,
              help=u'Indica si el problema es de clasificación.')
@click.option('--l2', '-l', is_flag=True, default=False, show_default=True,
              help=u'Indica si utilizaremos regularizacion de L2.')
@click.option('--eta', '-e', default=0.01, required=False,
              help=u'Indica el valor del parametro eta.')
@click.option('--outputs', '-o', default=1, required=False,
              help=u'Indica el numero de columnas de salida que tiene el conjunto.')
@click.option('--ratio_rbf', '-r', default=0.1, required=False,
              help=u'Indica la razón (en tanto por 1) de neuronas RBF con respecto al total de patrones en entrenamiento.')
@click.option('--model_file', '-m', default="", show_default=True,
              help=u'Fichero en el que se guardará o desde el que se cargará el modelo (si existe el flag p).') # KAGGLE
@click.option('--pred', '-p', is_flag=True, default=False, show_default=True,
              help=u'Activar el modo de predicción.') # KAGGLE

def entrenar_rbf_total(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model_file, pred):
    """ Modelo de aprendizaje supervisado mediante red neuronal de tipo RBF.
        Ejecución de 5 semillas.
    """
    if not pred:
        t0 = time.clock()
        if train_file is None:
            print("No se ha especificado el conjunto de entrenamiento (-t)")
            return

        train_mses = np.empty(5)
        train_ccrs = np.empty(5)
        test_mses = np.empty(5)
        test_ccrs = np.empty(5)
        train_inputs, train_outputs, test_inputs, test_outputs = lectura_datos(train_file, 
                                                                           test_file,
                                                                           outputs) 
        for s in range(1,6,1):
            print("-----------")
            print("Semilla: %d" % s)
            print("-----------")
            np.random.seed(s)
            train_mses[s-1], test_mses[s-1], train_ccrs[s-1], test_ccrs[s-1] = \
                entrenar_rbf( train_inputs, train_outputs, test_inputs, test_outputs, classification, ratio_rbf, l2, eta, outputs, \
                             model_file and "{}/{}.pickle".format(model_file, s) or "")
            print("MSE de entrenamiento: %f" % train_mses[s-1])
            print("MSE de test: %f" % test_mses[s-1])
            print("CCR de entrenamiento: %.2f%%" % train_ccrs[s-1])
            print("CCR de test: %.2f%%" % test_ccrs[s-1])

        print("*********************")
        print("Resumen de resultados")
        print("*********************")
        print("MSE de entrenamiento: %f +- %f" % (np.mean(train_mses), np.std(train_mses)))
        print("MSE de test: %f +- %f" % (np.mean(test_mses), np.std(test_mses)))
        print("CCR de entrenamiento: %.2f%% +- %.2f%%" % (np.mean(train_ccrs), np.std(train_ccrs)))
        print("CCR de test: %.2f%% +- %.2f%%" % (np.mean(test_ccrs), np.std(test_ccrs)))
        print (time.clock(), "segundos")
        #print(eta,ratio_rbf,l2,np.mean(train_mses), np.std(train_mses),np.mean(train_ccrs), np.std(train_ccrs),np.mean(test_mses), np.std(test_mses),np.mean(test_ccrs), np.std(test_ccrs))
    else:
        # KAGGLE
        if model_file is None:
            print("No se ha indicado un fichero que contenga el modelo (-m).")
            return

        # Obtener predicciones para el conjunto de test
        predictions = predict(test_file, model_file)

        # Imprimir las predicciones en formato csv
        print("Id,Category")
        for prediction, index in zip(predictions, range(len(predictions))):
            s = ""
            s += str(index)

            if isinstance(prediction, np.ndarray):
                for output in prediction:
                    s += ",{}".format(output)
            else:
                s += ",{}".format(int(prediction))

            print(s)


def entrenar_rbf(train_inputs, train_outputs, test_inputs, test_outputs, classification, ratio_rbf, l2, eta, outputs, model_file=""):
    """ Modelo de aprendizaje supervisado mediante red neuronal de tipo RBF.
        Una única ejecución.
        Recibe los siguientes parámetros:
            - train_file: nombre del fichero de entrenamiento.
            - test_file: nombre del fichero de test.
            - classification: True si el problema es de clasificacion.
            - ratio_rbf: Ratio (en tanto por uno) de neuronas RBF con 
              respecto al total de patrones.
            - l2: True si queremos utilizar L2 para la Regresión Logística. 
              False si queremos usar L1 (para regresión logística).
            - eta: valor del parámetro de regularización para la Regresión 
              Logística.
            - outputs: número de variables que se tomarán como salidas 
              (todas al final de la matriz).
        Devuelve:
            - train_mse: Error de tipo Mean Squared Error en entrenamiento. 
              En el caso de clasificación, calcularemos el MSE de las 
              probabilidades predichas frente a las objetivo.
            - test_mse: Error de tipo Mean Squared Error en test. 
              En el caso de clasificación, calcularemos el MSE de las 
              probabilidades predichas frente a las objetivo.
            - train_ccr: Error de clasificación en entrenamiento. 
              En el caso de regresión, devolvemos un cero.
            - test_ccr: Error de clasificación en test. 
              En el caso de regresión, devolvemos un cero.
    """


    num_rbf=round(ratio_rbf*len(train_inputs))
    print("Número de RBFs utilizadas: %d" %(num_rbf))
    kmedias, distancias, centros = clustering(classification, train_inputs, 
                                              train_outputs, num_rbf)
    
    radios = calcular_radios(centros, num_rbf)
    
    matriz_r = calcular_matriz_r(distancias, radios)

    if not classification:
        coeficientes = invertir_matriz_regresion(matriz_r, train_outputs)
        train_predictions = np.matmul(matriz_r, coeficientes)
        train_mse=sklearn.metrics.mean_squared_error(train_predictions,train_outputs)
        #train_ccr=100*sklearn.metrics.accuracy_score(train_outputs,np.around(train_predictions))
        train_ccr=0
    else:
        logreg = logreg_clasificacion(matriz_r, train_outputs, eta, l2)
        predicciones=logreg.predict_proba(matriz_r)   
        salidas_train= sklearn.preprocessing.OneHotEncoder(categories='auto').fit_transform(train_outputs).toarray()
        train_mse=sklearn.metrics.mean_squared_error(predicciones,salidas_train)    
        train_ccr=100*logreg.score(matriz_r, train_outputs)

    """
     Calcular las distancias de los centroides a los patrones de test
          y la matriz R de test
    """
    distancias_test=kmedias.transform(test_inputs)
    matriz_r_test=calcular_matriz_r(distancias_test,radios)
    # # # # KAGGLE # # # #
    if model_file != "":
        save_obj = {
            'classification': classification,
            'radios': radios,
            'kmedias': kmedias
        }
        if not classification:
            save_obj['coeficientes'] = coeficientes
        else:
            save_obj['logreg'] = logreg

        dir = os.path.dirname(model_file)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        with open(model_file, 'wb') as f:
            pickle.dump(save_obj, f)

    # # # # # # # # # # #

    if not classification:
        """
         Obtener las predicciones de entrenamiento y de test y calcular
              el MSE
        """
        test_predictions = np.matmul(matriz_r_test, coeficientes)
        test_mse=sklearn.metrics.mean_squared_error(test_predictions,test_outputs)
        #test_ccr=100*sklearn.metrics.accuracy_score(test_outputs,np.around(test_predictions))        
        test_ccr=0
    else:
        """
         Obtener las predicciones de entrenamiento y de test y calcular
              el CCR. Calcular también el MSE, comparando las probabilidades 
              obtenidas y las probabilidades objetivo
        """
        test_predictions = logreg.predict(matriz_r_test)
        predicciones=logreg.predict_proba(matriz_r_test)
        salidas_test= sklearn.preprocessing.OneHotEncoder(categories='auto').fit_transform(test_outputs).toarray()

        test_mse=sklearn.metrics.mean_squared_error(predicciones,salidas_test)
        test_ccr=100*logreg.score(matriz_r_test,test_outputs)
        print("Matríz de confusión del Test")
        print(sklearn.metrics.confusion_matrix(test_outputs, test_predictions))
        """for i in range(0,len(test_outputs)):
           if test_predictions[i]!=test_outputs[i]:
              print(i,test_predictions[i],test_outputs[i])"""
    return train_mse, test_mse, train_ccr, test_ccr

    
def lectura_datos(fichero_train, fichero_test, outputs):
    """ Realiza la lectura de datos.
        Recibe los siguientes parámetros:
            - fichero_train: nombre del fichero de entrenamiento.
            - fichero_test: nombre del fichero de test.
            - outputs: número de variables que se tomarán como salidas 
              (todas al final de la matriz).
        Devuelve:
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - test_inputs: matriz con las variables de entrada de 
              test.
            - test_outputs: matriz con las variables de salida de 
              test.
    """

    train=pd.read_csv(fichero_train,header=None)
    train=np.array(train)
    train = train.astype(np.float64)
    train_inputs=train[:,:-outputs]
    train_outputs=train[:,-outputs:]
    
    scaler = sklearn.preprocessing.StandardScaler()
    train_inputs = scaler.fit_transform(train_inputs)
    test=pd.read_csv(fichero_test,header=None)
    test=np.array(test)
    test = test.astype(np.float64)
    test_inputs=test[:,:-outputs]
    test_outputs=test[:,-outputs:]
    test_inputs = scaler.transform(test_inputs)
    return train_inputs, train_outputs, test_inputs, test_outputs

def inicializar_centroides_clas(train_inputs, train_outputs, num_rbf):
    """ Inicializa los centroides para el caso de clasificación.
        Debe elegir los patrones de forma estratificada, manteniendo
        la proporción de patrones por clase.
        Recibe los siguientes parámetros:
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - centroides: matriz con todos los centroides iniciales
                          (num_rbf x num_entradas).
    """

    x, centroides, y_train, y_test = train_test_split(train_inputs, train_outputs,
                                                    stratify=train_outputs, 
                                                    test_size=num_rbf/len(train_inputs))
    return centroides

def clustering(clasificacion, train_inputs, train_outputs, num_rbf):
    """ Realiza el proceso de clustering. En el caso de la clasificación, se
        deben escoger los centroides usando inicializar_centroides_clas()
        En el caso de la regresión, se escogen aleatoriamente.
        Recibe los siguientes parámetros:
            - clasificacion: True si el problema es de clasificacion.
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - kmedias: objeto de tipo sklearn.cluster.KMeans ya entrenado.
            - distancias: matriz (num_patrones x num_rbf) con la distancia 
              desde cada patrón hasta cada rbf.
            - centros: matriz (num_rbf x num_entradas) con los centroides 
              obtenidos tras el proceso de clustering.
    """
    
    if(clasificacion):
        centroides=inicializar_centroides_clas(train_inputs,train_outputs,num_rbf)
        kmedias=sklearn.cluster.KMeans(len(centroides),centroides,1,500).fit(train_inputs,train_outputs)
        #kmedias=sklearn.cluster.KMeans(num_rbf, init='k-means++', n_init=1, max_iter=500).fit(train_inputs,train_outputs)              
    else:
        #kmedias=sklearn.cluster.KMeans(num_rbf, init='k-means++', n_init=1, max_iter=500).fit(train_inputs,train_outputs)              
        kmedias=sklearn.cluster.KMeans(num_rbf, init='random', n_init=1, max_iter=500).fit(train_inputs,train_outputs)              
    centros=kmedias.cluster_centers_    

    distancias=kmedias.transform(train_inputs)
    return kmedias, distancias, centros

def calcular_radios(centros, num_rbf):
    """ Calcula el valor de los radios tras el clustering.
        Recibe los siguientes parámetros:
            - centros: conjunto de centroides.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - radios: vector (num_rbf) con el radio de cada RBF.
    """

    dist=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(centros))
    radios=np.array([],dtype=np.float64)
    for x in range(0,num_rbf):
        sumdist=0
        sumdist=sum(dist[x])    
        sumdist=sumdist/(2*(num_rbf-1))
        radios=np.append(radios,sumdist)

    return radios

def calcular_matriz_r(distancias, radios):
    """ Devuelve el valor de activación de cada neurona para cada patrón 
        (matriz R en la presentación)
        Recibe los siguientes parámetros:
            - distancias: matriz (num_patrones x num_rbf) con la distancia 
              desde cada patrón hasta cada rbf.
            - radios: array (num_rbf) con el radio de cada RBF.
        Devuelve:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
    """

    #Creamos una matriz vacia
    matriz_r=np.empty([len(distancias),len(radios)+1])
    matriz_r.astype(np.float64)
    for i in range(0,len(distancias)): 
            for j in range(0,len(radios)):
                aux=math.exp((distancias[i,j]*distancias[i,j])/(-2*radios[j]*radios[j]))
                matriz_r[i][j]=aux
             
    matriz_r[:,-1]=1
        
    return matriz_r

def invertir_matriz_regresion(matriz_r, train_outputs):
    """ Devuelve el vector de coeficientes obtenidos para el caso de la 
        regresión (matriz beta en las diapositivas)
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
        Devuelve:
            - coeficientes: vector (num_rbf+1) con el valor del sesgo y del 
              coeficiente de salida para cada rbf.
    """
    if len(matriz_r)==len(matriz_r[0]):
        coeficientes=np.matmul(np.linalg.inv(matriz_r),train_outputs)
    else:
        coeficientes=np.matmul(np.linalg.pinv(matriz_r),train_outputs)
    return coeficientes

def logreg_clasificacion(matriz_r, train_outputs, eta, l2):
    """ Devuelve el objeto de tipo regresión logística obtenido a partir de la
        matriz R.
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - eta: valor del parámetro de regularización para la Regresión 
              Logística.
            - l2: True si queremos utilizar L2 para la Regresión Logística. 
              False si queremos usar L1.
        Devuelve:
            - logreg: objeto de tipo sklearn.linear_model.LogisticRegression ya
              entrenado.
    """

    logreg=0
    if l2:
        logreg=sklearn.linear_model.LogisticRegression(penalty='l2',C=1/eta,solver='liblinear',multi_class='auto',max_iter=600)
    else:
        logreg=sklearn.linear_model.LogisticRegression(penalty='l1',C=1/eta,solver='liblinear',multi_class='auto',max_iter=600)
    logreg.fit(matriz_r,train_outputs.ravel())
    return logreg


def predict(test_file, model_file):
    """ Calcula las predicciones para un conjunto de test que recibe como parámetro. Para ello, utiliza un fichero que
    contiene un modelo guardado.
    :param test_file: fichero csv (separado por comas) que contiene los datos de test.
    :param model_file: fichero de pickle que contiene el modelo guardado.
    :return: las predicciones para la variable de salida del conjunto de datos proporcionado.
    """
    test_df = pd.read_csv(test_file, header=None)
    test_inputs = test_df.values[:, :]

    with open(model_file, 'rb') as f:
        saved_data = pickle.load(f)

    radios = saved_data['radios']
    classification = saved_data['classification']
    kmedias = saved_data['kmedias']

    test_distancias = kmedias.transform(test_inputs)
    test_r = calcular_matriz_r(test_distancias, radios)

    if classification:
        logreg = saved_data['logreg']
        test_predictions = logreg.predict(test_r)
    else:
        coeficientes = saved_data['coeficientes']
        test_predictions = np.dot(test_r, coeficientes)

    return test_predictions


if __name__ == "__main__":
    entrenar_rbf_total()

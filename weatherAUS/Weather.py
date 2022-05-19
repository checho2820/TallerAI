# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 04:32:56 2022

@author: Sergio Luis Gomez Lara
"""
import math, time, random, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import simplefilter
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



url = 'weatherAUS.csv'
data = pd.read_csv(url)

#Tratamiento de la data
    #Se eliminan las siguientes columnas, se consideran irrelevantes
data.drop(columns = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis = 1, inplace=True)    
data.drop(['Location','RISK_MM'],axis=1, inplace=True)

    #Los datos de fechas se han cambiado por el numero del mes en el que esta la tupla
data['Date'] = pd.to_datetime(data['Date']).dt.month

    #El nombre se ha cambiado de 'Date' a 'Month'
data.rename(columns={'Date':'Month'}, inplace=True)

    #Los valores de 'Yes' y 'No' será reemplazados por los valores numericos '1' y '2'
data.RainTomorrow.replace(['Yes', 'No'], [1, 0], inplace=True)
data.RainToday.replace(['Yes', 'No'], [1, 0], inplace=True)

    #Los nulos será eliminados al igual que los valores llamados 'Nan'
data.dropna(axis=0,how='any', inplace=True)

    #Los datos de las siguientes tablas seran reemplazados por valores numéricos
data.WindDir9am.replace(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', 
                         'SSW', 'N', 'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'], 
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                        inplace=True)

data.WindDir3pm.replace(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 
                         'NNW', 'SSW', 'SW', 'SE', 'N', 'S', 'NNE', 'NE'], 
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                        inplace=True)

data.WindGustDir.replace(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 
                         'ENE', 'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'], 
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                        inplace=True)



data_train = data[:564]
data_test = data[564:]

x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow)




kfold = KFold(n_splits=10) #Kfold se cambia a 10 splits

acc_scores_train_train = []
acc_scores_test_train = []
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

for train, test in kfold.split(x, y):
    logreg.fit(x[train], y[train])
    scores_train_train = logreg.score(x[train], y[train])
    scores_test_train = logreg.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = logreg.predict(x_test_out)

print(f'y de prediccion: {y_pred}')
print(f'Y real :{y_test_out}')

print('Regresión Logística Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión Regresion Loigistica: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución Regresion Loigistica")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_1 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_1}')


print('*'*50)



svc = SVC(kernel='rbf') #Se utiliza el núcleo rbf/gaussiano para adaptarse al modelo.

for train, test in kfold.split(x, y):
    svc.fit(x[train], y[train])
    scores_train_train = svc.score(x[train], y[train])
    scores_test_train = svc.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)

print(f'y de prediccion: {y_pred}')
print(f'Y real :{y_test_out}')

print('Maquina de soporte vectorial Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión SVC: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución svc")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_2 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_2}')


print('*'*50)




arbol = DecisionTreeClassifier(max_depth=2, random_state=42)# Se usa un árbol de profundidad 2 para que no haya overfitting

for train, test in kfold.split(x, y):
    arbol.fit(x[train], y[train])
    scores_train_train = arbol.score(x[train], y[train])
    scores_test_train = arbol.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = arbol.predict(x_test_out)

print(f'y de prediccion: {y_pred}')
print(f'Y real :{y_test_out}')

print('Arbol de Desicion Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión arbol de decision: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución arbol de decision")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_3 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_3}')


print('*'*50)

#Random Forest Classifier



clf = RandomForestClassifier(max_depth=2, random_state=0)

for train, test in kfold.split(x, y):
    clf.fit(x[train], y[train])
    scores_train_train = clf.score(x[train], y[train])
    scores_test_train = clf.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = clf.predict(x_test_out)

print(f'y de prediccion: {y_pred}')
print(f'Y real :{y_test_out}')

print('Arbol de Desicion Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {clf.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión Random Forest: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confusión Random Forest")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_4 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_4}')


print('*'*50)



#Vecino mas cercano

kn = KNeighborsClassifier()

for train, test in kfold.split(x, y):
    kn.fit(x[train], y[train])
    scores_train_train = kn.score(x[train], y[train])
    scores_test_train = kn.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = kn.predict(x_test_out)

print(f'y de prediccion: {y_pred}')
print(f'Y real :{y_test_out}')


print('Vecino mas cercano Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {kn.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión Vecino mas cercano: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución Vecino mas cercano")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score_5 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score_5}')
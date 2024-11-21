# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Directorios
root='C:\\Users\\camed\\OneDrive\\Documentos\\Git\\wine'
carpeta='imagenes'
path_imagenes=os.path.join(root, carpeta)
os.makedirs(path_imagenes, exist_ok=True)

# Funciones y clases
from core.utils import separa_num_cat, escala_num, balance, divide_train_test, ohe
from core.estadistica import Correlacion
from core.modelos.modelo_rf import BosqueAleatorioClasificador
from core.modelos.modelo_svm import SupportVectorMachineClasificador
from core.modelos.modelo_GaussianNB import NaiveBayes
from core.modelos.modelo_LogReg import RegresionLogistica
from core.modelos.modelo_knn import KVecinosCercanos
from core.evaluacion import Evaluacion
from core.modelos.error_rf import BosqueAleatorioError

# Data
path_data=os.path.join(root,'datos\\winequality-red.csv')
#print(path_data)

df = pd.read_csv(path_data)
print(df.head(3))
print(df.shape)


# Types
print(df.dtypes)
df=df.astype({  'fixed acidity': float,
                'volatile acidity': float,
                'citric acid': float,
                'residual sugar': float,
                'chlorides': float,
                'free sulfur dioxide': float,
                'total sulfur dioxide': float,
                'density': float,
                'pH': float,
                'sulphates': float,
                'alcohol': float,
                'quality': int})


# Calidad
print("\nNull:")
print(df.isna().sum())

print("\nDuplicados:")
print(df.duplicated().sum())


print("\nNuméricas:")
print(df.describe().T)

print("\nCategóricas:")
#print(df.describe(include=['object']).T)

# Features
df=df.drop(['Unnamed: 0'], axis=1)

print("\nTipos finales:")
print(df.dtypes)

print("\nData:")
print(df)

# Balance
print("\nBalance de clases:")
print(balance(df['quality']))

# Reagrupar
df['quality_class'] = df['quality'].apply(
    lambda x: 'Medium' if x in [3, 4, 5] else 'High'
)

print(balance(df['quality_class']))

# OHE
df=ohe(df)
print(df.columns)

# Correlación
print("\nMatriz de correlación:")
correlacion=Correlacion(df)
matriz=correlacion.matriz_correlacion()
print(matriz)

print("\nGráfica de correlación:")
fig=correlacion.grafica_correlacion()
fig.savefig(os.path.join(path_imagenes,'1_correlacion.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# Características y objetivo
X = df.drop('quality_class_Medium', axis=1)
y = df['quality_class_Medium']

# Escalado
X_scale=escala_num(X)
print(X_scale)

# Error preliminar
error=BosqueAleatorioError(X, y)
fig=error.calcular_error()
fig.savefig(os.path.join(path_imagenes,'2_error.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# Sets train test
X_train, X_test, y_train, y_test=divide_train_test(X_scale, y, 0.2)

print("\nX_train:")
print(X_train)

print("\ny_train:")
print(y_train)

# Modelamiento Random Forest
'''
modelo_rf=SupportVectorMachineClasificador(X_train, X_test, y_train, y_test)

n_estimators=[50, 100, 150, 200]
max_depth=[None, 10, 20, 30]
min_samples_split=[2, 5, 10, 15]
scoring='recall'
cv=5

modelo_rf.definir_modelo(n_estimators, max_depth, min_samples_split, scoring, cv)
modelo_rf.entrenar_modelo()
y_pred=modelo_rf.predecir(X_test)
'''
# Modelamiento SVM
'''
modelo_svm=SupportVectorMachineClasificador(X_train, X_test, y_train, y_test)

C=[0.1, 1, 10],  # Parámetro de regularización
kernel=['linear', 'rbf', 'poly'],  # Tipos de núcleo
gamma=['scale', 'auto', 0.1, 1],  # Parámetro de kernel
degree=[2, 3]  # Solo relevante para kernel 'poly'
scoring='recall'
cv=5

modelo_svm.definir_modelo(C, kernel, gamma, degree, scoring, cv)
modelo_svm.entrenar_modelo()
y_pred=modelo_svm.predecir(X_test)
'''
# Modelamiento Naive Bayes
'''
modelo_nb=NaiveBayes(X_train, X_test, y_train, y_test)

scoring='recall'
cv=5

modelo_nb.definir_modelo(scoring, cv)
modelo_nb.entrenar_modelo()
y_pred=modelo_nb.predecir(X_test)
'''
# Modelamiento Regresión Logística
'''
modelo_rl=RegresionLogistica(X_train, X_test, y_train, y_test)

C=[0.001, 0.01, 0.1, 1, 10, 100],          # Regularización
solver=['liblinear', 'lbfgs'],             # Métodos de optimización
penalty=['l2', 'none'],                    # Tipo de regularización
scoring='recall'
cv=5

modelo_rl.definir_modelo(C, solver, penalty, scoring, cv)
modelo_rl.entrenar_modelo()
y_pred=modelo_rl.predecir(X_test)
'''
# Modelamiento KNN
modelo_knn=KVecinosCercanos(X_train, X_test, y_train, y_test)

n_neighbors=[3, 5, 7, 9, 11],                   # Número de vecinos
weights=['uniform', 'distance'],                # Peso de los vecinos
metric=['euclidean', 'manhattan', 'minkowski']  # Métrica de distancia
scoring='recall'
cv=5

modelo_knn.definir_modelo(n_neighbors, weights, metric, scoring, cv)
modelo_knn.entrenar_modelo()
y_pred=modelo_knn.predecir(X_test)

# Evaluación
evaluacion=Evaluacion(y_test, y_pred)
evaluacion.reporte()
evaluacion.matriz_confusion()
plt.savefig(os.path.join(path_imagenes,'3_confusion.png'), dpi=300, bbox_inches='tight')
plt.close()

fig=evaluacion.curva_roc()
fig.savefig(os.path.join(path_imagenes,'4_roc.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

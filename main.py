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
from core.modelos.modelo_LogRegReg import RegresionLogisticaRegularizada
from core.modelos.modelo_knn import KVecinosCercanos
from core.modelos.modelo_arbol import ArbolClasificador
from core.modelos.modelo_gbm import AumentoGradienteClasificador
from core.modelos.modelo_xgbm import ExtremoAumentoGradienteClasificador
from core.modelos.modelo_lgbm import LivianoAumentoGradienteClasificador
from core.modelos.modelo_cb import CategoricalBoostingClasificador
from core.modelos.modelo_et import ArbolesExtraClasificador
from core.modelos.modelo_ann import AnnClasificador
from core.modelos.modelo_ensamble import EnsableModelos
from core.modelos.modelo_adab import AdaBoostClasificador
from core.modelos.modelo_histgbm import HistGradienteClasificador

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

# Reserva nombres de columnas
columnas_X = X.columns.tolist()  # Si usas pandas

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

C=[0.1, 1, 10]  # Parámetro de regularización
kernel=['linear', 'rbf', 'poly']  # Tipos de núcleo
gamma=['scale', 'auto', 0.1, 1]  # Parámetro de kernel
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

C=[0.001, 0.01, 0.1, 1, 10, 100]          # Regularización
solver=['liblinear', 'lbfgs']             # Métodos de optimización
penalty=['l2', 'none']                    # Tipo de regularización
scoring='recall'
cv=5

modelo_rl.definir_modelo(C, solver, penalty, scoring, cv)
modelo_rl.entrenar_modelo()
y_pred=modelo_rl.predecir(X_test)
'''
# Modelamiento KNN
'''
modelo_knn=KVecinosCercanos(X_train, X_test, y_train, y_test)

n_neighbors=[3, 5, 7, 9, 11]                    # Número de vecinos
weights=['uniform', 'distance']                 # Peso de los vecinos
metric=['euclidean', 'manhattan', 'minkowski']  # Métrica de distancia
scoring='recall'
cv=5

modelo_knn.definir_modelo(n_neighbors, weights, metric, scoring, cv)
modelo_knn.entrenar_modelo()
y_pred=modelo_knn.predecir(X_test)
'''
# Modelamiento Árbol
'''
modelo_tree=ArbolClasificador(X_train, X_test, y_train, y_test, columnas_X)
modelo_tree.importancia()

criterion=['gini', 'entropy']           # Función para medir la calidad de la división
#max_depth=[None, 5, 10, 20]            # Profundidad máxima del árbol
max_depth=[4, 5, 6, 7, 10, 20]          # Profundidad máxima del árbol
min_samples_split=[10, 20, 30, 40]      # Número mínimo de muestras para dividir un nodo
min_samples_leaf=[5, 10, 15, 20, 25]    # Número mínimo de muestras por hoja
scoring='f1'
cv=5

modelo_tree.definir_modelo(criterion, max_depth, min_samples_split, min_samples_leaf, scoring, cv)
modelo_tree.entrenar_modelo()
y_pred=modelo_tree.predecir(X_test)

fig=modelo_tree.graficar()
fig.savefig(os.path.join(path_imagenes,'mejor_arbol.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
'''
# Modelamiento Gradient Boosted Trees (GBT o GBM)
'''
modelo_gbm=AumentoGradienteClasificador(X_train, X_test, y_train, y_test)

n_estimators=[50, 100, 150]  # Número de estimadores
learning_rate=[0.01, 0.1, 0.2, 0.5] # Tasa de aprendizaje
max_depth=[3, 5, 7] # Máxima profundidad
scoring='f1'
cv=5

modelo_gbm.definir_modelo(n_estimators, learning_rate, max_depth, scoring, cv)
modelo_gbm.entrenar_modelo()
y_pred=modelo_gbm.predecir(X_test)

# XGBoost (Extreme Gradient Boosting)
# LightGBM (Light Gradient Boosting Machine)
# CatBoost
# Extra Trees (Extemely Randomized Trees)
'''
# Extreme Gradient Boosting (XGBoost)
'''
modelo_xgbm=ExtremoAumentoGradienteClasificador(X_train, X_test, y_train, y_test)

n_estimators=[50, 100, 150]  # Número de estimadores
learning_rate=[0.01, 0.1, 0.2, 0.5] # Tasa de aprendizaje
max_depth=[3, 5, 7] # Máxima profundidad
scoring='f1'
cv=5

modelo_xgbm.definir_modelo(n_estimators, learning_rate, max_depth, scoring, cv)
modelo_xgbm.entrenar_modelo()
y_pred=modelo_xgbm.predecir(X_test)
'''
# LightGBM (Light Gradient Boosting Machine)
'''
modelo_lgbm=LivianoAumentoGradienteClasificador(X_train, X_test, y_train, y_test)

n_estimators=[50, 100, 150]  # Número de estimadores
learning_rate=[0.01, 0.1, 0.2, 0.5] # Tasa de aprendizaje
num_leaves=[15, 31, 63] # ?
max_depth=[3, 5, 7] # Máxima profundidad
scoring='f1'
cv=5

modelo_lgbm.definir_modelo(n_estimators, learning_rate, num_leaves, max_depth, scoring, cv)
modelo_lgbm.entrenar_modelo()
y_pred=modelo_lgbm.predecir(X_test)
'''
# CatBoost
'''
modelo_cb=CategoricalBoostingClasificador(X_train, X_test, y_train, y_test)

iterations=[100, 200, 300]  # Iteraciones
learning_rate=[0.01, 0.1, 0.2, 0.5] # Tasa de aprendizaje
depth=[3, 5, 7] # Profundidad
scoring='f1'
cv=5

modelo_cb.definir_modelo(iterations, learning_rate, depth, scoring, cv)
modelo_cb.entrenar_modelo()
y_pred=modelo_cb.predecir(X_test)
# Extra Trees (Extemely Randomized Trees)
'''
# Modelo Extra Trees
'''
modelo_et=ArbolesExtraClasificador(X_train, X_test, y_train, y_test)

n_estimators=[60, 100, 300]  # Iteraciones
max_depth=[None, 5, 7] # Máxima profundidad
min_samples_split=[3, 6, 15] # Mínima muestra para división
scoring='f1'
cv=5

modelo_et.definir_modelo(n_estimators, max_depth, min_samples_split, scoring, cv)
modelo_et.entrenar_modelo()
y_pred=modelo_et.predecir(X_test)
'''
# Modelo ANN (ANN - Multilayer Perceptron)
'''
modelo_ann=AnnClasificador(X_train, X_test, y_train, y_test)

hidden_layer_sizes=[(50,),(100,),(100,50),(50,25)] # Capas ocultas
activation=['relu','tanh'] # Función de activación
solver=['adam','sgd'] # Solver
alpha=[0.0001, 0.001, 0.001] # Alpha
scoring='f1'
cv=5

modelo_ann.definir_modelo(hidden_layer_sizes, activation, solver, alpha, scoring, cv)
modelo_ann.entrenar_modelo()
y_pred=modelo_ann.predecir(X_test)
'''
# Regresión Logística Regularizada (l1, l2, Elasticnet)
'''
modelo_rlr=RegresionLogisticaRegularizada(X_train, X_test, y_train, y_test)

penalty=['l1','l2','elasticnet']
C=[0.1, 1, 10]
solver=['saga']
l1_ratio=[0.1, 0.6, 0.8] 
scoring='f1'
cv=5

modelo_rlr.definir_modelo(penalty, C, solver, l1_ratio, scoring, cv)
modelo_rlr.entrenar_modelo()
y_pred=modelo_rlr.predecir(X_test)
'''
# Ensamble de modelos
'''
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

modelo_ensamble=EnsableModelos(X_train, X_test, y_train, y_test)

base_models=[('rf', RandomForestClassifier(n_estimators=100)),
             ('gb', GradientBoostingClassifier(n_estimators=100))]
meta_model=LogisticRegression()
cv=5

modelo_ensamble.definir_modelo_stacking(base_models, meta_model, cv)
modelo_ensamble.entrenar_modelo_stacking()
y_pred=modelo_ensamble.predecir_modelo_stacking(X_test)
'''
# Modelo AdaBoost (Adaptative Boosting)
'''
modelo_ada=AdaBoostClasificador(X_train, X_test, y_train, y_test)

n_estimators=[50, 100, 200] # Número de estimadores
learning_rate= [0.01, 0.1, 0.2] # Tasa de aprendizaje
estimator__max_depth=[1, 2, 5] # Máxima profundidad estimador base
estimator__min_samples_split=[2,5,7,10] # Ejemplo de ajuste de árbol
scoring='f1'
cv=5

modelo_ada.definir_modelo(n_estimators, learning_rate, estimator__max_depth, estimator__min_samples_split, scoring, cv)
modelo_ada.entrenar_modelo()
y_pred=modelo_ada.predecir(X_test)
'''
# Modelo HistGradientBoosting
modelo_hgbm=HistGradienteClasificador(X_train, X_test, y_train, y_test)

learning_rate=[0.01, 0.1, 0.2] # Tasa de aprendizaje
max_iter=[100, 200] # Máximo de iteraciones
max_depth=[3,5,7] # Máxima profundidad
scoring='f1'
cv=5

modelo_hgbm.definir_modelo(learning_rate, max_iter, max_depth, scoring, cv)
modelo_hgbm.entrenar_modelo()
y_pred=modelo_hgbm.predecir(X_test)

# Evaluación
evaluacion=Evaluacion(y_test, y_pred)
evaluacion.reporte()
evaluacion.matriz_confusion()
plt.savefig(os.path.join(path_imagenes,'3_confusion.png'), dpi=300, bbox_inches='tight')
plt.close()

fig=evaluacion.curva_roc()
fig.savefig(os.path.join(path_imagenes,'4_roc.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Directorios
root='C:/Users/camed/OneDrive/Escritorio/astro'
carpeta='imagenes'
path_imagenes=os.path.join(root,carpeta)
os.makedirs(path_imagenes, exist_ok=True)

# Funciones y clases
from core.utils import separa_num_cat, escala_num, balance, divide_train_test
from core.estadistica import Correlacion
from core.modelos.modelo_rf import BosqueAleatorioClasificador
from core.evaluacion import Evaluacion
from core.modelos.error_rf import BosqueAleatorioError

# Data
path_data=os.path.join(root,'datos/neo_v2.csv')
df = pd.read_csv(path_data)
print(df.head(3))
print(df.shape)

# Types
print(df.dtypes)
df=df.astype({  'id':'object', 
                'name':'object', 
                'est_diameter_min':'float64',
                'est_diameter_max':'float64',
                'relative_velocity':'float64',
                'miss_distance':'float64',
                'orbiting_body':'object',
                'sentry_object':'int',
                'absolute_magnitude':'float64', 
                'hazardous':'int'   })

# Calidad
print("\nNull:")
print(df.isna().sum())

print("\nDuplicados:")
print(df.duplicated().sum())

print("\nNuméricas:")
print(df.describe().T)

print("\nCategóricas:")
print(df.describe(include=['object']).T)

# Features
df=df.drop(['id', 'name', 'orbiting_body', 'sentry_object', 'est_diameter_min'], axis=1)

print("\nTipos finales:")
print(df.dtypes)

print("\nData:")
print(df)

# Balance
print("\nBalance de clases:")
print(balance(df['hazardous']))

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
X = df.drop('hazardous', axis=1)
y = df['hazardous']

# Escalado
X_scale=escala_num(X)

# OHE (no aplica)

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

# Modelamiento
modelo_rf=BosqueAleatorioClasificador(X_train, X_test, y_train, y_test)

#n_estimators=[50, 100, 150, 200]
#max_depth=[None, 10, 20, 30]
#min_samples_split=[2, 5, 10, 15]
#scoring='recall'
#cv=5

#modelo_rf.definir_modelo(n_estimators, max_depth, min_samples_split, scoring, cv)
modelo_rf.definir_modelo() # defect
modelo_rf.entrenar_modelo()
#modelo_rf.reporte()
#modelo_rf.matriz_confusion()
y_pred=modelo_rf.predecir(X_test)

# Evaluación
evaluacion=Evaluacion(y_test, y_pred)
evaluacion.reporte()
evaluacion.matriz_confusion()
plt.savefig(os.path.join(path_imagenes,'3_confusion.png'), dpi=300, bbox_inches='tight')
plt.close()

fig=evaluacion.curva_roc()
fig.savefig(os.path.join(path_imagenes,'4_roc.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Funciones y clases
from core.utils import separa_num_cat, escala_num, balance, divide_train_test, ohe
from core.estadistica import Correlacion

from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from core.evaluacion import Evaluacion
from core.modelos.error_rf import BosqueAleatorioError

# Data
path_data=os.path.join(root,'datos\\winequality-red-clean.csv')
#print(path_data)

df = pd.read_csv(path_data)
print(df.head(3))
print(df.shape)

# Características y objetivo
X = df.drop('quality_class_Medium', axis=1)
y = df['quality_class_Medium']

# Reserva nombres de columnas
columnas_X = X.columns.tolist()  # Si usas pandas

# Escalado
X_scale=escala_num(X)
print(X_scale)

# Implementación de Pipeline (requiere modelos estándar, no personalizados)
#--------------------------------------------------------------------------------------
# Sets train test
X_train, X_test, y_train, y_test=divide_train_test(X_scale, y, 0.2)

print("\nX_train:")
print(X_train)

print("\ny_train:")
print(y_train)

# Diccionario de modelos y parámetros

modelos_parametros = {
    # Clasificación Logística
    'LogisticRegression': (
        LogisticRegression(solver='liblinear', random_state=42),
        {
            'model__penalty': ['l1', 'l2'],
            'model__C': [0.01, 0.1, 1, 10],
            'model__class_weight': [None, 'balanced'],
        }
    ),
    
    # Clasificación Logística Regularizada (L1-L2-Elastic Net)
    'LogisticRegressionRegularized': (
        LogisticRegression(solver='saga', random_state=42),
        {
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__l1_ratio': [0.1, 0.5, 0.9],
            'model__C': [0.01, 0.1, 1, 10],
            'model__class_weight': [None, 'balanced'],
        }
    ),
    
    # Ridge Classifier
    'RidgeClassifier': (
        RidgeClassifier(random_state=42),
        {
            'model__alpha': [0.1, 1.0, 10.0],
            'model__class_weight': [None, 'balanced'],
        }
    ),
    
    # Árboles de Decisión
    'DecisionTree': (
        DecisionTreeClassifier(random_state=42),
        {
            'model__criterion': ['gini', 'entropy'],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
        }
    ),
    
    # Random Forest
    'RandomForest': (
        RandomForestClassifier(random_state=42),
        {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__class_weight': [None, 'balanced'],
        }
    ),
    
    # Gradient Boosted Trees
    'GradientBoosting': (
        GradientBoostingClassifier(random_state=42),
        {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
        }
    ),
    
    # XGBoost
    'XGBoost': (
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
        }
    ),
    
    # LightGBM
    'LightGBM': (
        LGBMClassifier(random_state=42),
        {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
        }
    ),
    
    # CatBoost
    'CatBoost': (
        CatBoostClassifier(verbose=0, random_state=42),
        {
            'model__iterations': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__depth': [3, 5, 7],
        }
    ),
    
    # Extra Trees
    'ExtraTrees': (
        ExtraTreesClassifier(random_state=42),
        {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
        }
    ),
    
    # K-Nearest Neighbors
    'KNN': (
        KNeighborsClassifier(),
        {
            'model__n_neighbors': [3, 5, 7],
            'model__weights': ['uniform', 'distance'],
        }
    ),
    
    # Support Vector Machines
    'SVM': (
        SVC(probability=True, random_state=42),
        {
            'model__kernel': ['linear', 'rbf', 'poly'],
            'model__C': [0.1, 1, 10],
            'model__gamma': ['scale', 'auto'],
        }
    ),
    
    # Naive Bayes
    'NaiveBayes': (
        GaussianNB(),
        {}
    ),
    
    # Ensemble Stacking
    'Stacking': (
        StackingClassifier(
            estimators=[
                ('lr', LogisticRegression(random_state=42)),
                ('rf', RandomForestClassifier(random_state=42))
            ],
            final_estimator=LogisticRegression(),
        ),
        {}
    ),
    
    # Ada Boost
    'AdaBoost': (
        AdaBoostClassifier(random_state=42),
        {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
        }
    ),
    
    # HistGradientBoostingClassifier
    'HistGradientBoosting': (
        HistGradientBoostingClassifier(random_state=42),
        {
            'model__max_iter': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
        }
    ),
    
    # Perceptron
    'Perceptron': (
        Perceptron(random_state=42),
        {
            'model__penalty': [None, 'l2', 'elasticnet'],
            'model__alpha': [0.0001, 0.001, 0.01],
        }
    ),
    
    # MLP Classifier
    'MLP': (
        MLPClassifier(random_state=42, max_iter=500),
        {
            'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'model__activation': ['relu', 'tanh', 'logistic'],
            'model__alpha': [0.0001, 0.001, 0.01],
        }
    ),
}

# Resultados
resultados=[]
# Entrenamiento y evaluación con GridSearchCV
resultados = []
for nombre, (modelo, params) in modelos_parametros.items():
    print(f"Entrenando {nombre}...")
    pipeline = Pipeline(steps=[('model', modelo)])
    grid_search = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Mejores parámetros para {nombre}: {grid_search.best_params_}")
    y_pred = grid_search.predict(X_test)
    print(f"Reporte de clasificación para {nombre}:\n", classification_report(y_test, y_pred))
    
    resultados.append((nombre, grid_search.best_score_, grid_search.best_params_))

# Resultados finales
print("\nResultados comparativos:")
for modelo, score, params in resultados:
    print(f"{modelo}: F1 Score: {score:.4f}, Parámetros: {params}")
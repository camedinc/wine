# Librerías
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 9. **Árbol**
class ArbolClasificador:
    '''Implementacón con grilla de hiperparámetros'''
    def __init__(self, X_train, X_test, y_train, y_test, columnas_X):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.columnas_X=columnas_X
        self.y_pred=None
        self.tree_model=None
        self.grid_search=None
        self.best_model=None
        self.best_hiperparametros=None

    def definir_modelo(self, criterion=['gini'], max_depth=[5,10], min_samples_split=[10,15], min_samples_leaf=[5,10], scoring='accuracy', cv=None): # sin cv

        '''Crea el modelo'''
        self.tree_model = DecisionTreeClassifier(random_state=42) # problema desbalanceado 10% class 1
        print("\nModelo definido\n")
        
        '''Define la grilla'''
        param_grid = {
            'criterion': criterion,                   # Función para medir la calidad de la división
            'max_depth': max_depth,                   # Profundidad máxima del árbol
            'min_samples_split': min_samples_split,   # Número mínimo de muestras para dividir un nodo
            'min_samples_leaf': min_samples_leaf      # Número mínimo de muestras por hoja
        }

        self.grid_search = GridSearchCV(
        estimator=self.tree_model, 
        param_grid=param_grid, 
        scoring=scoring, 
        cv=cv,  # Validación cruzada
        verbose=1, 
        n_jobs=-1
        )
    
    def entrenar_modelo(self):
        '''Entrena el modelo con la grilla'''
        if self.grid_search is None:
            raise ValueError("Se debe definir el modelo")
        else:
            self.grid_search.fit(self.X_train, self.y_train)

        # Mejores hiperparámetros
        self.best_hiperparametros=self.grid_search.best_params_

        print("\nMejores hiperparámetros:\n")
        print(self.best_hiperparametros)

        print("\nModelo asignado!\n")
        self.best_model = self.grid_search.best_estimator_

    def reporte(self):
        '''Reporte métricas'''
        self.y_pred = self.best_model.predict(self.X_test)
        print("\nReporte de clasificación mejor modelo:\n")
        print(classification_report(self.y_test, self.y_pred))
        return self.y_pred

    def matriz_confusion(self):
        '''Matriz de confusión'''
        print("\nMatriz de confusión:\n")
        print(confusion_matrix(self.y_test, self.y_pred))
    
    def predecir(self, X):
        '''Predecir con el mejor modelo entrenado'''
        if self.best_model is None:
            raise ValueError("El modelo no ha sido entrenado")
        else:
            y = self.best_model.predict(X)
        return y
    
    def graficar(self):
        fig, ax=plt.subplots(figsize=(30, 15))
        plot_tree( self.best_model, 
                   filled=True,  # Colorear los nodos según la clase mayoritaria
                   feature_names=[f"Feature {i}" for i in range(self.X_train.shape[1])],  # Nombres de las características
                   class_names=["Clase 0", "Clase 1"],  # Nombres de las clases (binaria)
                   rounded=True,  # Bordes redondeados
                   fontsize=8, ax=ax)
        plt.title("Árbol de Decisión Entrenado")
        return fig
    def importancia(self):
        modelo_tree = DecisionTreeClassifier(random_state=42)
        modelo_tree.fit(self.X_train, self.y_train)
        importances = modelo_tree.feature_importances_

        # Ordenar las importancias
        indices = np.argsort(importances)[::-1]

        # Imprimir las características más importantes
        for i in range(self.X_train.shape[1]):
            print(f"{i+1}. {self.columnas_X[indices[i]]} - Importancia: {round(importances[indices[i]],3)}")
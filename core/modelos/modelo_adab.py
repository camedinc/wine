# Librerías
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# 9. **AdaBoost**
class AdaBoostClasificador:
    '''Implementacón con grilla de hiperparámetros'''
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.y_pred=None
        self.ada_model=None
        self.grid_search=None
        self.best_model=None
        self.best_hiperparametros=None

    def definir_modelo(self, n_estimators=[50, 100, 200], learning_rate= [0.01, 0.1, 0.2], estimator__max_depth=[1, 2, 3], estimator__min_samples_split=[2,5,10], scoring='roc_auc', cv=None): # sin cv

        '''Crea el modelo'''
        self.ada_model = AdaBoostClassifier(estimator=DecisionTreeClassifier())
        print("\nModelo definido!\n")
        
        '''Define la grilla'''
        param_grid = {
            'n_estimators': n_estimators,  # Número de estimadores
            'learning_rate': learning_rate,  # Tasa de aprendizaje
            'estimator__max_depth': estimator__max_depth, # Hiperparámetro para el árbol de decisión
            'estimator__min_samples_split':estimator__min_samples_split # Ejemplo de ajuste de árbol
            }
        
        self.grid_search = GridSearchCV(
        estimator=self.ada_model, 
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
# Librerías
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# 9. **Ensamble de modelos**
class EnsableModelos:
    '''Implementacón con grilla de hiperparámetros'''
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.y_pred=None
        self.base_models=None
        self.meta_model=None
        self.stacking_model=None
        self.grid_search=None
        self.best_model=None
        self.best_hiperparametros=None

    def definir_modelo_stacking(self, base_models, meta_model, cv=None): # sin cv

        '''Crea modelos de base'''
        self.base_models = base_models
        print("\nModelos base definidos\n")

        '''Crea modelo meta'''
        self.meta_model = meta_model
        print("\nModelo meta definido\n")

        '''Define modelo stacking'''
        self.stacking_model=StackingClassifier(estimators=self.base_models, final_estimator=self.meta_model, cv=cv)
    
    def entrenar_modelo_stacking(self):
        '''Entrena el modelo ensamble'''
        if self.stacking_model is None:
            raise ValueError("Se debe definir el modelo")
        else:
            self.stacking_model.fit(self.X_train, self.y_train)
            print("Modelo stacking entrenado")

    def reporte_modelo_stacking(self):
        '''Reporte métricas'''
        self.y_pred = self.stacking_model.predict(self.X_test)
        print("\nReporte de clasificación modelo stacking:\n")
        print(classification_report(self.y_test, self.y_pred))
        return self.y_pred

    def matriz_confusion_modelo_stacking(self):
        '''Matriz de confusión modelo stacking'''
        print("\nMatriz de confusión modelo stacking:\n")
        print(confusion_matrix(self.y_test, self.y_pred))
    
    def predecir_modelo_stacking(self, X):
        '''Predecir con el mejor modelo entrenado'''
        if self.stacking_model is None:
            raise ValueError("El modelo no ha sido entrenado")
        else:
            y = self.stacking_model.predict(X)
        return y
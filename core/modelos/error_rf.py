# Librerías
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

# Relación de error vs. número de árboles (estimadores)
class BosqueAleatorioError:
    def __init__(self, X, y, test_size=0.2, n_estimators=200, pasos=10):
        self.X=X
        self.y=y
        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None
        self.y_train_pred=None
        self.y_test_pred=None
        self.test_size=test_size
        self.n_estimators=n_estimators
        self.pasos=pasos

    def calcular_error(self):
        self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)
        train_errors=[]
        test_errors=[]

        n_estimators_rango = np.arange(1, self.n_estimators + 1, self.pasos)
        for estimators in n_estimators_rango:
            modelo=RandomForestClassifier(n_estimators=estimators, random_state=42)
            modelo.fit(self.X_train, self.y_train)

            self.y_train_pred=modelo.predict(self.X_train)
            self.y_test_pred=modelo.predict(self.X_test)

            train_errors.append(1-recall_score(self.y_train, self.y_train_pred)) # recall desblanceadas
            test_errors.append(1-recall_score(self.y_test, self.y_test_pred))

        fig, ax=plt.subplots()
        ax.plot(n_estimators_rango, train_errors, label="Error de entrenamiento", marker='o')
        ax.plot(n_estimators_rango, test_errors, label="Error de prueba", marker='o')
        ax.set_xlabel("Número de árboles (n_estimators)")
        ax.set_ylabel("Error")
        ax.set_title("Evolución del Error vs. Número de Árboles")
        ax.legend()
        ax.grid(True)
        return fig

# Librerías
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay

# Evaluación CamelCase/snake_case
class Evaluacion:
    def __init__(self, y_test, y_pred):
        self.y_test=y_test
        self.y_pred=y_pred
        '''Reporte'''
    def reporte(self):
        print("Evaluación-Reporte de clasificación:")
        print(classification_report(self.y_test, self.y_pred))
    
    def matriz_confusion(self):
        '''Matriz heat map'''
        print("Evaluación-Matriz de confusión:")
        cm=confusion_matrix(self.y_test, self.y_pred)

        fig, ax=plt.subplots()
        disp=ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=ax)
        ax.set_title("Matriz de confusión")
        return fig

    def curva_roc(self):
        '''Gráfica AUC ROC'''
        print("Evaluación-curva_roc:")
        fpr, tpr, thresholds=roc_curve(self.y_test, self.y_pred)
        roc_auc=auc(fpr, tpr)

        fig, ax=plt.subplots()
        disp=RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="example estimator")
        disp.plot(ax=ax)
        ax.set_title("Curva ROC")
        return fig
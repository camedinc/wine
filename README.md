# **Clasificación de Vinos con Modelos de Machine Learning**

### **Descripción del Proyecto**
El proyecto **Wine** se centra en la clasificación y análisis de datos de vinos mediante el uso de múltiples algoritmos de Machine Learning implementados como clases modulares. Esta estructura orientada a objetos (OOP) proporciona una solución escalable y mantenible para el desarrollo y experimentación con diversos modelos de clasificación.

---

### **Características Principales**
1. **Estructura Orientada a Objetos**:
   - Cada modelo de Machine Learning está implementado como una clase independiente, lo que facilita su comprensión, mantenimiento y escalabilidad.
   - Las clases incluyen funcionalidades como ajuste del modelo, predicción y evaluación del desempeño.
   - Modularidad: Cambiar o probar un modelo es tan sencillo como instanciar la clase correspondiente.

2. **Clases Esenciales del Proyecto**:
   - **Evaluación**: Proporciona reportes de clasificación, matrices de confusión y curvas ROC/AUC.
   - **Correlación**: Permite analizar y visualizar las relaciones entre las variables en un mapa de calor.
   - **Preprocesamiento**: Funciones para escalar datos, balancear clases y realizar codificación One-Hot.

3. **Variedad de Algoritmos Implementados**:
   - Modelos tradicionales como Regresión Logística, Naive Bayes y Árboles de Decisión.
   - Algoritmos avanzados como Gradient Boosting, XGBoost, LightGBM y Redes Neuronales.
   - Métodos de ensamble como Random Forest y AdaBoost.

4. **Automatización y Flexibilidad**:
   - El preprocesamiento de datos, la selección de modelos y la evaluación se pueden realizar fácilmente gracias al diseño modular.
   - Escalable para incluir nuevos algoritmos o técnicas.

---

### **Ventajas de la Estructura del Proyecto**
- **Claridad y Legibilidad**:
  La implementación de clases separadas para cada modelo y componente del pipeline hace que el código sea intuitivo y fácil de seguir.
  
- **Reutilización**:
  Las clases y funciones pueden ser reutilizadas en otros proyectos con mínimos cambios, fomentando un desarrollo eficiente.

- **Escalabilidad**:
  Añadir nuevos modelos, métricas de evaluación o técnicas de preprocesamiento es sencillo y no afecta la estructura existente.

- **Mantenibilidad**:
  El diseño modular permite identificar y solucionar errores de manera localizada sin impactar otras partes del proyecto.

- **Flexibilidad**:
  Los usuarios pueden combinar diferentes componentes (clases y funciones) según sus necesidades específicas, ajustando desde el preprocesamiento hasta la evaluación.

---

### **Estructura del Proyecto**
```plaintext
wine/
│
├── core/                          # Código fuente principal
│   ├── modelos/                   # Modelos implementados como clases
│   │   ├── modelo_rf.py           # Random Forest
│   │   ├── modelo_svm.py          # Support Vector Machine
│   │   ├── modelo_xgbm.py         # XGBoost
│   │   ├── modelo_lgbm.py         # LightGBM
│   │   ├── modelo_ann.py          # Redes Neuronales
│   │   └── ...                    # Otros modelos
│   ├── evaluacion.py              # Clase Evaluación
│   ├── estadistica.py             # Clase Correlación
│   └── preprocesamiento.py        # Funciones de preprocesamiento
├── datos/                         # Datos crudos y procesados
│   ├── vinos_original.csv         # Datos originales
│   └── vinos_limpios.csv          # Datos preprocesados
├── imagenes/                      # Visualizaciones generadas
│   ├── matriz_correlacion.png     # Matriz de correlación
│   ├── curva_roc.png              # Curva ROC
│   └── matriz_confusion.png       # Matriz de confusión
└── README.md                      # Documentación del proyecto
```

---

### **Cómo Utilizar el Proyecto**
#### **1. Requisitos**
- Python 3.8 o superior.
- Librerías necesarias: `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`.

Instala las dependencias con:
```bash
pip install -r requirements.txt
```

#### **2. Flujo de Trabajo**
1. **Preprocesamiento**:
   - Usa las funciones del módulo `preprocesamiento.py` para escalar los datos, balancear clases y codificar variables categóricas.

2. **Selección de Modelo**:
   - Instancia la clase del modelo que deseas usar (e.g., `modelo_rf.RandomForest` para Random Forest).
   - Ajusta los hiperparámetros y entrena el modelo con tus datos.

3. **Evaluación**:
   - Usa la clase `Evaluacion` para generar reportes de clasificación, matriz de confusión y curva ROC.

4. **Análisis**:
   - Analiza las relaciones entre variables con la clase `Correlacion`.

---

### **Visualizaciones Generadas**
#### **Matriz de Correlación**:
![Matriz de Correlación](wine/imagenes/matriz_correlacion.png)

#### **Curva ROC**:
![Curva ROC](imagenes/curva_roc.png)

#### **Matriz de Confusión**:
![Matriz de Confusión](imagenes/matriz_confusion.png)

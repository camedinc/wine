# Librerías
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Separa variables numéricas y objects
def separa_num_cat(df):
    df_num=df.select_dtypes(include=['number'])
    df_cat=df.select_dtypes(exclude=['number'])
    return df_num, df_cat

# Escalado
def escala_num(df):
    scaler=StandardScaler()
    df_escalado=scaler.fit_transform(df)
    return df_escalado

# Entrenamiento y prueba
def divide_train_test(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Balance de clases
def balance(columna):
    print("Balance (%):\n", columna.value_counts(normalize=True))

# One Hot Encoding
def ohe(df):
    # Separar
    df_num=df.select_dtypes(include=float, exclude=object)
    df_cat=df.select_dtypes(include=object, exclude=float)
    # OHE categóricas
    df_encoded = pd.get_dummies(df_cat, drop_first=True)
    # Unir
    df_ohe=pd.concat([df_num, df_encoded], axis=1)
    return df_ohe
# Librerías
import os
import pandas as pd

# Data
path_data='C:\\Users\\camed\\OneDrive\\Documentos\\Git\\wine\\datos'
path_data_red=os.path.join(path_data,'winequality-red.csv')
path_data_white=os.path.join(path_data,'winequality-white.csv')
path_data_names=os.path.join(path_data,'winequality.names')

df_red = pd.read_csv(path_data_red, sep=';')
df_white = pd.read_csv(path_data_white, sep=';')
print(df_red.head(3))
print(df_red.head(3))

with open(path_data_names, 'r') as file:
    contenido = file.read()
    print(contenido)

# Escritura y formato csv
df_red.to_csv(path_data_red, sep=',', index=True, encoding='utf-8')
df_white.to_csv(path_data_white, sep=',', index=True, encoding='utf-8')
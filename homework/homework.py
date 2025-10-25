#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import os
import pandas as pd
import numpy as np
import gzip
import pickle
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


def cargar_datos():
    # Leo los datos comprimidos
    df_train = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    df_test = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    return df_train, df_test


def preparar_datos(df_train, df_test):
    # Creo la edad del carro
    df_train["Age"] = 2021 - df_train["Year"]
    df_test["Age"] = 2021 - df_test["Year"]

    # Quito columnas que no sirven
    df_train = df_train.drop(["Year", "Car_Name"], axis=1)
    df_test = df_test.drop(["Year", "Car_Name"], axis=1)

    # Separo variables
    x_train = df_train.drop("Present_Price", axis=1)
    y_train = df_train["Present_Price"]
    x_test = df_test.drop("Present_Price", axis=1)
    y_test = df_test["Present_Price"]

    return x_train, y_train, x_test, y_test


def crear_pipeline(x_train):
    # Defino columnas
    categorical = ["Fuel_Type", "Selling_type", "Transmission"]
    carac_num = [col for col in x_train.columns if col not in categorical]

    # Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), carac_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        ],
        remainder=MinMaxScaler(),
    )

    # Armo el pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("Skbest", SelectKBest(score_func=f_regression)),
        ("model", LinearRegression())
    ])

    return pipeline


def entrenar_modelo(pipeline, x_train, y_train):
    # Defino los parámetros para buscar
    parametros = {"Skbest__k": range(1, 20)}

    modelo = GridSearchCV(
        pipeline,
        parametros,
        scoring="neg_mean_absolute_error",
        cv=10,
        n_jobs=-1
    )

    # Entreno el modelo
    modelo.fit(x_train, y_train)
    return modelo


def guardar_modelo(modelo):
    # Guardo el modelo comprimido
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pickle.dump(modelo, file)


def evaluar_modelo(modelo, x_train, y_train, x_test, y_test):
    # Hago las predicciones
    y_train_pred = modelo.predict(x_train)
    y_test_pred = modelo.predict(x_test)

    # Calculo las métricas
    metrics_train = {
        "type": "metrics",
        "dataset": "train",
        "r2": float(r2_score(y_train, y_train_pred)),
        "mse": float(mean_squared_error(y_train, y_train_pred)),
        "mad": float(median_absolute_error(y_train, y_train_pred))
    }

    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "r2": float(r2_score(y_test, y_test_pred)),
        "mse": float(mean_squared_error(y_test, y_test_pred)),
        "mad": float(median_absolute_error(y_test, y_test_pred)),
    }

    return metrics_train, metrics_test


def guardar_metricas(metrics_train, metrics_test):
    # Guardo las métricas en un archivo JSON
    os.makedirs("files/output", exist_ok=True)
    output_path = os.path.join("files/output", "metrics.json")
    with open(output_path, "w") as f:
        f.write(json.dumps(metrics_train) + "\n")
        f.write(json.dumps(metrics_test) + "\n")


def main():
    df_train, df_test = cargar_datos()
    x_train, y_train, x_test, y_test = preparar_datos(df_train, df_test)
    pipeline = crear_pipeline(x_train)
    modelo = entrenar_modelo(pipeline, x_train, y_train)
    guardar_modelo(modelo)
    metrics_train, metrics_test = evaluar_modelo(modelo, x_train, y_train, x_test, y_test)
    guardar_metricas(metrics_train, metrics_test)
    print("Modelo entrenado y métricas guardadas.")


if __name__ == "__main__":
    main()

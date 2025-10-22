# src/validate.py
import os
import pathlib
import pickle
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_diabetes  # Importar load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Parámetro de umbral
THRESHOLD = 0.7

print("--- Debug: Cargando dataset de validación ---")


# Generación de x y y
# data = pd.read_csv(os.path.join(Path.cwd().parent.as_posix(), "data", "validation.csv"))
# data = "../data/validation.csv"
data = pd.read_csv("data/validation.csv")
X_valid = data.drop(columns="Exited")
y_valid = data[["Exited"]]

print(f"--- Debug: Dimensiones de X_valid: {X_valid.shape} ---")

# --- Cargar modelo previamente entrenado ---
# model_path = os.path.join(Path.cwd().parent.as_posix(), "pkl", "model.pkl")
model_path = "pkl/model.pkl"
print(f"--- Debug: Intentando cargar modelo desde: {model_path} ---")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(
        f"--- ERROR: No se encontró el archivo del modelo en '{model_path}'. Asegúrate de que el paso 'make train' lo haya guardado correctamente en la raíz del proyecto. ---"
    )
    # Listar archivos en el directorio actual para depuración
    print(f"--- Debug: Archivos en {os.getcwd()}: ---")
    try:
        print(os.listdir(os.getcwd()))
    except Exception as list_err:
        print(f"(No se pudo listar el directorio: {list_err})")
    print("---")
    sys.exit(1)  # Salir con error

# --- Predicción y Validación ---
print("--- Debug: Realizando predicciones ---")
try:
    y_pred = model.predict(X_valid)  # Ahora X_valid tiene 10 features
except ValueError as pred_err:
    print(f"--- ERROR durante la predicción: {pred_err} ---")
    # Imprimir información de características si el error persiste
    print(f"Modelo esperaba {model.n_features_in_} features.")
    print(f"X_valid tiene {X_valid.shape[1]} features.")
    sys.exit(1)

accuracy = model.score(X_valid, y_valid)
print(f"🔍 Accuracy del modelo: {accuracy:.4f} (umbral: {THRESHOLD})")

# Validación
if accuracy >= THRESHOLD:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)  # éxito
else:
    print("❌ El modelo no cumple el umbral. Deteniendo pipeline.")
    sys.exit(1)  # error

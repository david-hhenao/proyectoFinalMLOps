import datetime
import os
import pathlib
import pickle
import sys
import traceback
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

print(f"--- Debug: Initial CWD: {os.getcwd()} ---") # Indicar y crear rutas de tracking de MLflow

workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
artifact_location = "file://" + os.path.abspath(mlruns_dir)

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")

os.makedirs(mlruns_dir, exist_ok=True)
mlflow.set_tracking_uri(tracking_uri)

dt_srt = datetime.datetime.now().strftime("%Y%m%d-%H%M")
experiment_name = f"CI-CD-ProyectoFinal-{dt_srt}"
experiment_id = None 
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location,
    )
    print(
        f"--- Debug: Creado Experimento '{experiment_name}' con ID: {experiment_id} ---"
    )
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(
            f"--- Debug: Experimento '{experiment_name}' ya existe. Obteniendo ID. ---"
        )
        experiment = mlflow.get_experiment_by_name(experiment_name) # Obtener el experimento existente para conseguir su ID
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"--- Debug: ID del Experimento Existente: {experiment_id} ---")
            print(
                f"--- Debug: Ubicación de Artefacto del Experimento Existente: {experiment.artifact_location} ---"
            )
            if experiment.artifact_location != artifact_location: # Opcional: Verificar si la ubicación del artefacto es la correcta
                print(
                    f"--- WARNING: La ubicación del artefacto del experimento existente ('{experiment.artifact_location}') NO coincide con la deseada ('{artifact_location}')! ---"
                )
        else: # Esto no debería ocurrir si RESOURCE_ALREADY_EXISTS fue el error
            print(
                f"--- ERROR: No se pudo obtener el experimento existente '{experiment_name}' por nombre. ---"
            )
            sys.exit(1)
    else:
        print(f"--- ERROR creando/obteniendo experimento: {e} ---")
        raise e 

if experiment_id is None:
    print(
        f"--- ERROR FATAL: No se pudo obtener un ID de experimento válido para '{experiment_name}'. ---"
    )
    sys.exit(1)

# --- Cargar Datos y Entrenar Modelo ---

le_gender = LabelEncoder()
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
scaler = StandardScaler(with_mean=False)

data_path = "data/Churn_Modelling.csv"
print(os.getcwd())

data = pd.read_csv(data_path)[
    [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
]

# como son variables categoricas binarias se utiliza LabelEncoder
data["Gender"] = le_gender.fit_transform(
    data["Gender"]
)
# Como son variables categoricas no binarias se utiliza OneHotEncoder
X_geo = ohe.fit_transform(
    data[["Geography"]]
)
geo_df = pd.DataFrame(
    X_geo, columns=ohe.get_feature_names_out(["Geography"]), index=data.index
)
data[["Geography_A", "Geography_B"]] = geo_df.astype(int)
data = data.drop(columns="Geography")
X = data.drop(columns="Exited")
y = data["Exited"]
del X_geo, geo_df, data

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Se genearan tres dataframes para la validación

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns=X_train.columns)

X_val["Exited"] = y_val.values
X_val.to_csv("data/validation.csv", index=False)

svm = SVC(kernel="rbf", class_weight="balanced")
svm.fit(X_train, y_train)

accuracy = svm.score(X_test, y_test)

os.makedirs("pkl", exist_ok=True)

with open("pkl/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("pkl/ohe.pkl", "wb") as f:
    pickle.dump(ohe, f)

with open("pkl/le_gender.pkl", "wb") as f:
    pickle.dump(le_gender, f)

with open("pkl/model.pkl", "wb") as f:
    pickle.dump(svm, f)

# --- Iniciar Run de MLflow ---
print(
    f"--- Debug: Iniciando run de MLflow en Experimento ID: {experiment_id} ---"
) 
run = None
try:
    # Iniciar el run PASANDO EXPLÍCITAMENTE el experiment_id
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        actual_artifact_uri = run.info.artifact_uri
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: URI Real del Artefacto del Run: {actual_artifact_uri} ---")

        # Comprobar si coincide con el patrón esperado basado en artifact_location del experimento
        # (La artifact_uri del run incluirá el run_id)
        expected_artifact_uri_base = os.path.join(
            artifact_location, run_id, "artifacts"
        )
        if actual_artifact_uri != expected_artifact_uri_base:
            print(
                f"--- WARNING: La URI del Artefacto del Run '{actual_artifact_uri}' no coincide exactamente con la esperada '{expected_artifact_uri_base}' (esto puede ser normal si la estructura difiere ligeramente). Lo importante es que NO sea la ruta local incorrecta. ---"
            )

        signature = infer_signature(X_train, svm.predict(X_train))
        input_example = X_train.head(3)

        mlflow.log_metric("Accuracy", accuracy)
        print(f"--- Debug: Intentando log_model con artifact_path='model' ---")

        mlflow.sklearn.log_model(
            sk_model=svm,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )
        print(f"✅ Modelo registrado correctamente. Accuracy: {accuracy:.4f}")

except Exception as e:
    print(f"\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    print(f"--- Fin de la Traza de Error ---")
    print(f"CWD actual en el error: {os.getcwd()}")
    print(f"Tracking URI usada: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID intentado: {experiment_id}")
    if run:
        print(f"URI del Artefacto del Run en el error: {run.info.artifact_uri}")
    else:
        print("El objeto Run no se creó con éxito.")
    sys.exit(1)

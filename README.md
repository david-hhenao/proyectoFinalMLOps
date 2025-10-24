# **Proyecto final MLOps - Grupo 9**

## **Integrantes:** David Hernando Henao y Mario Andrés Rodriguez

1. ¿Cómo ejecutarlo?:

    - Clonar el repo:
    ```bash
        git clone https://github.com/david-hhenao/proyectoFinalMLOps.git
    ```
    - Ingresar a la carpeta del repo:
    ```bash
        cd proyectoFinalMLOps
    ```
    - Correr el pipeline por medio de make:
    ```bash
        make ci_sc
    ```
    - Ejectutar la UI de MLFlow:
    ```bash
        make show_mlflow
    ```

2. Estructura del repo:

    ```
    proyectoFinalMLOps/
        ├─ .github/workflows/     # Pipelines de CI / CD 
        ├─ .venv                  # Entorno virtual *
        ├─ data/                  # Datos de entrenamiento y validación *
        ├─ mlruns/                # (Se crea al correr) tracking local de MLflow *
        ├─ pkl/                   # Archivos binarios, tales como: LabelEncoder, OneHotEncoder y Scaler *
        ├─ src/                   # Código fuente
        ├─ .gitignore             # Documento y/o rutas omitidas
        ├─ LICENSE                # Licencia
        ├─ Makefile               # Atajos de ejecución
        ├─ README.md              # Este documento
        └─ requirements.txt       # Dependencias del proyecto
    
    * POSTERIOR DE LA CORRIDA / POR PRIMERA VEZ
    ```
        
3. Uso de MLFlow

    Al ejecutar ```make ci_sc``` se crea la carpeta MLFlow, se puede ejecutar  ```make show_mlflow``` para ver el UI de MLFlow para ver los parámetros registrados, metricas y artefactos.

4. Resultados

    | Conjunto   | Accuracy | F1 (Macro) |
    |:-----------|:--------:|:-----------|
    | Validación | 0.8102 | 0.7833 | 
    | Test       | 0.8212 | 0.7990 | 
    | Límite     | 0.75   | 0.7    |
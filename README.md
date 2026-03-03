# 🫀 Modelo Predictivo de Paro Cardíaco

<p align="center">
  <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
      <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" alt="CCDS Project Template"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white" alt="Python 3.13"/>
  <img src="https://img.shields.io/badge/Status-Completado-success.svg" alt="Status"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
</p>

---

## 📖 Descripción del Proyecto
Este proyecto de análisis de ciencia de datos utiliza Machine Learning para **predecir el riesgo de que un paciente sufra un paro cardíaco**. Para ello, se indaga exhaustivamente en las características clínicas, los signos vitales basales y los biomarcadores, determinando cuáles factores biomecánicos incrementan fuertemente la probabilidad de un evento adverso.  

El objetivo fundamental es crear un modelo capaz de distinguir el riesgo inminente de forma precisa, interpretable y médicamente verificable, priorizando la sensibilidad (Recall) para no dejar escapar a ningún paciente en situación de emergencia.

## 🗂️ Organización del Proyecto (Cookiecutter Data Science)

```text
├── LICENSE            <- Licencia de código abierto del proyecto (MIT).
├── Makefile           <- Comandos automatizados usando `make format` o `make lint` etc.
├── README.md          <- El archivo principal de documentación para todo el grupo de trabajo.
│
├── data
│   ├── external       <- Datos provenientes de fuentes externas.
│   ├── interim        <- Datos intermedios tras la primera fase de limpieza y transformación.
│   ├── processed      <- Los conjuntos de datos canónicos, normalizados y listos para modelar.
│   └── raw            <- Datos base en bruto (ej. HeartAttackDataSet.csv) e inmutables.
│
├── docs               <- Diseño de documentación por MkDocs.
│
├── models             <- Modelos serializados (entrenados), predicciones e inferencias.
│                         Incluye `modelo_produccion.pkl`.
│
├── notebooks          <- Jupyter Notebooks utilizados para Análisis Exploratorio (EDA).
│                         (e.g. exploratoriamente `1.0-eda-paro-cardiaco.ipynb`).
│
├── pyproject.toml     <- Configuración y metadata del empaquetado del proyecto y linting (Ruff/Black).
│
├── references         <- Diccionarios de datos, guías médicas o manuales técnicos adicionales.
│
├── reports            <- Análisis documentado y resumido del proyecto en forma de Reporte.
│   └── figures        <- Imágenes, matrices y gráficas guardadas desde los análisis.
│
├── requirements.txt   <- Librerías y dependencias utilizadas para clonar el entorno.
│
└── src                <- Scripts de código fuente (Módulos de Python).
    ├── __init__.py    <- Transforma el directorio src en un módulo nativo de Python.
    ├── config.py      <- Almacena variables y configuraciones constantes para todo el ciclo.
    ├── dataset.py     <- Scripts para leer, limpiar, preprocesar y generar los datos `processed`.
    ├── features.py    <- Lógica para generar y construir features antes del modelado.
    ├── modeling       <- Scripts referentes al corazón del Machine Learning.
    │   ├── __init__.py
    │   ├── predict.py <- Código listo para inferir en base al modelo pkl guardado.
    │   └── train.py   <- Setup y configuración de entrenamiento Random Forest.
    └── plots.py       <- Utilidades modulares para visualizar (Hist, Pairplots, BoxPlots).
```

## 🛠️ Tecnologías Usadas
- **Procesamiento de Datos:** Pandas, Numpy.
- **Análisis Estadístico:** Scipy (Shapiro-Wilk).
- **Machine Learning:** Scikit-Learn (Pipelines, RandomForest, StandardScaler).
- **Interpretación (XAI):** SHAP values.
- **Visualización:** Matplotlib y Seaborn.
- **Formato y Estructura:** Cookiecutter Data Science.

## 🚀 Instalación y Reproducción
Instala los requisitos a través del archivo correspondiente:
```bash
pip install -r requirements.txt
```
Luego navega hasta los Notebooks o corre los módulos de `/src`.

---
> Proyecto confeccionado priorizando **la ciencia de datos aplicable al ámbito de la salud de precisión**.

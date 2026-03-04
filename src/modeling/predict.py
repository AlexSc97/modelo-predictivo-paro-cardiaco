import pandas as pd
import joblib
import os

def load_model(model_path="models/modelo_produccion.pkl"):
    """
    Carga el modelo serializado .pkl u otro.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo {model_path} no fue encontrado.")
    return joblib.load(model_path)

def make_prediction(data, model_path="models/modelo_produccion.pkl"):
    """
    Recibe un dataframe o diccionario y retorna la predicción realizada.
    """
    model = load_model(model_path)
    
    # Validacion simple
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)[:, 1]
    
    return prediction, prediction_proba

if __name__ == "__main__":
    import numpy as np
    
    print("Prueba de inferencia del modelo:")
    # Valores de ejemplo de características necesarias
    sample_data = {
        'age': 50,
        'sex': 1,
        'cp': 2,
        'trestbps': 130,
        'chol': 250,
        'fbs': 0,
        'restecg': 1,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 1.0,
        'slope': 2,
        'ca': 0,
        'thal': 2
    }
    
    df_sample = pd.DataFrame([sample_data])
    
    try:
        pred, pred_proba = make_prediction(df_sample)
        print(f"Prediccion final: Clase {pred[0]} (Riesgo: {pred_proba[0]:.2%})")
    except Exception as e:
        print(f"Error realizando predicción de prueba: {e}")

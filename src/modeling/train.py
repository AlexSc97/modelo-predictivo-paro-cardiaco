import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_model(data_path="data/processed/clean_dataset.csv", model_output="models/modelo_produccion.pkl"):
    print("Cargando datos procesados para el entrenamiento...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Archivo de datos {data_path} no encontrado.")
    
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Tamaño de conjunto de entrenamiento: {X_train.shape}")
    print(f"Tamaño de conjunto de prueba: {X_test.shape}")

    preprocesador = StandardScaler()
    modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')

    pipeline = Pipeline([
        ('scaler', preprocesador),
        ('classifier', modelo_rf)
    ])

    print("Iniciando entrenamiento...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nResultados del Entranamiento :")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Guardar modelo
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(pipeline, model_output)
    print(f"\nModelo guardado exitosamente en: {model_output}")
    return pipeline

if __name__ == "__main__":
    train_model()

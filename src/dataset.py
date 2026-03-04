import pandas as pd
import os

def load_data(file_path="data/raw/HeartAttackDataSet.csv"):
    """
    Carga los datos en bruto y los devuelve como un DataFrame de Pandas.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no fue encontrado.")
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Realiza las operaciones de limpieza de datos.
    - Eliminación de registros duplicados.
    """
    df_cleaned = df.drop_duplicates().copy()
    return df_cleaned

def save_data(df, output_path="data/processed/clean_dataset.csv"):
    """
    Guarda el dataframe limpio o procesado en la ruta seleccionada.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Datos guardados exitosamente en {output_path}")

if __name__ == "__main__":
    print("Iniciando procesamiento de datos...")
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    save_data(df_clean)
    print("Preprocesamiento inicial finalizado.")

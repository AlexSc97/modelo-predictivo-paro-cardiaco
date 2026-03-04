from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "HeartAttackDataSet.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "clean_dataset.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "modelo_produccion.pkl"

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 5
CLASS_WEIGHT = 'balanced'

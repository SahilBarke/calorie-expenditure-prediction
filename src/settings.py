# settings.py
from typing import List
from pathlib import Path

# Path to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_FILE = DATA_DIR / "Calorie_expenditure.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "calorie_expenditure_model.pkl"

def get_feature_columns() -> List[str]:
    """Return the list of feature column names used for modeling."""
    return [
        "Age",
        "Height",
        "Weight",
        "Duration",
        "Heart_Rate",
        "Body_Temp",
        "BMI",
        "BMI_Category"
    ]

def get_target_column() -> str:
    """Return the name of the target column."""
    return "Calories"

RANDOM_SEED = 42  # Seed for reproducibility

TEST_SIZE = 0.2  # Proportion of data to be used for testing

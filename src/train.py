import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from src.settings import (
    get_feature_columns,
    get_target_column,
    RANDOM_SEED,
    TEST_SIZE,
    MODEL_FILE,
)

from src.settings import MODEL_FILE


def train_model(
    data: pd.DataFrame,
    cv_folds: int = 5,
) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    """
    Train a machine learning model using cross-validation on training data.

    Returns:
        - trained model pipeline
        - X_test
        - y_test
    """

    # --------------------------
    # Select features & target
    # --------------------------
    feature_columns = get_feature_columns()
    target_column = get_target_column()

    X = data[feature_columns]
    y = data[target_column]

    # --------------------------
    # Train / Test split
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    # --------------------------
    # Identify column types
    # --------------------------
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # --------------------------
    # Preprocessing
    # --------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # --------------------------
    # Build pipeline
    # --------------------------
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                random_state=RANDOM_SEED,
                # n_estimators=200,
            )),
        ]
    )

    # --------------------------
    # Cross-validation on TRAIN
    # --------------------------
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv_folds,
        scoring="r2",
    )

    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f}")

    # --------------------------
    # Train final model
    # --------------------------
    model.fit(X_train, y_train)
    
    # --------------------------
    # Save the model
    #---------------------------    
    # save_model(model, MODEL_FILE)
    
    return model, X_test, y_test


def save_model(model: Pipeline, file_path: str) -> None:
    """Save the trained model to a file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")
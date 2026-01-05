import pandas as pd
import joblib
from typing import Dict

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Function to Evaluate Model
def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate a trained regression model on the test dataset.

    Args:
        model (Pipeline): Trained model pipeline.
        X_test (pd.DataFrame): Test feature data.
        y_test (pd.Series): True target values for test data.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """

    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": root_mean_squared_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }

    return metrics

# Function to Load Model
def load_model(file_path: str) -> Pipeline:
    """Load a trained model from a file."""
    return joblib.load(file_path)

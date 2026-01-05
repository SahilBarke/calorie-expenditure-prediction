import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(str(file_path))


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save a pandas DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

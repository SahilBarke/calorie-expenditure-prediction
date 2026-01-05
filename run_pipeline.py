from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.train import train_model
from src.evaluate import evaluate_model
from src.settings import RAW_DATA_FILE


def run_pipeline():
    # Load data
    data = load_data(RAW_DATA_FILE)
    print("Data loaded successfully✅.")

    # Preprocess data
    data = preprocess_data(data)
    print("Data preprocessed successfully✅.")

    # Feature engineering
    data = engineer_features(data)
    print("Feature engineering completed successfully✅.")

    # Train model
    model, X_test, y_test = train_model(data)
    print("Model trained successfully✅.")

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("Model evaluation completed successfully✅.")

    # Print evaluation metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    run_pipeline()

import os
from typing import Dict

import shap
import matplotlib.pyplot as plt
import pandas as pd

from src.settings import get_feature_columns, MODEL_FILE, REPORTS_FIGS_DIR, RANDOM_SEED


def generate_shap_explanations(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str = REPORTS_FIGS_DIR,
    sample_size: int = 100,
    test_sample_size: int = 200,
) -> Dict[str, str]:
    """
    Generate SHAP explanations for the given model and save plots.

    Args:
        model: Trained machine learning model.
        X_train: Training feature set.
        X_test: Testing feature set.
        output_dir: Directory to save the SHAP plots.
        sample_size: Number of samples to use for SHAP explanation.
    Returns:
        A dictionary with paths to the saved SHAP plots.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------
    # Sample data for SHAP
    # --------------------------
    background = X_train.sample(n=sample_size, random_state=RANDOM_SEED)
    test_samples = X_test.sample(n=test_sample_size, random_state=RANDOM_SEED)

    # --------------------------
    # Initialize SHAP explainer
    # --------------------------
    explainer = shap.Explainer(model, background)

    shap_values = explainer(test_samples)

    saved_plots = {}

    # --------------------------
    # SHAP Summary Plot
    # --------------------------
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    summary_plot_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(summary_plot_path, bbox_inches="tight", dpi=200)
    plt.close()

    saved_plots["summary"] = summary_plot_path

    # --------------------------
    # SHAP Dependence Plots
    # --------------------------
    feature_columns = get_feature_columns()
    for feature in feature_columns:
        plt.figure()
        shap.plots.scatter(shap_values[:, feature], show=False)
        dependence_plot_path = os.path.join(
            output_dir, f"shap_dependence_{feature}.png"
        )
        plt.savefig(dependence_plot_path, bbox_inches="tight", dpi=200)
        plt.close()

        saved_plots[f"dependence_{feature}"] = dependence_plot_path

    # --------------------------
    # SHAP Bar Plot
    # --------------------------
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    bar_plot_path = os.path.join(output_dir, "shap_bar_plot.png")
    plt.savefig(bar_plot_path, bbox_inches="tight", dpi=200)
    plt.close()

    saved_plots["bar"] = bar_plot_path

    # --------------------------
    # SHAP Waterfall Plot for first test sample
    # --------------------------
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    waterfall_plot_path = os.path.join(output_dir, "shap_waterfall_plot_sample0.png")
    plt.savefig(waterfall_plot_path, bbox_inches="tight", dpi=200)
    plt.close()

    saved_plots["waterfall_sample0"] = waterfall_plot_path

    return saved_plots

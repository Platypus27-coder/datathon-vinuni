"""
utils.py — Logging, evaluation metrics, SHAP plots, and model I/O.
"""

import logging
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===================================================================
# LOGGING
# ===================================================================

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure root logger to write to file + console."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (utf-8)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return root


# ===================================================================
# EVALUATION
# ===================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """
    Compute MAE, RMSE, R² and print a summary.

    Returns dict with keys: mae, rmse, r2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    prefix = f"[{label}] " if label else ""
    logger = logging.getLogger(__name__)
    logger.info(f"{prefix}MAE = {mae:,.0f} | RMSE = {rmse:,.0f} | R² = {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


# ===================================================================
# PLOTTING
# ===================================================================

def plot_predictions(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Revenue Forecast",
    save_path: str | None = None,
) -> None:
    """Plot actual vs predicted revenue."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(dates, y_true, label="Actual", linewidth=0.8, alpha=0.8)
    ax.plot(dates, y_pred, label="Predicted", linewidth=0.8, alpha=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logging.getLogger(__name__).info("Plot saved: %s", save_path)
    plt.close(fig)


def plot_test_forecast(
    train_dates: pd.Series,
    train_rev: np.ndarray,
    test_dates: pd.Series,
    test_pred: np.ndarray,
    title: str = "Revenue Forecast - Test Period",
    save_path: str | None = None,
) -> None:
    """Plot train tail + test forecast."""
    fig, ax = plt.subplots(figsize=(18, 5))

    # Show last 365 days of train
    n_tail = min(365, len(train_dates))
    ax.plot(train_dates.iloc[-n_tail:], train_rev[-n_tail:],
            label="Train (last year)", linewidth=0.8, alpha=0.6, color="steelblue")
    ax.plot(test_dates, test_pred,
            label="Forecast", linewidth=1.0, color="tomato")
    ax.axvline(train_dates.iloc[-1], color="gray", ls="--", lw=0.8, label="Train/Test split")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_shap_plots(
    model,
    X: pd.DataFrame,
    save_dir: str,
    model_name: str = "model",
    max_display: int = 25,
) -> None:
    """Generate SHAP summary and bar plots for a tree-based model."""
    try:
        import shap
    except ImportError:
        logging.getLogger(__name__).warning("shap not installed, skipping SHAP plots")
        return

    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(__name__)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Summary plot (beeswarm)
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.title(f"SHAP Summary Plot - {model_name.replace('_', ' ').title()}", fontweight='bold')
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.tight_layout()
        path = os.path.join(save_dir, f"shap_{model_name}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("SHAP plot saved: %s", path)

    except Exception as e:
        logger.warning("SHAP plot failed for %s: %s", model_name, e)


# ===================================================================
# MODEL I/O
# ===================================================================

def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logging.getLogger(__name__).info("Model saved: %s", path)


def load_model(path: str):
    return joblib.load(path)

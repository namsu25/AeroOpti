"""Evaluation metrics for models and route quality."""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (avoids division by zero)."""
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1 - ss_res / ss_tot)


def route_savings_pct(baseline_value: float, optimized_value: float) -> float:
    """Percentage savings of optimized vs baseline (positive = savings)."""
    if abs(baseline_value) < 1e-12:
        return 0.0
    return float((baseline_value - optimized_value) / baseline_value * 100)

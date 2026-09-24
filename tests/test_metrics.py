"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.utils.metrics import mae, mape, r_squared, rmse, route_savings_pct


def test_rmse_and_mae_perfect_prediction():
    """Zero error when predictions exactly match targets."""
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0


def test_rmse_known_value():
    """RMSE of a simple known error pattern."""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx(3.5355339059327378)


def test_mape_ignores_near_zero_targets():
    """MAPE should skip near-zero targets to avoid division by zero."""
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 110.0])
    assert mape(y_true, y_pred) == 10.0


def test_mape_all_zero_targets_returns_zero():
    """MAPE with no valid (non-zero) targets returns 0 instead of NaN/inf."""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    assert mape(y_true, y_pred) == 0.0


def test_r_squared_perfect_fit():
    """R² of 1.0 for a perfect fit."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == 1.0


def test_r_squared_constant_target_returns_zero():
    """R² is defined as 0 (not NaN) when the target has zero variance."""
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    assert r_squared(y_true, y_pred) == 0.0


def test_route_savings_pct_positive_savings():
    """Positive savings when optimized value is lower than baseline."""
    assert route_savings_pct(100.0, 80.0) == 20.0


def test_route_savings_pct_negative_savings():
    """Negative result when optimized value is worse than baseline."""
    assert route_savings_pct(100.0, 120.0) == -20.0


def test_route_savings_pct_zero_baseline():
    """Zero baseline should return 0 rather than dividing by zero."""
    assert route_savings_pct(0.0, 50.0) == 0.0

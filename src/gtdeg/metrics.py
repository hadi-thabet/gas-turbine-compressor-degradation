"""Regression error metrics, with input checks."""
import numpy as np


def _check(y_true, y_pred):
    """Convert to float arrays and validate shape, size and finiteness."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("Inputs are empty")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Inputs contain NaN or infinite values")
    return y_true, y_pred


def mae(y_true, y_pred):
    """Mean absolute error, in the units of the target."""
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root mean squared error, in the units of the target."""
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    """Coefficient of determination: 1 - SS_residual / SS_total.

    Raises:
        ValueError: If ``y_true`` is constant, so SS_total is zero.
    """
    y_true, y_pred = _check(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        raise ValueError("R2 is undefined when y_true is constant")
    return float(1.0 - ss_res / ss_tot)


def report(y_true, y_pred):
    """Return MAE, RMSE and R2 as a dict."""
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
            "R2": r2(y_true, y_pred)}
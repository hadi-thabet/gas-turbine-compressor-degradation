"""Regression error metrics, with input checks.

All metrics take two array-likes of equal length (true and predicted
values) and return a float. They raise ``ValueError`` for bad inputs
instead of returning a misleading number.
"""
import numpy as np


def _check(y_true, y_pred):
    """Convert inputs to float arrays and validate them.

    Args:
        y_true: True values, array-like.
        y_pred: Predicted values, array-like.

    Returns:
        tuple: ``(y_true, y_pred)`` as float numpy arrays.

    Raises:
        ValueError: If the shapes differ, the inputs are empty, or either
            contains NaN or infinite values.
    """
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
    """Mean absolute error, in the units of the target.

    Args:
        y_true: True values, array-like of shape (n,).
        y_pred: Predicted values, array-like of shape (n,).

    Returns:
        float: Mean of the absolute differences.

    Raises:
        ValueError: If the inputs differ in shape, are empty, or contain
            NaN or infinite values.
    """
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root mean squared error, in the units of the target.

    Args:
        y_true: True values, array-like of shape (n,).
        y_pred: Predicted values, array-like of shape (n,).

    Returns:
        float: Square root of the mean squared difference.

    Raises:
        ValueError: If the inputs differ in shape, are empty, or contain
            NaN or infinite values.
    """
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    """Coefficient of determination: 1 - SS_residual / SS_total.

    SS_residual is the sum of squared prediction errors and SS_total is
    the sum of squared deviations of ``y_true`` from its own mean. A value
    of 1 is a perfect fit, 0 means no better than predicting the mean of
    ``y_true``, and negative values are worse than that.

    Args:
        y_true: True values, array-like of shape (n,).
        y_pred: Predicted values, array-like of shape (n,).

    Returns:
        float: The R2 score.

    Raises:
        ValueError: If the inputs differ in shape, are empty, or contain
            NaN or infinite values, or if ``y_true`` is constant so that
            SS_total is zero.
    """
    y_true, y_pred = _check(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        raise ValueError("R2 is undefined when y_true is constant")
    return float(1.0 - ss_res / ss_tot)


def report(y_true, y_pred):
    """Compute MAE, RMSE and R2 together.

    Args:
        y_true: True values, array-like of shape (n,).
        y_pred: Predicted values, array-like of shape (n,).

    Returns:
        dict: Keys ``"MAE"``, ``"RMSE"`` and ``"R2"`` with float values.

    Raises:
        ValueError: Under the same conditions as ``mae``, ``rmse`` and ``r2``.
    """
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
            "R2": r2(y_true, y_pred)}
"""Tests for metrics: hand-computed values and agreement with scikit-learn."""
import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from metrics import mae, r2, rmse


def test_known_values():
    """y = [1,2,3], p = [1,2,4]: MAE 1/3, RMSE sqrt(1/3), R2 0.5."""
    y, p = [1, 2, 3], [1, 2, 4]
    assert mae(y, p) == pytest.approx(1 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt(1 / 3))
    assert r2(y, p) == pytest.approx(0.5)


def test_mean_prediction_gives_zero_r2():
    """Predicting the mean of y itself scores R2 = 0 by definition."""
    y = np.array([0.95, 0.96, 0.99, 1.0])
    assert r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_matches_sklearn():
    """The hand-written metrics agree with scikit-learn on random data."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    p = y + rng.normal(scale=0.3, size=50)
    assert mae(y, p) == pytest.approx(mean_absolute_error(y, p))
    assert rmse(y, p) == pytest.approx(np.sqrt(mean_squared_error(y, p)))
    assert r2(y, p) == pytest.approx(r2_score(y, p))


def test_bad_inputs_raise():
    """Mismatched, empty, non-finite and constant inputs are rejected."""
    with pytest.raises(ValueError):
        mae([1, 2], [1])
    with pytest.raises(ValueError):
        mae([], [])
    with pytest.raises(ValueError):
        rmse([1, np.nan], [1, 2])
    with pytest.raises(ValueError):
        r2([1, 1, 1], [1, 1, 1])
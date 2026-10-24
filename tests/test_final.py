"""Tests for the leakage checks and the per-group error helper."""
import numpy as np
import pytest

from checks import duplicate_rows, shuffled_target_scores
from final_eval import rmse_by_group
from metrics import rmse
from models import build_model


def test_duplicate_rows_counts_exact_matches():
    """One held-out row repeats a training row; the other does not."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [5.0, 6.0]])
    split = {"train": np.array([0, 1]), "val": np.array([], dtype=int),
             "test": np.array([2, 3])}
    assert duplicate_rows(X, split) == 1


def test_shuffled_target_destroys_skill():
    """A model that is exact on real labels has no skill on shuffled labels."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = 0.97 + 0.01 * X[:, 0]
    split = {"train": np.arange(200), "val": np.arange(200, 300),
             "test": np.array([], dtype=int)}
    real = build_model("linear").fit(X[:200], y[:200])
    assert rmse(y[200:], real.predict(X[200:])) < 1e-6
    shuffled = shuffled_target_scores("linear", {}, X, y, split, seeds=(0, 1, 2))
    assert all(s > 0.005 for s in shuffled)


def test_rmse_by_group_known_values():
    """Group a has errors 1 and 0, group b has 0 and 2."""
    out = rmse_by_group([1, 1, 2, 2], [0, 1, 2, 4], ["a", "a", "b", "b"])
    assert out["a"] == pytest.approx(np.sqrt(0.5))
    assert out["b"] == pytest.approx(np.sqrt(2.0))
"""Tests for experiments: noise generation, FLOP estimates and timing."""
import numpy as np

from experiments import add_noise, flop_estimate, timed
from models import build_model


def test_add_noise_zero_is_identity():
    """Zero noise returns the input unchanged."""
    A = np.arange(12.0).reshape(4, 3)
    assert np.array_equal(add_noise(A, 0.0, np.ones(3), seed=0), A)


def test_add_noise_scale_and_reproducibility():
    """Noise std is frac * scale per column; the seed makes it repeatable."""
    A = np.zeros((20000, 2))
    scale = np.array([1.0, 10.0])
    noisy = add_noise(A, 0.1, scale, seed=3)
    assert np.allclose(noisy.std(axis=0), [0.1, 1.0], rtol=0.05)
    assert np.array_equal(noisy, add_noise(A, 0.1, scale, seed=3))
    assert not np.array_equal(noisy, add_noise(A, 0.1, scale, seed=4))


def test_flop_estimates_follow_the_formulas():
    """Estimates match the formulas for linear, SVR and tree models."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = 0.97 + 0.01 * np.sin(X[:, 0])
    lin = build_model("linear").fit(X, y)
    assert flop_estimate("linear", lin, 4, 10) == 2 * 4 * 10
    svr = build_model("svr", C=10, gamma=0.3, epsilon=0.01).fit(X, y)
    n_sv = len(svr.regressor_[-1].support_)
    assert flop_estimate("svr", svr, 4, 10) == n_sv * (3 * 4 + 2) * 10
    tree = build_model("tree", max_depth=3).fit(X, y)
    assert flop_estimate("tree", tree, 4, 10) == tree.regressor_[-1].get_depth() * 10


def test_timed_calls_the_function_each_repeat():
    """``timed`` runs the function ``repeats`` times and returns a time."""
    calls = []
    elapsed = timed(lambda: calls.append(1), repeats=5)
    assert len(calls) == 5 and elapsed >= 0
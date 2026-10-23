"""Tests for models: grids, exact recovery, SVR scaling and selection."""
import numpy as np
import pytest

from metrics import r2
from models import GRIDS, build_model, param_combinations, select_on_validation


def toy_xy(n=300, seed=0):
    """Random inputs and a smooth target on a kMc-like scale (about 0.01)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 3))
    y = 0.97 + 0.01 * np.sin(1.5 * X[:, 0]) + 0.004 * X[:, 1]
    return X, y


def test_param_combinations_counts():
    """Grid sizes are the product of the option counts; empty grid gives one."""
    assert param_combinations({}) == [{}]
    assert len(param_combinations(GRIDS["tree"])) == 10
    assert len(param_combinations(GRIDS["svr"])) == 12


def test_linear_recovers_a_linear_relationship():
    """Linear regression through the scaling pipeline is exact on linear data."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    y = 0.97 + 0.01 * X[:, 0] - 0.005 * X[:, 1]
    model = build_model("linear").fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-10)


def test_svr_is_not_constant_on_a_small_scale_target():
    """With target scaling, SVR learns a nonlinear target of spread ~0.01."""
    X, y = toy_xy()
    model = build_model("svr", C=10, gamma=0.3, epsilon=0.01).fit(X, y)
    assert r2(y, model.predict(X)) > 0.8


def test_tree_respects_max_depth():
    """The depth setting actually limits the fitted tree."""
    X, y = toy_xy()
    model = build_model("tree", max_depth=4).fit(X, y)
    assert model.regressor_[-1].get_depth() <= 4


def test_select_on_validation_prefers_the_better_setting():
    """A depth-6 tree beats a depth-1 stump on validation for a smooth target."""
    X, y = toy_xy(n=400)
    split = {"train": np.arange(300), "val": np.arange(300, 400),
             "test": np.array([], dtype=int)}
    best = select_on_validation("tree", X, y, split,
                                grid={"max_depth": [1, 6]})
    assert best["params"] == {"max_depth": 6}
    assert best["n_tried"] == 2


def test_unknown_model_raises():
    """An unknown model name is rejected."""
    with pytest.raises(ValueError):
        build_model("forest")
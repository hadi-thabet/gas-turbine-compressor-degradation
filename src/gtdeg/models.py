"""Model definitions, hyper-parameter grids and selection on validation data."""
import itertools
import time

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from metrics import report

# Settings searched for each model. SVR's epsilon is in standardised-target
# units (fractions of one standard deviation of kMc).
GRIDS = {
    "linear": {},
    "tree": {"max_depth": [4, 8, 12, 16, None], "min_samples_leaf": [1, 5]},
    "svr": {"C": [1, 10, 100], "gamma": [0.03, 0.3], "epsilon": [0.01, 0.1]},
}


def build_model(name, **params):
    """Build a model that standardises inputs and the target.

    Args:
        name: ``"linear"``, ``"tree"`` or ``"svr"``.
        **params: Settings passed to the underlying estimator.

    Returns:
        A scikit-learn regressor. Features are standardised before the
        model, and the target is standardised for fitting and converted
        back to the original units for predictions.

    Raises:
        ValueError: If ``name`` is not a known model.
    """
    if name == "linear":
        reg = LinearRegression(**params)
    elif name == "tree":
        reg = DecisionTreeRegressor(random_state=0, **params)
    elif name == "svr":
        reg = SVR(kernel="rbf", **params)
    else:
        raise ValueError(f"Unknown model: {name}")
    inner = make_pipeline(StandardScaler(), reg)
    return TransformedTargetRegressor(regressor=inner,
                                      transformer=StandardScaler())


def param_combinations(grid):
    """List every combination of the settings in ``grid``.

    Args:
        grid: Dict mapping a setting name to a list of values.

    Returns:
        list[dict]: One dict per combination; ``[{}]`` for an empty grid.
    """
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values))
            for values in itertools.product(*(grid[k] for k in keys))]


def select_on_validation(name, X, y, split, grid=None):
    """Fit every setting on the training rows and keep the best on validation.

    The test rows are never touched here.

    Args:
        name: Model name for ``build_model``.
        X: Feature array of shape (n_rows, n_features).
        y: Target array of shape (n_rows,).
        split: Dict with ``train`` and ``val`` index arrays.
        grid: Settings to search. Defaults to ``GRIDS[name]``.

    Returns:
        dict: ``params`` (best settings), ``model`` (fitted on training rows),
        ``val`` (MAE, RMSE, R2 on validation), ``fit_s`` (seconds to fit the
        best setting) and ``n_tried`` (number of settings tried).
    """
    grid = GRIDS[name] if grid is None else grid
    best, n_tried = None, 0
    for params in param_combinations(grid):
        model = build_model(name, **params)
        start = time.perf_counter()
        model.fit(X[split["train"]], y[split["train"]])
        fit_s = time.perf_counter() - start
        val = report(y[split["val"]], model.predict(X[split["val"]]))
        n_tried += 1
        if best is None or val["RMSE"] < best["val"]["RMSE"]:
            best = {"params": params, "model": model, "val": val,
                    "fit_s": fit_s}
    best["n_tried"] = n_tried
    return best
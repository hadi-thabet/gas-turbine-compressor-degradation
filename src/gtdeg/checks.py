"""Leakage and sanity checks for the evaluation.

Two checks: (1) no held-out row has a feature vector identical to a
training row, and (2) a model trained on shuffled training targets shows no
skill on validation rows, so the scores cannot come from information that
leaks through the features or the split.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import SVR_BEST, TREE_BEST
from load_data import load_data, select_features
from metrics import rmse
from models import build_model
from splits import block_split

OUT = Path("results/leakage_checks.csv")


def duplicate_rows(X, split, a="train", b="test", decimals=6):
    """Count rows of part ``b`` whose features exactly match a row of part ``a``.

    Args:
        X: Array-like of shape (n_rows, n_features).
        split: Dict of index arrays (``train``, ``val``, ``test``).
        a: Name of the reference part.
        b: Name of the part whose rows are checked.
        decimals: Rounding applied before comparing rows.

    Returns:
        int: Number of rows in ``b`` identical to some row in ``a``.
    """
    A = np.round(np.asarray(X, dtype=float), decimals)
    seen = {tuple(row) for row in A[split[a]]}
    return sum(tuple(row) in seen for row in A[split[b]])


def shuffled_target_scores(name, params, X, y, split, seeds=(0, 1, 2)):
    """Validation RMSE of a model trained on shuffled training targets.

    The training targets are permuted among the training rows, which
    destroys any real link between features and target. A sound pipeline
    should then score no better than predicting the mean.

    Args:
        name: Model name for ``build_model``.
        params: Settings for the model.
        X: Array-like of shape (n_rows, n_features).
        y: Array-like of true targets.
        split: Dict with ``train`` and ``val`` index arrays.
        seeds: Random seeds for the shuffles, one fit per seed.

    Returns:
        list[float]: Validation RMSE against the true targets, per seed.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    train, val = split["train"], split["val"]
    scores = []
    for seed in seeds:
        shuffled = np.random.default_rng(seed).permutation(y[train])
        model = build_model(name, **params).fit(X[train], shuffled)
        scores.append(rmse(y[val], model.predict(X[val])))
    return scores


def main():
    """Run both checks on the block split and save the results."""
    df = load_data()
    X = select_features(df).to_numpy()
    y = df["kMc"].to_numpy()
    sp = block_split(df)

    print("Exact duplicate feature rows (held-out part vs training):")
    for part in ("val", "test"):
        print(f"  {part}: {duplicate_rows(X, sp, 'train', part)}")

    base = rmse(y[sp["val"]], np.full(len(sp["val"]), y[sp["train"]].mean()))
    print(f"\nMean-baseline validation RMSE: {base:.5f}")
    rows = []
    for name, params in (("linear", {}), ("tree", TREE_BEST), ("svr", SVR_BEST)):
        print(f"shuffled-target fits: {name} ...", flush=True)
        scores = shuffled_target_scores(name, params, X, y, sp)
        ratio = min(scores) / base
        rows.append({"model": name, "shuffled_val_RMSE_min": min(scores),
                     "shuffled_val_RMSE_max": max(scores),
                     "baseline_val_RMSE": base,
                     "min_ratio_to_baseline": ratio, "no_skill": ratio >= 0.9})
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
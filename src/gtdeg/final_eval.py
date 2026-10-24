"""Score the final models once on the held-out test rows of the block split.

The list of models in ``CONFIGS`` was fixed before any test score was
computed for it. Models are fitted on the training rows only. Validation
scores are shown for reference; test scores were not used to choose
anything.
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write figures to file; no display window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import SVR_BEST, SVR_FAST, SVR_TUNED, TREE_BEST
from load_data import MODEL_FEATURES, load_data, select_features
from metrics import report
from models import build_model
from splits import block_split

RESULTS = Path("results")
NINE = [c for c in MODEL_FEATURES if c not in ("mf", "TIC", "T48")]
CONFIGS = [
    ("linear", "linear", {}, MODEL_FEATURES),
    ("tree", "tree", TREE_BEST, MODEL_FEATURES),
    ("svr (Step 4 setting)", "svr", SVR_BEST, MODEL_FEATURES),
    ("svr fast", "svr", SVR_FAST, MODEL_FEATURES),
    ("svr tuned", "svr", SVR_TUNED, MODEL_FEATURES),
    ("svr tuned, 9 sensors", "svr", SVR_TUNED, NINE),
]
COLUMNS = ["model", "n_features", "val_RMSE", "test_MAE", "test_RMSE",
           "test_R2", "test_RMSE_vs_baseline", "fit_s", "predict_s"]


def rmse_by_group(y_true, y_pred, groups):
    """Root mean squared error computed separately for each group.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        groups: Group label for each row (same length).

    Returns:
        pandas.Series: RMSE per group, indexed by group label.
    """
    err2 = (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2
    frame = pd.DataFrame({"group": np.asarray(groups), "err2": err2})
    return np.sqrt(frame.groupby("group")["err2"].mean())


def score_config(name, params, A, y, split):
    """Fit on the training rows, then score validation and test rows.

    Args:
        name: Model name for ``build_model``.
        params: Settings for the model.
        A: Feature array of shape (n_rows, n_features).
        y: Target array.
        split: Dict with ``train``, ``val`` and ``test`` index arrays.

    Returns:
        tuple: ``(test_predictions, row)`` where ``row`` is a dict of
        validation RMSE, test MAE/RMSE/R2 and timings in seconds.
    """
    tr, va, te = split["train"], split["val"], split["test"]
    model = build_model(name, **params)
    start = time.perf_counter()
    model.fit(A[tr], y[tr])
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    pred_test = model.predict(A[te])
    predict_s = time.perf_counter() - start
    val = report(y[va], model.predict(A[va]))
    test = report(y[te], pred_test)
    return pred_test, {"val_RMSE": val["RMSE"], "test_MAE": test["MAE"],
                       "test_RMSE": test["RMSE"], "test_R2": test["R2"],
                       "fit_s": fit_s, "predict_s": predict_s}


def main():
    """Score every model once, break the errors down, and save results."""
    df = load_data()
    X, y, sp = select_features(df), df["kMc"].to_numpy(), block_split(df)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    mean = y[tr].mean()
    base_val = report(y[va], np.full(len(va), mean))
    base_test = report(y[te], np.full(len(te), mean))
    rows = [{"model": "mean baseline", "n_features": 0,
             "val_RMSE": base_val["RMSE"], "test_MAE": base_test["MAE"],
             "test_RMSE": base_test["RMSE"], "test_R2": base_test["R2"],
             "test_RMSE_vs_baseline": 1.0, "fit_s": 0.0, "predict_s": 0.0}]
    preds = {}
    for label, name, params, feats in CONFIGS:
        print(f"{label} ...", flush=True)
        pred, row = score_config(name, params, X[feats].to_numpy(), y, sp)
        row.update(model=label, n_features=len(feats),
                   test_RMSE_vs_baseline=row["test_RMSE"] / base_test["RMSE"])
        rows.append(row)
        preds[label] = pred
    table = pd.DataFrame(rows)[COLUMNS]
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "final_test.csv", index=False)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    lp = df["lp"].round(3).to_numpy()[te]
    shown = ["linear", "tree", "svr tuned"]
    by_lp = pd.DataFrame({m: rmse_by_group(y[te], preds[m], lp) for m in shown})
    by_lp.index.name = "lp"
    by_lp.to_csv(RESULTS / "final_by_lp.csv")
    print("\nTest RMSE by lever position:")
    print(by_lp.to_string(float_format=lambda v: f"{v:.6f}"))
    kmc = df["kMc"].round(3).to_numpy()[te]
    per_level = rmse_by_group(y[te], preds["svr tuned"], kmc)
    print(f"\nsvr tuned, RMSE per held-out kMc level: "
          f"min {per_level.min():.6f}, max {per_level.max():.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y[te], preds["svr tuned"], s=3, alpha=0.4, label="svr tuned")
    axes[0].scatter(y[te], preds["tree"], s=3, alpha=0.4, label="tree")
    lims = [y[te].min(), y[te].max()]
    axes[0].plot(lims, lims, color="black", linewidth=1)
    axes[0].set_xlabel("true kMc")
    axes[0].set_ylabel("predicted kMc")
    axes[0].legend()
    for m in shown:
        axes[1].plot(by_lp.index, by_lp[m], marker="o", label=m)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("lever position")
    axes[1].set_ylabel("test RMSE of kMc")
    axes[1].legend()
    fig.tight_layout()
    path = RESULTS / "figures" / "final_test.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"\nSaved {RESULTS / 'final_test.csv'}, {RESULTS / 'final_by_lp.csv'} and {path}")


if __name__ == "__main__":
    main()
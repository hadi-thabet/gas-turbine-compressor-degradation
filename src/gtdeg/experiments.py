"""Step 5 experiments on the models: tuning, sensor ablation, noise
robustness, training-size scaling, and timing/FLOP estimates.

Usage:  python src/gtdeg/experiments.py <tuning|ablation|noise|scaling|profile>

All choices are made on the validation rows of the block split. The test
rows are not used here.
"""
import argparse
import cProfile
import io
import pickle
import pstats
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write figures to file; no display window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import MODEL_FEATURES, load_data, select_features
from metrics import report
from models import build_model, param_combinations
from splits import block_split

RESULTS = Path("results")
SVR_BEST = {"C": 100, "gamma": 0.3, "epsilon": 0.01}  # Step 4 setting, used by ablation, noise, scaling and profile
SVR_TUNED = {"C": 1000, "gamma": 0.3, "epsilon": 0.001}  # final setting, chosen from tuning.csv
SVR_FAST = {"C": 1000, "gamma": 0.3, "epsilon": 0.01}  # cheap alternative, from tuning.csv
TREE_BEST = {"max_depth": None, "min_samples_leaf": 5}  # best on block split
TUNING_GRID = {"C": [100, 1000], "gamma": [0.1, 0.3, 1.0, 3.0],
               "epsilon": [0.001, 0.01]}
NOISE_LEVELS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
SIZES = [250, 500, 1000, 2000, 4000, 7254]


def load_block():
    """Load the data and the block split.

    Returns:
        tuple: ``(X, y, split)`` with X a DataFrame of model features,
        y the kMc array and split the block-split index dict.
    """
    df = load_data()
    return select_features(df), df["kMc"].to_numpy(), block_split(df)


def save(table, name):
    """Print a results table and write it to ``results/<name>``."""
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / name, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nSaved {RESULTS / name}")


def fit_eval(name, params, X_tr, y_tr, X_ev, y_ev):
    """Fit a model and score it, timing the fit and the prediction.

    Args:
        name: Model name for ``build_model``.
        params: Settings for the model.
        X_tr, y_tr: Training inputs and targets.
        X_ev, y_ev: Evaluation inputs and targets.

    Returns:
        tuple: ``(model, dict)`` where the dict holds MAE, RMSE, R2,
        ``fit_s`` and ``predict_s`` (seconds).
    """
    model = build_model(name, **params)
    start = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(X_ev)
    predict_s = time.perf_counter() - start
    return model, {**report(y_ev, pred), "fit_s": fit_s, "predict_s": predict_s}


def add_noise(A, frac, scale, seed):
    """Add Gaussian noise with standard deviation ``frac * scale`` per column.

    Args:
        A: Array of shape (n_rows, n_features).
        frac: Noise level as a fraction of ``scale``.
        scale: Per-column scale, normally the training standard deviation.
        seed: Random seed, so the noise is reproducible.

    Returns:
        numpy.ndarray: A new array; ``A`` is not modified.
    """
    rng = np.random.default_rng(seed)
    return A + rng.normal(size=A.shape) * (frac * np.asarray(scale))


def timed(fn, repeats=5):
    """Median wall-clock time of ``fn()`` over several repeats.

    Args:
        fn: Function taking no arguments.
        repeats: Number of times to call it.

    Returns:
        float: Median time in seconds.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def flop_estimate(name, model, n_features, n_rows):
    """Estimate the arithmetic needed to predict ``n_rows`` rows.

    Assumptions: linear = one multiply-add per coefficient (2 per feature);
    SVR = for each support vector, 3d - 1 operations for the squared
    distance plus 3 more (scale, exp counted as one, multiply by the dual
    coefficient) and 1 to accumulate, giving 3d + 2; tree = one comparison
    per level (comparisons, not floating-point operations). Input
    standardisation is not counted.

    Args:
        name: ``"linear"``, ``"tree"`` or ``"svr"``.
        model: Fitted model from ``build_model``.
        n_features: Number of input features d.
        n_rows: Number of rows predicted.

    Returns:
        int: Estimated operation count.
    """
    inner = model.regressor_[-1]
    if name == "linear":
        per_row = 2 * n_features
    elif name == "tree":
        per_row = inner.get_depth()
    elif name == "svr":
        per_row = len(inner.support_) * (3 * n_features + 2)
    else:
        raise ValueError(f"Unknown model: {name}")
    return int(per_row * n_rows)


def arithmetic_intensity(name, n_features):
    """Estimated FLOPs per byte moved, per predicted row (float64 data).

    SVR assumes the support-vector data is streamed from memory for every
    row (no cache reuse), which gives a lower bound. Linear assumes only
    the input row is read. Trees are skipped (comparisons, not FLOPs).

    Args:
        name: ``"linear"``, ``"tree"`` or ``"svr"``.
        n_features: Number of input features d.

    Returns:
        float or None: FLOPs per byte, or None for trees.
    """
    d = n_features
    if name == "linear":
        return 2 * d / (8 * d)
    if name == "svr":
        return (3 * d + 2) / (8 * (d + 1))
    return None


def run_tuning():
    """Search an extended SVR grid on the block-split validation set."""
    X, y, sp = load_block()
    A = X.to_numpy()
    rows = []
    for params in param_combinations(TUNING_GRID):
        _, r = fit_eval("svr", params, A[sp["train"]], y[sp["train"]],
                        A[sp["val"]], y[sp["val"]])
        rows.append({**params, "val_RMSE": r["RMSE"], "val_MAE": r["MAE"],
                     "fit_s": r["fit_s"]})
        print(rows[-1], flush=True)
    save(pd.DataFrame(rows).sort_values("val_RMSE"), "tuning.csv")


def run_ablation():
    """Drop sensors one at a time, and three named subsets."""
    X, y, sp = load_block()
    configs = {
        "all 12 (reference)": MODEL_FEATURES,
        "drop ambiguous (mf, TIC, T48)":
            [c for c in MODEL_FEATURES if c not in ("mf", "TIC", "T48")],
        "kMc-sensitive only + lp": ["lp", "GGn", "T2", "P48", "Pexh", "GTT"],
    }
    for c in MODEL_FEATURES:
        configs[f"drop {c}"] = [f for f in MODEL_FEATURES if f != c]
    rows = []
    for label, feats in configs.items():
        Z = X[feats].to_numpy()
        _, r = fit_eval("svr", SVR_BEST, Z[sp["train"]], y[sp["train"]],
                        Z[sp["val"]], y[sp["val"]])
        rows.append({"features": label, "n_features": len(feats),
                     "val_RMSE": r["RMSE"], "val_MAE": r["MAE"],
                     "fit_s": r["fit_s"]})
        print(label, flush=True)
    table = pd.DataFrame(rows)
    table["rmse_vs_all"] = table["val_RMSE"] / table.loc[0, "val_RMSE"]
    save(table.sort_values("rmse_vs_all"), "ablation.csv")


def run_noise():
    """Add input noise and track validation error for each model."""
    X, y, sp = load_block()
    A = X.to_numpy()
    scale = A[sp["train"]].std(axis=0)
    base_rmse = report(y[sp["val"]],
                       np.full(len(sp["val"]), y[sp["train"]].mean()))["RMSE"]
    models = {"linear": {}, "tree": TREE_BEST, "svr": SVR_BEST}
    rows = []
    for name, params in models.items():
        print(name, flush=True)
        clean = build_model(name, **params).fit(A[sp["train"]], y[sp["train"]])
        for frac in NOISE_LEVELS:
            Xv = add_noise(A[sp["val"]], frac, scale, seed=1)
            noisy = build_model(name, **params).fit(
                add_noise(A[sp["train"]], frac, scale, seed=0), y[sp["train"]])
            rows.append({
                "model": name, "noise_frac": frac,
                "rmse_train_clean": report(y[sp["val"]], clean.predict(Xv))["RMSE"],
                "rmse_train_noisy": report(y[sp["val"]], noisy.predict(Xv))["RMSE"],
                "baseline_RMSE": base_rmse})
    table = pd.DataFrame(rows)
    save(table, "noise.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(NOISE_LEVELS))
    for name in models:
        t = table[table["model"] == name]
        ax.plot(xs, t["rmse_train_clean"], marker="o",
                label=f"{name} (trained clean)")
        ax.plot(xs, t["rmse_train_noisy"], marker="x", linestyle="--",
                label=f"{name} (trained noisy)")
    ax.axhline(base_rmse, color="grey", linestyle=":", label="mean baseline")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(f) for f in NOISE_LEVELS])
    ax.set_xlabel("noise std as a fraction of each feature's training std")
    ax.set_ylabel("validation RMSE of kMc")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = RESULTS / "figures" / "noise_robustness.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def run_scaling():
    """Vary the number of training rows; measure accuracy and fit time."""
    X, y, sp = load_block()
    A = X.to_numpy()
    order = np.random.default_rng(0).permutation(sp["train"])
    rows = []
    for name, params in (("svr", SVR_BEST), ("tree", TREE_BEST)):
        for n in SIZES:
            idx = order[:n]
            runs = [fit_eval(name, params, A[idx], y[idx],
                             A[sp["val"]], y[sp["val"]])[1] for _ in range(3)]
            rows.append({"model": name, "n_train": n,
                         "val_RMSE": runs[-1]["RMSE"],
                         "fit_s": statistics.median(r["fit_s"] for r in runs)})
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    save(table, "scaling.csv")
    for name in ("svr", "tree"):
        t = table[table["model"] == name]
        slope = np.polyfit(np.log(t["n_train"]), np.log(t["fit_s"]), 1)[0]
        print(f"{name}: fit time grows roughly as n^{slope:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ("svr", "tree"):
        t = table[table["model"] == name]
        axes[0].loglog(t["n_train"], t["val_RMSE"], marker="o", label=name)
        axes[1].loglog(t["n_train"], t["fit_s"], marker="o", label=name)
    axes[0].set_ylabel("validation RMSE of kMc")
    axes[1].set_ylabel("fit time (s)")
    for ax in axes:
        ax.set_xlabel("training rows")
        ax.legend()
    fig.tight_layout()
    path = RESULTS / "figures" / "scaling.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def run_profile():
    """Time each model, estimate its FLOPs and memory, and profile SVR."""
    X, y, sp = load_block()
    A = X.to_numpy()
    d = A.shape[1]
    Xtr, ytr, Xv = A[sp["train"]], y[sp["train"]], A[sp["val"]]
    rows = []
    for name, params in (("linear", {}), ("tree", TREE_BEST), ("svr", SVR_BEST)):
        model = build_model(name, **params).fit(Xtr, ytr)
        fit_s = timed(lambda: build_model(name, **params).fit(Xtr, ytr))
        predict_s = timed(lambda: model.predict(Xv))
        flops = flop_estimate(name, model, d, len(Xv))
        ai = arithmetic_intensity(name, d)
        rows.append({"model": name, "fit_s": fit_s, "predict_s": predict_s,
                     "est_ops": flops,
                     "GFLOPs_per_s": flops / predict_s / 1e9 if name != "tree" else np.nan,
                     "arith_intensity": ai if ai is not None else np.nan,
                     "model_kB": len(pickle.dumps(model)) / 1024})
        print(name, flush=True)
    save(pd.DataFrame(rows), "profile.csv")

    model = build_model("svr", **SVR_BEST)
    prof = cProfile.Profile()
    prof.enable()
    model.fit(Xtr, ytr)
    prof.disable()
    stream = io.StringIO()
    pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(8)
    (RESULTS / "profile_svr_fit.txt").write_text(stream.getvalue())
    print(stream.getvalue())


COMMANDS = {"tuning": run_tuning, "ablation": run_ablation,
            "noise": run_noise, "scaling": run_scaling,
            "profile": run_profile}


def main():
    """Run the experiment named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=COMMANDS)
    COMMANDS[parser.parse_args().experiment]()


if __name__ == "__main__":
    main()
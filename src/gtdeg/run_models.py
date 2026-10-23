"""Compare the mean baseline, linear regression, decision tree and SVR.

Settings are chosen on the validation set; the test set is scored once
per model and split.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from load_data import load_data, select_features
from metrics import report
from models import GRIDS, select_on_validation
from splits import block_split, check_split, grouped_split, random_split

OUT = Path("results/model_comparison.csv")


def main():
    """Run every model on every split and save the results table."""
    df = load_data()
    X = select_features(df).to_numpy()
    y = df["kMc"].to_numpy()

    grouped = grouped_split(df)
    n_train, n_val = len(grouped["train"]), len(grouped["val"])
    splits = {"random": random_split(df, n_train, n_val),
              "grouped": grouped,
              "block": block_split(df)}

    rows = []
    for split_name, sp in splits.items():
        check_split(df, sp)
        mean = y[sp["train"]].mean()
        val = report(y[sp["val"]], np.full(len(sp["val"]), mean))
        test = report(y[sp["test"]], np.full(len(sp["test"]), mean))
        rows.append({"split": split_name, "model": "mean baseline",
                     "params": "", "val_RMSE": val["RMSE"],
                     "test_MAE": test["MAE"], "test_RMSE": test["RMSE"],
                     "test_R2": test["R2"], "fit_s": 0.0, "predict_s": 0.0})

        for name in GRIDS:
            print(f"[{split_name}] {name} ...", flush=True)
            best = select_on_validation(name, X, y, sp)
            start = time.perf_counter()
            pred = best["model"].predict(X[sp["test"]])
            predict_s = time.perf_counter() - start
            test = report(y[sp["test"]], pred)  # test set scored once
            rows.append({"split": split_name, "model": name,
                         "params": str(best["params"]),
                         "val_RMSE": best["val"]["RMSE"],
                         "test_MAE": test["MAE"], "test_RMSE": test["RMSE"],
                         "test_R2": test["R2"], "fit_s": best["fit_s"],
                         "predict_s": predict_s})

    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
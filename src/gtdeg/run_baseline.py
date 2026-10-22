"""Mean-prediction baseline under a random split and a grouped split."""
import numpy as np

from load_data import load_data, select_features
from metrics import report
from splits import (check_split, grouped_split, nearest_train_distance,
                    random_split, shared_kmc_levels)


def main():
    """Print split sizes, leakage measures and baseline metrics."""
    df = load_data()
    X = select_features(df).to_numpy()
    y = df["kMc"].to_numpy()

    grouped = grouped_split(df)
    random = random_split(df, len(grouped["train"]), len(grouped["val"]))

    for name, sp in (("random", random), ("grouped by kMc", grouped)):
        check_split(df, sp)
        pred_value = y[sp["train"]].mean()  # the baseline: training mean
        print(f"\n=== {name} split ===")
        print(f"rows: train {len(sp['train'])}, val {len(sp['val'])}, "
              f"test {len(sp['test'])}")
        print(f"kMc values shared by train and test: "
              f"{shared_kmc_levels(df, sp)}")
        d = nearest_train_distance(X, sp, "test")
        print(f"median nearest-training-row distance (test): "
              f"{np.median(d):.4f}")
        for part in ("val", "test"):
            m = report(y[sp[part]], np.full(len(sp[part]), pred_value))
            print(f"mean baseline, {part}: MAE {m['MAE']:.5f}  "
                  f"RMSE {m['RMSE']:.5f}  R2 {m['R2']:.4f}")


if __name__ == "__main__":
    main()
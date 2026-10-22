"""Load and inspect the UCI naval propulsion plant (CBM) dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48", "T1", "T2",
            "P48", "P1", "P2", "Pexh", "TIC", "mf"]
TARGETS = ["kMc", "kMt"]
COLUMNS = FEATURES + TARGETS


def load_data(path="data/raw/UCI CBM Dataset/data.txt"):
    """Read the dataset and check that it has the expected structure.

    Args:
        path: Location of ``data.txt`` (whitespace-separated, no header).

    Returns:
        pandas.DataFrame: 16 feature columns followed by the targets
        ``kMc`` (compressor decay) and ``kMt`` (turbine decay).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the column count is wrong, a value is not numeric,
            or any value is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns, got {df.shape[1]}")
    df.columns = COLUMNS
    df = df.apply(pd.to_numeric, errors="raise")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")
    return df


def grid_report(df, decimals=3):
    """Check whether the rows form a complete grid of the three settings.

    Args:
        df: DataFrame from ``load_data``.
        decimals: Rounding applied before counting distinct values, which
            avoids floating-point noise being counted as new values.

    Returns:
        dict: Number of distinct values per setting, the expected and found
        number of (lp, kMc, kMt) combinations, duplicates, and whether the
        grid is complete.
    """
    keys = ["lp", "kMc", "kMt"]
    rounded = df[keys].round(decimals)
    levels = {k: rounded[k].nunique() for k in keys}
    expected = int(np.prod(list(levels.values())))
    found = len(rounded.drop_duplicates())
    return {"levels": levels, "expected": expected, "found": found,
            "duplicates": len(df) - found, "complete": found == expected}


def inspect(df):
    """Print summary statistics, constant columns, grid structure and
    the correlation of each feature with both decay coefficients.

    Args:
        df: DataFrame from ``load_data``.

    Returns:
        None. Results are printed to the terminal.
    """
    print(f"Rows: {len(df)}, columns: {df.shape[1]}")
    print("\nSummary statistics:")
    print(df.agg(["min", "max", "mean", "std"]).T.to_string(
        float_format=lambda x: f"{x:.4g}"))

    constant = [c for c in FEATURES if df[c].nunique() == 1]
    print(f"\nConstant feature columns: {constant}")

    g = grid_report(df)
    print("\nGrid structure (values rounded to 3 decimals):")
    for name, n in g["levels"].items():
        print(f"  {name}: {n} distinct values")
    print(f"  expected combinations {g['expected']}, found {g['found']}, "
          f"duplicates {g['duplicates']}, complete grid: {g['complete']}")

    usable = [c for c in FEATURES if c not in constant]
    corr = pd.DataFrame({t: df[usable].corrwith(df[t]) for t in TARGETS})
    corr = corr.reindex(corr["kMc"].abs().sort_values(ascending=False).index)
    print("\nPearson correlation of each feature with the targets:")
    print(corr.round(3).to_string())


if __name__ == "__main__":
    inspect(load_data())
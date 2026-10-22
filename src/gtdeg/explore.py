"""Explore why raw correlations with the decay coefficients are so weak.

Compares pooled correlations (all lever positions together) with
correlations computed within each lever position, and plots one sensor
both ways.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write figures to file; no display window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import FEATURES, load_data

# sensor columns only: drop operating-point settings and constant columns
SENSORS = [c for c in FEATURES if c not in ("lp", "v", "T1", "P1")]


def identical_columns(df, columns=FEATURES, atol=1e-9):
    """Find pairs of columns whose values are numerically identical.

    Args:
        df: DataFrame from ``load_data``.
        columns: Columns to compare.
        atol: Absolute tolerance for treating two values as equal.

    Returns:
        list[tuple]: Pairs ``(a, b)`` of identical columns.
    """
    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            if np.allclose(df[a], df[b], atol=atol):
                pairs.append((a, b))
    return pairs


def within_lp_correlation(df, target):
    """Correlate each sensor with ``target`` separately at each lever position.

    Args:
        df: DataFrame from ``load_data``.
        target: ``"kMc"`` or ``"kMt"``.

    Returns:
        pandas.DataFrame: Rows are sensors, columns are lever positions,
        values are Pearson correlations.
    """
    groups = df.groupby(df["lp"].round(3))
    return pd.DataFrame({lp: g[SENSORS].corrwith(g[target])
                         for lp, g in groups})


def compare_correlations(df):
    """Tabulate pooled versus within-lever-position correlation strength.

    Args:
        df: DataFrame from ``load_data``.

    Returns:
        pandas.DataFrame: Absolute pooled correlation and mean absolute
        within-lever-position correlation, for both targets, sorted by the
        within-position value for ``kMc``.
    """
    out = pd.DataFrame(index=SENSORS)
    for t in ("kMc", "kMt"):
        out[f"pooled_{t}"] = df[SENSORS].corrwith(df[t]).abs()
        out[f"within_lp_{t}"] = within_lp_correlation(df, t).abs().mean(axis=1)
    return out.sort_values("within_lp_kMc", ascending=False)


def plot_signal(df, sensor="T2", out="results/figures/signal_vs_noise.png"):
    """Plot a sensor against kMc for all data and for one lever position.

    Args:
        df: DataFrame from ``load_data``.
        sensor: Column to plot.
        out: Output image path.

    Returns:
        None. The figure is saved to ``out``.
    """
    lp_mid = sorted(df["lp"].round(3).unique())[4]
    sub = df[df["lp"].round(3) == lp_mid]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(df["kMc"], df[sensor], s=2, alpha=0.3)
    axes[0].set_title(f"{sensor} vs kMc, all lever positions")
    sc = axes[1].scatter(sub["kMc"], sub[sensor], c=sub["kMt"], s=6)
    axes[1].set_title(f"{sensor} vs kMc, lp = {lp_mid}")
    fig.colorbar(sc, ax=axes[1], label="kMt")
    for ax in axes:
        ax.set_xlabel("kMc")
        ax.set_ylabel(sensor)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    data = load_data()
    print("Identical columns:", identical_columns(data))
    print("\nDistinct v values at each lp (1 means a one-to-one mapping):")
    print(data.groupby(data["lp"].round(3))["v"].nunique().to_string())
    print("\nMean |correlation| with targets, pooled vs within lever position:")
    print(compare_correlations(data).round(3).to_string())
    plot_signal(data)
    print("\nSaved results/figures/signal_vs_noise.png")
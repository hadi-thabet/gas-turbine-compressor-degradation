"""Train/validation/test splits for the gas-turbine dataset.

The 11,934 rows form a complete grid of (lever position, kMc, kMt), so a
random row split puts near-identical neighbours in both training and test
sets. ``grouped_split`` holds out whole kMc values instead.
"""
import numpy as np
from scipy.spatial import cKDTree

DECIMALS = 3


def kmc_levels(df, decimals=DECIMALS):
    """Return the sorted distinct kMc values in the data.

    Args:
        df: DataFrame with a ``kMc`` column.
        decimals: Rounding applied before taking distinct values.

    Returns:
        numpy.ndarray: Sorted distinct kMc values.
    """
    return np.unique(df["kMc"].round(decimals).to_numpy())


def grouped_split(df, period=5, test_phase=2, val_phase=4):
    """Hold out whole kMc values for validation and test.

    kMc levels are numbered 0, 1, 2, ... in increasing order. A level goes
    to test if its index mod ``period`` equals ``test_phase``, to validation
    if it equals ``val_phase``, and to training otherwise. All rows sharing
    a kMc value therefore stay together.

    Args:
        df: DataFrame with a ``kMc`` column and a default integer index.
        period: Length of the repeating pattern of levels.
        test_phase: Phase assigned to the test set.
        val_phase: Phase assigned to the validation set.

    Returns:
        dict: Positional row indices under the keys ``train``, ``val``
        and ``test``.

    Raises:
        ValueError: If the phases are equal or outside ``0..period-1``.
    """
    if test_phase == val_phase or not (
            0 <= test_phase < period and 0 <= val_phase < period):
        raise ValueError("test_phase and val_phase must differ and be < period")
    levels = kmc_levels(df)
    index = np.searchsorted(levels, df["kMc"].round(DECIMALS).to_numpy())
    phase = index % period
    return {
        "train": np.flatnonzero((phase != test_phase) & (phase != val_phase)),
        "val": np.flatnonzero(phase == val_phase),
        "test": np.flatnonzero(phase == test_phase),
    }


def random_split(df, n_train, n_val, seed=0):
    """Split rows at random, ignoring the grid structure.

    Args:
        df: DataFrame to split.
        n_train: Number of training rows.
        n_val: Number of validation rows. The rest go to test.
        seed: Random seed, so the split is reproducible.

    Returns:
        dict: Sorted positional row indices under ``train``, ``val``, ``test``.

    Raises:
        ValueError: If ``n_train + n_val`` exceeds the number of rows.
    """
    if n_train + n_val > len(df):
        raise ValueError("n_train + n_val exceeds the number of rows")
    perm = np.random.default_rng(seed).permutation(len(df))
    return {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train:n_train + n_val]),
        "test": np.sort(perm[n_train + n_val:]),
    }


def check_split(df, split):
    """Confirm every row appears in exactly one of train, val and test.

    Args:
        df: The DataFrame that was split.
        split: Dict from ``grouped_split`` or ``random_split``.

    Raises:
        ValueError: If any row is missing or used more than once.
    """
    all_idx = np.concatenate([split["train"], split["val"], split["test"]])
    if len(all_idx) != len(df) or len(np.unique(all_idx)) != len(df):
        raise ValueError("Split must cover every row exactly once")


def shared_kmc_levels(df, split, a="train", b="test"):
    """Count kMc values that appear in both of two parts of a split.

    Args:
        df: DataFrame with a ``kMc`` column.
        split: Dict of index arrays.
        a: First part name.
        b: Second part name.

    Returns:
        int: Number of kMc values present in both parts.
    """
    k = df["kMc"].round(DECIMALS).to_numpy()
    return len(set(k[split[a]]) & set(k[split[b]]))


def nearest_train_distance(X, split, part="test"):
    """Distance from each row of ``part`` to its nearest training row.

    Features are standardised with the training mean and standard
    deviation, so all columns count equally. Small distances mean the
    held-out rows have near-duplicates in training (leakage).

    Args:
        X: Array-like of shape (n_rows, n_features).
        split: Dict of index arrays.
        part: Which part to measure, ``"val"`` or ``"test"``.

    Returns:
        numpy.ndarray: One distance per row of ``part``.
    """
    X = np.asarray(X, dtype=float)
    train = X[split["train"]]
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std == 0] = 1.0  # avoid dividing by zero for any constant column
    tree = cKDTree((train - mean) / std)
    dist, _ = tree.query((X[split[part]] - mean) / std)
    return dist



def block_split(df, test_block=(20, 30), val_block=(40, 50)):
    """Hold out contiguous blocks of kMc levels for test and validation.

    kMc levels are numbered 0, 1, 2, ... in increasing order. Levels in
    ``range(*test_block)`` go to test and levels in ``range(*val_block)``
    to validation. The rest go to training. Unlike ``grouped_split``,
    held-out rows have no training neighbours inside their block, so the
    model must interpolate across a gap.

    Args:
        df: DataFrame with a ``kMc`` column.
        test_block: ``(start, stop)`` level indices for test (stop excluded).
        val_block: ``(start, stop)`` level indices for validation.

    Returns:
        dict: Positional row indices under ``train``, ``val`` and ``test``.

    Raises:
        ValueError: If the two blocks overlap.
    """
    if test_block[0] < val_block[1] and val_block[0] < test_block[1]:
        raise ValueError("test_block and val_block overlap")
    levels = kmc_levels(df)
    index = np.searchsorted(levels, df["kMc"].round(DECIMALS).to_numpy())
    in_test = (index >= test_block[0]) & (index < test_block[1])
    in_val = (index >= val_block[0]) & (index < val_block[1])
    return {"train": np.flatnonzero(~in_test & ~in_val),
            "val": np.flatnonzero(in_val),
            "test": np.flatnonzero(in_test)}
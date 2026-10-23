"""Tests for splits: coverage, grouping by kMc, reproducibility, distances."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from load_data import load_data
from splits import (check_split, grouped_split, kmc_levels,
                    nearest_train_distance, random_split, shared_kmc_levels)

DATA = Path("data/raw/UCI CBM Dataset/data.txt")


def toy_grid(n_kmc=11, n_kmt=2, n_lp=3):
    """Small full grid with the same structure as the real data."""
    rows = [(lp, 0.95 + 0.001 * i, 0.975 + 0.001 * j)
            for lp, i, j in itertools.product(range(n_lp), range(n_kmc),
                                              range(n_kmt))]
    return pd.DataFrame(rows, columns=["lp", "kMc", "kMt"])


def test_grouped_split_holds_out_whole_levels():
    """No kMc value is shared between parts, and endpoints stay in train."""
    df = toy_grid()
    sp = grouped_split(df)
    check_split(df, sp)
    assert shared_kmc_levels(df, sp, "train", "test") == 0
    assert shared_kmc_levels(df, sp, "train", "val") == 0
    assert shared_kmc_levels(df, sp, "val", "test") == 0
    levels = kmc_levels(df)
    train_levels = set(df["kMc"].round(3).to_numpy()[sp["train"]])
    assert levels[0] in train_levels and levels[-1] in train_levels
    # 11 levels: 7 train, 2 val, 2 test; 6 rows per level
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (42, 12, 12)


def test_random_split_sizes_and_reproducibility():
    """Sizes are as requested; same seed repeats, different seed differs."""
    df = toy_grid()
    a = random_split(df, 42, 12, seed=1)
    b = random_split(df, 42, 12, seed=1)
    c = random_split(df, 42, 12, seed=2)
    check_split(df, a)
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (42, 12, 12)
    assert np.array_equal(a["train"], b["train"])
    assert not np.array_equal(a["train"], c["train"])


def test_random_split_leaks_levels_but_grouped_does_not():
    """The random split shares kMc values between train and test."""
    df = toy_grid()
    assert shared_kmc_levels(df, random_split(df, 42, 12), "train", "test") > 0
    assert shared_kmc_levels(df, grouped_split(df), "train", "test") == 0


def test_check_split_detects_overlap():
    """A row used twice is rejected."""
    df = toy_grid(n_kmc=1, n_kmt=2, n_lp=2)  # 4 rows
    bad = {"train": np.array([0, 1]), "val": np.array([1, 2]),
           "test": np.array([3])}
    with pytest.raises(ValueError):
        check_split(df, bad)


def test_nearest_distance_is_zero_for_a_duplicate_row():
    """A test row identical to a training row has distance zero."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    sp = {"train": np.array([0, 1]), "val": np.array([], dtype=int),
          "test": np.array([2])}
    assert nearest_train_distance(X, sp)[0] == pytest.approx(0.0)


@pytest.mark.skipif(not DATA.exists(), reason="dataset not downloaded")
def test_real_data_split_sizes():
    """The grouped split on the real data has the expected sizes."""
    df = load_data(DATA)
    sp = grouped_split(df)
    check_split(df, sp)
    assert len(kmc_levels(df)) == 51
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (7254, 2340, 2340)



def test_block_split_blocks_are_contiguous_and_separate():
    """Held-out kMc levels form contiguous blocks that share nothing."""
    from splits import block_split
    df = toy_grid()  # 11 levels, 6 rows per level
    sp = block_split(df, test_block=(3, 5), val_block=(7, 9))
    check_split(df, sp)
    k = df["kMc"].round(3).to_numpy()
    levels = kmc_levels(df)
    assert set(k[sp["test"]]) == set(levels[3:5])
    assert set(k[sp["val"]]) == set(levels[7:9])
    assert shared_kmc_levels(df, sp, "train", "test") == 0
    assert shared_kmc_levels(df, sp, "train", "val") == 0
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (42, 12, 12)
    with pytest.raises(ValueError):
        block_split(df, test_block=(3, 6), val_block=(5, 8))


@pytest.mark.skipif(not DATA.exists(), reason="dataset not downloaded")
def test_real_data_block_split_sizes():
    """The block split on the real data has the same sizes as the others."""
    from splits import block_split
    df = load_data(DATA)
    sp = block_split(df)
    check_split(df, sp)
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (7254, 2340, 2340)
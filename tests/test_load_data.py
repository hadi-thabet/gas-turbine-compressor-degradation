"""Tests for load_data: column selection and the reasons for dropping columns."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from load_data import DROPPED, FEATURES, MODEL_FEATURES, load_data, select_features

DATA = Path("data/raw/UCI CBM Dataset/data.txt")


def test_select_features_removes_dropped_columns():
    """The selected inputs contain exactly MODEL_FEATURES, none of DROPPED."""
    df = pd.DataFrame({c: [1.0, 2.0] for c in FEATURES + ["kMc", "kMt"]})
    X = select_features(df)
    assert list(X.columns) == MODEL_FEATURES
    assert not set(DROPPED) & set(X.columns)
    assert len(MODEL_FEATURES) == 12


@pytest.mark.skipif(not DATA.exists(), reason="dataset not downloaded")
def test_drop_reasons_hold_on_real_data():
    """Each reason given in DROPPED is actually true for the dataset."""
    df = load_data(DATA)
    assert df["T1"].nunique() == 1 and df["P1"].nunique() == 1
    assert np.allclose(df["Ts"], df["Tp"])
    assert (df.groupby(df["lp"].round(3))["v"].nunique() == 1).all()
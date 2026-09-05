from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qmt_quant.quantile_profiles import (
    summarize_tail_profiles,
    tail_linearity_score,
    tail_profile_for_date,
    tail_profile_observations,
)


def test_tail_profile_detects_monotonic_signal():
    symbols = [f"s{i:03d}" for i in range(100)]
    factor = pd.Series(np.arange(100, dtype=float), index=symbols)
    forward = factor / 1000.0
    row = tail_profile_for_date(factor, forward, fractions=(0.05, 0.10, 0.20))
    assert row["spread_5"] > row["spread_10"] > row["spread_20"] > 0


def test_tail_profile_observations_and_summary():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    columns = [f"s{i:03d}" for i in range(60)]
    values = np.tile(np.arange(60, dtype=float), (2, 1))
    factor = pd.DataFrame(values, index=dates, columns=columns)
    forward = factor / 1000.0
    observations = tail_profile_observations(
        factor, forward, fractions=(0.10, 0.20), min_symbols=50
    )
    summary = summarize_tail_profiles(observations, fractions=(0.10, 0.20))
    assert len(observations) == 2
    assert (summary["mean_spread"] > 0).all()
    assert tail_linearity_score(summary) > 0


def test_invalid_tail_fraction_is_rejected():
    with pytest.raises(ValueError):
        tail_profile_for_date(
            pd.Series([1.0, 2.0]),
            pd.Series([0.1, 0.2]),
            fractions=(0.5,),
            min_symbols=2,
        )

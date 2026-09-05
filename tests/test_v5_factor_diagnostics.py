import numpy as np
import pandas as pd

from qmt_quant.factor_diagnostics import (
    factor_observations,
    forward_return_panel,
    rank_ic,
    summarize_factor_observations,
    yearly_factor_summary,
)


def test_forward_return_panel_uses_future_only_as_label():
    idx = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 15.0]}, index=idx)
    forward = forward_return_panel(close, 2)
    assert np.isclose(forward.loc[idx[0], "A"], 0.20)
    assert np.isclose(forward.loc[idx[1], "A"], 15.0 / 11.0 - 1.0)
    assert pd.isna(forward.loc[idx[-1], "A"])


def test_rank_ic_detects_monotonic_predictive_ordering():
    factor = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("ABCD"))
    forward = pd.Series([-0.04, 0.01, 0.03, 0.10], index=list("ABCD"))
    assert np.isclose(rank_ic(factor, forward, min_symbols=4), 1.0)


def test_factor_observations_and_summary_capture_positive_spread():
    idx = pd.to_datetime(["2021-01-04", "2021-01-11", "2022-01-03"])
    cols = [f"S{i}" for i in range(10)]
    base = np.arange(10, dtype=float)
    factor = pd.DataFrame([base, base[::-1], base], index=idx, columns=cols)
    forward = pd.DataFrame(
        [base / 100.0, base[::-1] / 100.0, base / 50.0],
        index=idx,
        columns=cols,
    )

    obs = factor_observations(factor, forward, quantiles=5, min_symbols=10)
    summary = summarize_factor_observations(obs)
    yearly = yearly_factor_summary(obs)

    assert len(obs) == 3
    assert np.isclose(summary["mean_rank_ic"], 1.0)
    assert summary["positive_ic_ratio"] == 1.0
    assert summary["mean_top_bottom_spread"] > 0.0
    assert set(yearly["year"]) == {2021, 2022}

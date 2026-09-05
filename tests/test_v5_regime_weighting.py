from __future__ import annotations

import numpy as np
import pandas as pd

from qmt_quant.regime_weighting import (
    apply_regime_composite,
    classify_regimes,
    fit_regime_factor_weights,
    fit_regime_model,
)


def test_regime_model_is_training_only():
    dates = pd.bdate_range("2018-01-01", periods=900)
    returns = np.sin(np.arange(len(dates)) / 30.0) * 0.004
    close = pd.Series(100.0 * np.cumprod(1.0 + returns), index=dates)
    train_end = dates[600]
    model_before = fit_regime_model(
        close,
        train_start=dates[100],
        train_end=train_end,
        min_dates=100,
    )
    mutated = close.copy()
    mutated.loc[dates[650]:] *= np.linspace(1.0, 4.0, len(mutated.loc[dates[650]:]))
    model_after = fit_regime_model(
        mutated,
        train_start=dates[100],
        train_end=train_end,
        min_dates=100,
    )
    assert model_before == model_after


def test_regime_factor_weights_use_only_training_rows():
    dates = pd.bdate_range("2020-01-01", periods=120)
    regimes = pd.Series(["up_calm"] * 60 + ["down_volatile"] * 60, index=dates)
    rows = []
    for ts in dates:
        rows.extend(
            [
                {"date": ts, "factor": "a", "horizon": 20, "rank_ic": 0.05},
                {"date": ts, "factor": "b", "horizon": 20, "rank_ic": -0.03},
            ]
        )
    observations = pd.DataFrame(rows)
    train_end = dates[79]
    fitted_before = fit_regime_factor_weights(
        observations,
        regimes,
        train_start=dates[0],
        train_end=train_end,
        factors=["a", "b"],
        min_regime_dates=10,
    )
    future = observations.copy()
    future.loc[future["date"] > train_end, "rank_ic"] = 99.0
    fitted_after = fit_regime_factor_weights(
        future,
        regimes,
        train_start=dates[0],
        train_end=train_end,
        factors=["a", "b"],
        min_regime_dates=10,
    )
    assert fitted_before == fitted_after
    assert fitted_before.global_weights["a"] > 0
    assert fitted_before.global_weights["b"] < 0


def test_apply_regime_composite_uses_frozen_weights():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    columns = ["A", "B"]
    factors = {
        "a": pd.DataFrame([[1.0, -1.0], [1.0, -1.0]], index=dates, columns=columns),
        "b": pd.DataFrame([[-1.0, 1.0], [-1.0, 1.0]], index=dates, columns=columns),
    }
    observations = pd.DataFrame(
        [
            {"date": "2023-01-02", "factor": "a", "horizon": 20, "rank_ic": 0.08},
            {"date": "2023-01-02", "factor": "b", "horizon": 20, "rank_ic": -0.02},
            {"date": "2023-02-02", "factor": "a", "horizon": 20, "rank_ic": 0.08},
            {"date": "2023-02-02", "factor": "b", "horizon": 20, "rank_ic": -0.02},
        ]
    )
    training_regimes = pd.Series(
        ["up_calm", "up_calm"], index=pd.to_datetime(["2023-01-02", "2023-02-02"])
    )
    fitted = fit_regime_factor_weights(
        observations,
        training_regimes,
        train_start="2023-01-01",
        train_end="2023-12-31",
        factors=["a", "b"],
        min_regime_dates=1,
    )
    score = apply_regime_composite(
        factors,
        pd.Series(["up_calm", "unseen_regime"], index=dates),
        fitted,
    )
    assert score.loc[dates[0], "A"] > score.loc[dates[0], "B"]
    assert score.loc[dates[1], "A"] > score.loc[dates[1], "B"]

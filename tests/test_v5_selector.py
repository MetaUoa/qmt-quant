from __future__ import annotations

import pandas as pd

from qmt_quant.v5_selector import DEFAULT_SAFE_FACTORS, select_training_composite


def _observations() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2018-01-02", periods=120, freq="7D")
    for horizon in (5, 20):
        for i, date in enumerate(dates):
            values = {
                "low_volatility": 0.080 + 0.004 * ((i % 5) - 2),
                "low_downside_risk": 0.055 + 0.00275 * ((i % 5) - 2),
                "liquidity_stability": 0.052 + 0.003 * (((i * 2) % 7) - 3),
                "short_reversal": 0.035 + 0.002 * (((i * 3) % 11) - 5),
                "momentum_60_5": -0.045 + 0.002 * (((i * 5) % 13) - 6),
                "relative_strength_60_5": -0.045 + 0.002 * (((i * 5) % 13) - 6),
                "liquidity": -0.200 + 0.002 * ((i % 3) - 1),
            }
            for factor, value in values.items():
                rows.append(
                    {
                        "factor": factor,
                        "horizon": horizon,
                        "date": date,
                        "rank_ic": value,
                        "top_bottom_spread": value / 10.0,
                    }
                )
    return pd.DataFrame(rows)


def test_default_safe_set_excludes_raw_liquidity_capacity_risk():
    assert "liquidity" not in DEFAULT_SAFE_FACTORS


def test_training_selector_orients_inverse_factors_and_drops_redundancy():
    selection = select_training_composite(
        _observations(),
        train_start="2018-01-01",
        train_end="2020-12-31",
    )
    assert "liquidity" not in selection.selected_factors
    assert selection.orientations.get("momentum_60_5") == -1
    assert "low_volatility" in selection.selected_factors
    assert "low_downside_risk" not in selection.selected_factors
    assert len(selection.selected_factors) >= 2
    assert abs(sum(abs(v) for v in selection.spec.weights.values()) - 1.0) < 1e-12


def test_future_validation_rows_cannot_change_training_selection():
    observations = _observations()
    baseline = select_training_composite(
        observations,
        train_start="2018-01-01",
        train_end="2020-12-31",
    ).to_dict()
    future = observations.copy()
    future["date"] = pd.Timestamp("2025-06-30")
    future["rank_ic"] = -future["rank_ic"] * 20.0
    future["top_bottom_spread"] = -future["top_bottom_spread"] * 20.0
    combined = pd.concat([observations, future], ignore_index=True)
    after = select_training_composite(
        combined,
        train_start="2018-01-01",
        train_end="2020-12-31",
    ).to_dict()
    assert after == baseline

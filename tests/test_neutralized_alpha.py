import numpy as np
import pandas as pd
import pytest

from qmt_quant.composites import CompositeSpec
from qmt_quant.neutralized_alpha import (
    NeutralizationInputs,
    build_neutralized_composite,
    neutralize_factor_panels,
)


def _panel(values):
    return pd.DataFrame(
        [values],
        index=[pd.Timestamp("2024-01-02")],
        columns=[f"S{i:03d}.SZ" for i in range(len(values))],
        dtype=float,
    )


def test_raw_variant_is_identity_copy():
    panel = _panel(np.arange(60, dtype=float))
    out = neutralize_factor_panels(
        {"low_volatility": panel},
        variant="raw",
        inputs=NeutralizationInputs(),
    )
    pd.testing.assert_frame_equal(out["low_volatility"], panel)
    assert out["low_volatility"] is not panel


def test_missing_requested_exposure_fails_closed():
    panel = _panel(np.arange(60, dtype=float))
    with pytest.raises(RuntimeError, match="without PIT liquidity exposure"):
        neutralize_factor_panels(
            {"low_volatility": panel},
            variant="liquidity",
            inputs=NeutralizationInputs(),
        )


def test_liquidity_neutralized_composite_removes_linear_exposure():
    x = np.linspace(-2.0, 2.0, 60)
    factor = _panel(3.0 * x + np.sin(np.arange(60)))
    liquidity = _panel(x)
    spec = CompositeSpec(name="one", weights={"low_volatility": 1.0})
    score = build_neutralized_composite(
        {"low_volatility": factor},
        spec,
        variant="liquidity",
        inputs=NeutralizationInputs(liquidity_panel=liquidity),
        min_symbols=50,
        min_coverage=0.95,
    )
    corr = score.iloc[0].corr(liquidity.iloc[0])
    assert abs(float(corr)) < 1e-10

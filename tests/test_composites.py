import pandas as pd

from qmt_quant.composites import apply_composite, default_v5_candidate_specs, ic_weight_spec


def test_default_candidate_specs_are_small_and_interpretable():
    specs = default_v5_candidate_specs()
    assert [spec.name for spec in specs] == [
        "defensive_quality",
        "defensive_reversal",
        "risk_controlled",
    ]
    assert all(abs(sum(abs(v) for v in spec.weights.values()) - 1.0) < 1e-12 for spec in specs)


def test_ic_weight_spec_respects_training_orientation_and_cap():
    diagnostics = pd.DataFrame(
        {"factor": ["a", "b", "c"], "mean_rank_ic": [0.10, -0.30, 0.02]}
    )
    spec = ic_weight_spec(
        "ic",
        diagnostics,
        orientations={"a": 1, "b": -1, "c": 0},
        cap=0.20,
    )
    assert set(spec.weights) == {"a", "b"}
    assert spec.weights["a"] > 0
    assert spec.weights["b"] < 0
    assert abs(sum(abs(v) for v in spec.weights.values()) - 1.0) < 1e-12


def test_apply_composite_renormalizes_missing_factor_observations():
    idx = pd.date_range("2020-01-01", periods=2)
    a = pd.DataFrame({"X": [1.0, 1.0], "Y": [0.0, 0.0]}, index=idx)
    b = pd.DataFrame({"X": [1.0, None], "Y": [1.0, 1.0]}, index=idx)
    spec = default_v5_candidate_specs()[0]
    panels = {"low_volatility": a, "liquidity_stability": b}
    out = apply_composite(panels, spec)
    assert out.loc[idx[0], "X"] == 1.0
    assert out.loc[idx[1], "X"] == 1.0

import pandas as pd

from qmt_quant.neutralization_diagnostics import (
    aggregate_variant_quality,
    summarize_neutralization_variants,
)


def test_neutralization_diagnostics_are_date_bounded_and_factor_level():
    frame = pd.DataFrame(
        [
            {"variant": "raw", "factor": "a", "date": "2021-01-01", "rank_ic": 0.02},
            {"variant": "raw", "factor": "a", "date": "2021-02-01", "rank_ic": 0.04},
            {"variant": "industry", "factor": "a", "date": "2021-01-01", "rank_ic": 0.01},
            {"variant": "industry", "factor": "a", "date": "2021-02-01", "rank_ic": 0.01},
            {"variant": "raw", "factor": "a", "date": "2026-01-01", "rank_ic": -1.0},
        ]
    )
    summary = summarize_neutralization_variants(frame, start="2021-01-01", end="2021-12-31")
    raw = summary.loc[summary["variant"] == "raw"].iloc[0]
    assert raw["dates"] == 2
    assert abs(raw["mean_rank_ic"] - 0.03) < 1e-12


def test_variant_quality_aggregates_without_performance_selection():
    frame = pd.DataFrame(
        [
            {"variant": "raw", "factor": "a", "date": "2021-01-01", "rank_ic": 0.02},
            {"variant": "raw", "factor": "b", "date": "2021-01-01", "rank_ic": -0.03},
            {"variant": "industry", "factor": "a", "date": "2021-01-01", "rank_ic": 0.01},
            {"variant": "industry", "factor": "b", "date": "2021-01-01", "rank_ic": 0.01},
        ]
    )
    summary = summarize_neutralization_variants(frame, start="2021-01-01", end="2021-12-31")
    quality = aggregate_variant_quality(summary)
    assert set(quality["variant"]) == {"raw", "industry"}
    assert set(quality["factors"]) == {2}

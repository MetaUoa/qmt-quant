from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import run_v5_c9_neutralization_diagnostics as c9


def test_c9_rejects_holdout_dates() -> None:
    c9._assert_pre_2026_only(["--end", "20251231"])
    with pytest.raises(RuntimeError, match="holdout remains blinded"):
        c9._assert_pre_2026_only(["--end", "20260101"])


def test_c9_fold_safe_diagnostics_do_not_change_selection(tmp_path: Path) -> None:
    c9._CAPTURED.clear()
    dates = pd.to_datetime(["2018-01-02", "2019-01-02", "2020-01-02"])
    for variant in c9.c1.VARIANTS:
        rows = []
        for factor in c9.c1.CORE_ALPHA_FACTORS:
            for i, date in enumerate(dates):
                rows.append({"variant": variant, "factor": factor, "date": date, "rank_ic": 0.01 * (i + 1)})
        c9._CAPTURED.append(pd.DataFrame(rows))

    choices = [
        {
            "outer_validation_year": 2021,
            "inner": {
                variant: {"selection": {"train_start": "2018-01-01", "train_end": "2019-12-31"}}
                for variant in c9.c1.VARIANTS
            },
            "outer_selection": {"train_start": "2018-01-01", "train_end": "2020-12-31"},
        }
    ]
    (tmp_path / "nested_choices.json").write_text(__import__("json").dumps(choices), encoding="utf-8")

    manifest = c9._build_fold_safe_diagnostics(tmp_path)
    assert manifest["selection_changed"] is False
    assert manifest["candidate_changed"] is False
    assert manifest["holdout_unlocked"] is False
    assert manifest["pre_2026_only"] is True
    assert manifest["core_factors_only"] == list(c9.c1.CORE_ALPHA_FACTORS)
    assert (tmp_path / "c9_neutralization_factor_summary.csv").exists()
    assert (tmp_path / "c9_neutralization_variant_quality.csv").exists()

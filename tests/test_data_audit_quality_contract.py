from __future__ import annotations

from pathlib import Path

import pandas as pd

import run_data_audit


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_data_audit.py"


def test_data_audit_keeps_original_coverage_thresholds_and_adds_quality_gate():
    args = run_data_audit.validate_args(run_data_audit.parse_args([]))
    assert args.min_symbol_coverage == 0.98
    assert args.min_session_coverage == 0.97

    text = SCRIPT.read_text(encoding="utf-8")
    assert "audit_bar_collection" in text
    assert "audit_limit_reference_table" in text
    assert "adjusted_quality.passed" in text
    assert "raw_quality.passed" in text
    assert "limit_quality.passed" in text
    assert "market_data_quality.csv" in text


def test_legacy_none_sentinel_is_reported_as_query_provenance(tmp_path):
    frame = pd.DataFrame(
        {
            "trade_date": ["20250102", "20250102", "20250103"],
            "ts_code": ["__NONE__", "000001.SZ", "__NONE__"],
        }
    )
    frame.to_parquet(tmp_path / "stock_st.parquet", index=False)
    summary = run_data_audit._legacy_sentinel_summary(tmp_path, "stock_st")
    assert summary == {"rows": 2, "dates": 2}


def test_legacy_none_sentinel_is_not_described_as_a_real_symbol():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "legacy_st_empty_response_sentinel" in text
    assert "must never be treated as a real symbol" in text

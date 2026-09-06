from __future__ import annotations

import pandas as pd

from qmt_quant.market_data_quality import (
    audit_bar_collection,
    audit_limit_reference_table,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.index = pd.date_range("2025-01-02", periods=len(frame), freq="D")
    return frame


def test_adjusted_ohlc_quality_passes_valid_rows():
    bars = {
        "000001.SZ": _frame(
            [
                {
                    "open": 10.0,
                    "high": 10.6,
                    "low": 9.8,
                    "close": 10.2,
                    "volume": 1000,
                    "amount": 10200,
                }
            ]
        )
    }
    quality, detail = audit_bar_collection(bars, label="adjusted", require_ohlc=True)
    assert quality.passed is True
    assert quality.rows_checked == 1
    assert bool(detail.iloc[0]["passed"]) is True


def test_adjusted_ohlc_quality_rejects_inverted_and_nonpositive_rows():
    bars = {
        "000001.SZ": _frame(
            [
                {
                    "open": 10.0,
                    "high": 9.5,
                    "low": 10.5,
                    "close": 0.0,
                    "volume": -1,
                    "amount": -1,
                }
            ]
        )
    }
    quality, _ = audit_bar_collection(bars, label="adjusted", require_ohlc=True)
    assert quality.passed is False
    assert quality.nonpositive_price_rows == 1
    assert quality.invalid_ohlc_rows == 1
    assert quality.negative_volume_rows == 1
    assert quality.negative_amount_rows == 1


def test_known_suspension_placeholder_is_excluded_from_quality_gate():
    day = pd.Timestamp("2025-01-02")
    frame = _frame(
        [
            {
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "volume": 0,
                "amount": 0,
            }
        ]
    )
    frame.index = pd.DatetimeIndex([day])
    quality, detail = audit_bar_collection(
        {"000001.SZ": frame},
        label="adjusted",
        require_ohlc=True,
        excluded_dates_by_code={"000001.SZ": {day}},
    )
    assert quality.passed is True
    assert quality.rows_checked == 0
    assert detail.empty


def test_raw_reference_requires_positive_open_close_and_preclose():
    raw = {
        "000001.SZ": _frame(
            [{"open": 10.0, "close": 10.2, "preClose": 0.0}]
        )
    }
    quality, _ = audit_bar_collection(raw, label="raw", require_ohlc=False)
    assert quality.passed is False
    assert quality.nonpositive_price_rows == 1


def test_price_limit_quality_rejects_missing_nonpositive_and_inverted_rows():
    limits = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "pre_close": 10.0,
                "up_limit": 9.0,
                "down_limit": 11.0,
            },
            {
                "ts_code": "000002.SZ",
                "pre_close": 0.0,
                "up_limit": 11.0,
                "down_limit": 9.0,
            },
            {
                "ts_code": "000003.SZ",
                "pre_close": None,
                "up_limit": 11.0,
                "down_limit": 9.0,
            },
        ]
    )
    quality = audit_limit_reference_table(limits)
    assert quality.passed is False
    assert quality.inverted_rows >= 1
    assert quality.nonpositive_rows == 1
    assert quality.missing_rows == 1


def test_price_limit_legacy_none_sentinel_is_provenance_not_a_price_row():
    limits = pd.DataFrame(
        [
            {
                "ts_code": "__NONE__",
                "pre_close": None,
                "up_limit": None,
                "down_limit": None,
            },
            {
                "ts_code": "000001.SZ",
                "pre_close": 10.0,
                "up_limit": 11.0,
                "down_limit": 9.0,
            },
        ]
    )
    quality = audit_limit_reference_table(limits)
    assert quality.passed is True
    assert quality.rows_checked == 1

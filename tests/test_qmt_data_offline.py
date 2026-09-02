from __future__ import annotations

import pandas as pd
import pytest

from qmt_quant import qmt_data


class FakeXtData:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.calls = []

    def get_market_data_ex(self, **kwargs):
        self.calls.append(kwargs)
        return {code: self.frames.get(code) for code in kwargs["stock_list"]}


def test_normalize_frame_accepts_yyyymmdd_index_and_deduplicates():
    raw = pd.DataFrame(
        {"close": [10, 11, 12]},
        index=["20200102", "20200102", "20200103"],
    )
    out = qmt_data._normalize_frame(raw, ["close", "open"])
    assert list(out.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert out.loc[pd.Timestamp("2020-01-02"), "close"] == 11
    assert out["open"].isna().all()


def test_market_cache_prevents_second_qmt_fetch(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    raw = pd.DataFrame(
        {
            "open": [10.0, 10.2],
            "close": [10.1, 10.3],
        },
        index=["20200102", "20200103"],
    )
    fake = FakeXtData({"AAA.SZ": raw})
    monkeypatch.setattr(qmt_data, "_xtdata", lambda: fake)

    first = qmt_data.load_market_fields(
        ["AAA.SZ"],
        "20200101",
        "20200131",
        ["open", "close"],
        dividend_type="front",
        cache_dir=tmp_path,
    )
    assert "AAA.SZ" in first
    assert len(fake.calls) == 1
    assert fake.calls[0]["dividend_type"] == "front"

    def should_not_be_called():
        raise AssertionError("QMT should not be called when the requested range is cached")

    monkeypatch.setattr(qmt_data, "_xtdata", should_not_be_called)
    second = qmt_data.load_market_fields(
        ["AAA.SZ"],
        "20200101",
        "20200131",
        ["open", "close"],
        dividend_type="front",
        cache_dir=tmp_path,
    )
    pd.testing.assert_frame_equal(first["AAA.SZ"], second["AAA.SZ"], check_freq=False)


def test_coverage_report_marks_missing_symbols():
    idx = pd.bdate_range("2020-01-01", periods=2)
    bars = {"AAA.SZ": pd.DataFrame({"close": [10.0, 10.2]}, index=idx)}
    report = qmt_data.coverage_report(["AAA.SZ", "BBB.SH"], bars).set_index("code")
    assert bool(report.loc["AAA.SZ", "loaded"])
    assert not bool(report.loc["BBB.SH", "loaded"])
    assert int(report.loc["AAA.SZ", "rows"]) == 2
    assert int(report.loc["BBB.SH", "rows"]) == 0

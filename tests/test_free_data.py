from __future__ import annotations

import pandas as pd
import pytest

from qmt_quant.free_data import (
    baostock_to_ts_code,
    build_reference_tables,
    fetch_history,
    fetch_stock_basic,
    fetch_trade_calendar,
    prepare_baostock_cache,
    price_limit_rate,
    ts_code_to_baostock,
    verify_with_akshare,
)
from qmt_quant.qmt_data import load_daily_bars, load_limit_reference_bars
from qmt_quant.reference_data import ReferenceData


class FakeResult:
    def __init__(self, fields, rows, error_code="0", error_msg="success"):
        self.fields = list(fields)
        self._rows = [list(row) for row in rows]
        self._i = -1
        self.error_code = error_code
        self.error_msg = error_msg

    def next(self):
        self._i += 1
        return self._i < len(self._rows)

    def get_row_data(self):
        return self._rows[self._i]


class FakeBaoStock:
    def query_stock_basic(self):
        fields = ["code", "code_name", "ipoDate", "outDate", "type", "status"]
        rows = [
            ["sz.000001", "Ping An Bank", "1991-04-03", "", "1", "1"],
            ["sh.900901", "B Share", "1990-01-01", "", "1", "1"],
            ["sh.000905", "CSI 500", "2007-01-15", "", "2", "1"],
        ]
        return FakeResult(fields, rows)

    def query_trade_dates(self, start_date, end_date):
        fields = ["calendar_date", "is_trading_day"]
        rows = [
            ["2020-01-02", "1"],
            ["2020-01-03", "1"],
            ["2020-01-04", "0"],
        ]
        return FakeResult(fields, rows)

    def query_history_k_data_plus(
        self, code, fields, start_date, end_date, frequency, adjustflag
    ):
        cols = fields.split(",")
        if code == "sh.000905":
            source = {
                "date": ["2020-01-02", "2020-01-03"],
                "code": [code, code],
                "open": ["5000", "5010"],
                "high": ["5020", "5030"],
                "low": ["4980", "5000"],
                "close": ["5010", "5020"],
                "preclose": ["4990", "5010"],
                "volume": ["100", "120"],
                "amount": ["1000000", "1200000"],
                "pctChg": ["0.4", "0.2"],
            }
        else:
            factor = 0.5 if adjustflag == "2" else 1.0
            source = {
                "date": ["2020-01-02", "2020-01-03"],
                "code": [code, code],
                "open": [str(10 * factor), str(10.5 * factor)],
                "high": [str(10.5 * factor), str(11 * factor)],
                "low": [str(9.8 * factor), str(10.4 * factor)],
                "close": [str(10.2 * factor), str(10.8 * factor)],
                "preclose": [str(10 * factor), str(10.2 * factor)],
                "volume": ["100000", "120000"],
                "amount": ["1000000", "1200000"],
                "adjustflag": [adjustflag, adjustflag],
                "turn": ["1.0", "1.2"],
                "tradestatus": ["1", "0"],
                "pctChg": ["2.0", "5.8"],
                "isST": ["0", "1"],
            }
        rows = list(zip(*(source[col] for col in cols)))
        return FakeResult(cols, rows)


class FakeAkShare:
    def stock_zh_a_hist(self, symbol, period, start_date, end_date, adjust):
        return pd.DataFrame(
            {
                "日期": ["2020-01-02", "2020-01-03"],
                "股票代码": [symbol, symbol],
                "开盘": [10.0, 10.5],
                "收盘": [10.2, 10.8],
                "最高": [10.5, 11.0],
                "最低": [9.8, 10.4],
                "成交量": [1000, 1200],
                "成交额": [1000000, 1200000],
                "振幅": [1.0, 1.0],
                "涨跌幅": [2.0, 5.8],
                "涨跌额": [0.2, 0.6],
                "换手率": [1.0, 1.2],
            }
        )


def test_code_mapping_and_price_limit_rules():
    assert ts_code_to_baostock("600000.SH") == "sh.600000"
    assert ts_code_to_baostock("000001.SZ") == "sz.000001"
    assert baostock_to_ts_code("sh.600000") == "600000.SH"
    assert price_limit_rate("600000.SH", "20200102", False) == pytest.approx(0.10)
    assert price_limit_rate("600000.SH", "20200102", True) == pytest.approx(0.05)
    assert price_limit_rate("300001.SZ", "20200821", False) == pytest.approx(0.10)
    assert price_limit_rate("300001.SZ", "20200824", False) == pytest.approx(0.20)
    assert price_limit_rate("688001.SH", "20200102", False) == pytest.approx(0.20)


def test_baostock_schema_normalization():
    api = FakeBaoStock()
    basic = fetch_stock_basic(api)
    assert list(basic["ts_code"]) == ["000001.SZ"]
    assert basic.iloc[0]["exchange"] == "SZSE"

    cal = fetch_trade_calendar(api, "20200101", "20200104")
    assert int(cal["is_open"].sum()) == 2
    assert cal.iloc[0]["cal_date"] == "20200102"

    raw = fetch_history(api, "000001.SZ", "20200101", "20200104", adjusted=False, include_meta=True)
    assert set(["preClose", "suspendFlag", "isST", "tradestatus"]).issubset(raw.columns)
    assert raw.loc[1, "suspendFlag"] == pytest.approx(1.0)

    front = fetch_history(api, "000001.SZ", "20200101", "20200104", adjusted=True)
    assert front.loc[0, "close"] == pytest.approx(5.1)


def test_reference_tables_have_st_sentinel_suspend_and_limits():
    api = FakeBaoStock()
    basic = fetch_stock_basic(api)
    cal = fetch_trade_calendar(api, "20200101", "20200104")
    raw = fetch_history(api, "000001.SZ", "20200101", "20200104", adjusted=False, include_meta=True)
    st, limits, susp = build_reference_tables(basic, cal, {"000001.SZ": raw})

    day1 = st.loc[st["trade_date"].eq("20200102")]
    day2 = st.loc[st["trade_date"].eq("20200103")]
    assert "__NONE__" in set(day1["ts_code"])
    assert "000001.SZ" in set(day2["ts_code"])
    assert "000001.SZ" in set(susp["ts_code"])

    row = limits.loc[
        limits["trade_date"].eq("20200103") & limits["ts_code"].eq("000001.SZ")
    ].iloc[0]
    assert row["up_limit"] == pytest.approx(10.71)
    assert row["down_limit"] == pytest.approx(9.69)


def test_prepare_cache_is_compatible_with_existing_qmt_loader(tmp_path, monkeypatch):
    api = FakeBaoStock()
    ref_dir = tmp_path / "reference"
    cache_dir = tmp_path / "qmt_bars"
    manifest = prepare_baostock_cache(
        ref_dir,
        cache_dir,
        "20200101",
        "20200104",
        benchmark="000905.SH",
        sleep_seconds=0,
        api=api,
    )
    assert manifest["strict_ready"] is True

    ref = ReferenceData.from_dir(ref_dir)
    assert ref.codes_ever_active("20200101", "20200104") == ["000001.SZ"]
    assert pd.Timestamp("2020-01-02") in ref.st_dates
    assert pd.Timestamp("2020-01-03") in ref.limit_dates

    monkeypatch.setenv("QMT_QUANT_CACHE_ONLY", "1")
    bars = load_daily_bars(
        ["000001.SZ", "000905.SH"],
        "20200101",
        "20200104",
        cache_dir=cache_dir / "front_20200101_20200104",
    )
    raw = load_limit_reference_bars(
        ["000001.SZ"],
        "20200101",
        "20200104",
        cache_dir=cache_dir / "none_limits_20200101_20200104",
    )
    assert set(bars) == {"000001.SZ", "000905.SH"}
    assert set(raw) == {"000001.SZ"}
    assert bars["000001.SZ"].iloc[0]["close"] == pytest.approx(5.1)


def test_akshare_crosscheck_uses_unadjusted_close(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "open": [10.0, 10.5],
            "high": [10.5, 11.0],
            "low": [9.8, 10.4],
            "close": [10.2, 10.8],
            "volume": [100000, 120000],
            "amount": [1000000, 1200000],
            "preClose": [10.0, 10.2],
            "suspendFlag": [0.0, 0.0],
        }
    ).to_parquet(raw_dir / "000001.SZ.parquet", index=False)

    result = verify_with_akshare(
        ["000001.SZ"],
        "20200101",
        "20200104",
        raw_dir,
        sample_size=1,
        api=FakeAkShare(),
    )
    assert result.iloc[0]["status"] == "pass"
    assert result.iloc[0]["overlap_days"] == 2


def test_cache_only_never_falls_back_to_qmt(tmp_path, monkeypatch):
    from qmt_quant import qmt_data

    monkeypatch.setenv("QMT_QUANT_CACHE_ONLY", "1")

    def fail():
        raise AssertionError("xtquant must not be touched in cache-only mode")

    monkeypatch.setattr(qmt_data, "_xtdata", fail)
    result = qmt_data.load_daily_bars(
        ["000001.SZ"],
        "20200101",
        "20200104",
        cache_dir=tmp_path,
    )
    assert result == {}

    with pytest.raises(RuntimeError, match="CACHE_ONLY"):
        qmt_data.download_daily_history(["000001.SZ"], "20200101", "20200104")

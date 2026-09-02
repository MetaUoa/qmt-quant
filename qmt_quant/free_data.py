from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import time
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .qmt_data import FIELDS


STOCK_HISTORY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,isST"
)
INDEX_HISTORY_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"


def _import_baostock():
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError(
            "baostock is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return bs


def _import_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError(
            "akshare is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return ak


def _result_frame(result) -> pd.DataFrame:
    error_code = str(getattr(result, "error_code", "0"))
    if error_code != "0":
        raise RuntimeError(
            f"BaoStock request failed: {error_code} "
            f"{getattr(result, 'error_msg', '')}".strip()
        )
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    fields = list(getattr(result, "fields", []))
    return pd.DataFrame(rows, columns=fields)


@contextmanager
def baostock_session(api=None) -> Iterator[object]:
    bs = api or _import_baostock()
    login = bs.login()
    if str(getattr(login, "error_code", "0")) != "0":
        raise RuntimeError(
            f"BaoStock login failed: {getattr(login, 'error_code', '')} "
            f"{getattr(login, 'error_msg', '')}".strip()
        )
    try:
        yield bs
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def ts_code_to_baostock(code: str) -> str:
    text = str(code).strip()
    if "." not in text:
        raise ValueError(f"Expected Tushare/QMT style code, got: {code}")
    symbol, exchange = text.split(".", 1)
    exchange = exchange.upper()
    if exchange == "SH":
        return f"sh.{symbol}"
    if exchange == "SZ":
        return f"sz.{symbol}"
    raise ValueError(f"Unsupported exchange for free A-share data: {code}")


def baostock_to_ts_code(code: str) -> str:
    text = str(code).strip().lower()
    if text.startswith("sh."):
        return f"{text[3:]}.SH"
    if text.startswith("sz."):
        return f"{text[3:]}.SZ"
    raise ValueError(f"Unsupported BaoStock code: {code}")


def _is_a_share_baostock(code: str) -> bool:
    text = str(code).strip().lower()
    if text.startswith("sh."):
        symbol = text[3:]
        return symbol.startswith(("600", "601", "603", "605", "688"))
    if text.startswith("sz."):
        symbol = text[3:]
        return symbol.startswith(("000", "001", "002", "003", "300", "301"))
    return False


def fetch_stock_basic(api) -> pd.DataFrame:
    frame = _result_frame(api.query_stock_basic())
    if frame.empty:
        raise RuntimeError("BaoStock query_stock_basic returned no rows.")
    frame = frame.loc[
        frame.get("type", pd.Series(index=frame.index, dtype=str)).astype(str).eq("1")
        & frame["code"].map(_is_a_share_baostock)
    ].copy()
    frame["ts_code"] = frame["code"].map(baostock_to_ts_code)
    frame["symbol"] = frame["ts_code"].str.split(".").str[0]
    frame["name"] = frame.get("code_name", "")
    frame["exchange"] = frame["ts_code"].str.split(".").str[1].map(
        {"SH": "SSE", "SZ": "SZSE"}
    )
    frame["market"] = np.where(
        frame["symbol"].str.startswith(("688", "300", "301")), "Growth", "Main"
    )
    frame["list_status"] = np.where(frame.get("status", "1").astype(str).eq("1"), "L", "D")
    frame["list_date"] = (
        frame.get("ipoDate", "")
        .astype(str)
        .str.replace("-", "", regex=False)
        .replace({"": pd.NA, "nan": pd.NA})
    )
    frame["delist_date"] = (
        frame.get("outDate", "")
        .astype(str)
        .str.replace("-", "", regex=False)
        .replace({"": pd.NA, "nan": pd.NA})
    )
    columns = [
        "ts_code",
        "symbol",
        "name",
        "market",
        "exchange",
        "list_status",
        "list_date",
        "delist_date",
    ]
    return frame[columns].drop_duplicates("ts_code", keep="last").sort_values("ts_code").reset_index(drop=True)


def fetch_trade_calendar(api, start: str, end: str) -> pd.DataFrame:
    start_date = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(end).strftime("%Y-%m-%d")
    frame = _result_frame(api.query_trade_dates(start_date=start_date, end_date=end_date))
    if frame.empty:
        raise RuntimeError("BaoStock query_trade_dates returned no rows.")
    if "calendar_date" not in frame or "is_trading_day" not in frame:
        raise ValueError("Unexpected BaoStock trade calendar schema.")
    out = pd.DataFrame()
    out["cal_date"] = pd.to_datetime(frame["calendar_date"], errors="coerce").dt.strftime("%Y%m%d")
    out["is_open"] = pd.to_numeric(frame["is_trading_day"], errors="coerce").fillna(0).astype(int)
    out["exchange"] = "SSE"
    out = out.dropna(subset=["cal_date"]).sort_values("cal_date").reset_index(drop=True)
    pretrade = pd.Series(pd.NA, index=out.index, dtype="object")
    open_days = out.loc[out["is_open"].eq(1), "cal_date"]
    previous = None
    for idx, value in open_days.items():
        pretrade.at[idx] = previous
        previous = value
    out["pretrade_date"] = pretrade
    return out[["exchange", "cal_date", "is_open", "pretrade_date"]]


def _normalize_history(frame: pd.DataFrame, include_meta: bool) -> pd.DataFrame:
    if frame is None or frame.empty:
        columns = ["date"] + FIELDS + (["isST", "tradestatus"] if include_meta else [])
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    rename = {"preclose": "preClose"}
    out = out.rename(columns=rename)
    numeric = ["open", "high", "low", "close", "volume", "amount", "preClose"]
    for col in numeric:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "tradestatus" in out:
        trade_status = pd.to_numeric(out["tradestatus"], errors="coerce").fillna(0)
        out["suspendFlag"] = (1 - trade_status.clip(0, 1)).astype(float)
    else:
        out["suspendFlag"] = 0.0
    if "isST" not in out:
        out["isST"] = "0"
    if "tradestatus" not in out:
        out["tradestatus"] = "1"
    columns = ["date"] + FIELDS
    if include_meta:
        columns += ["isST", "tradestatus"]
    return out[columns].reset_index(drop=True)


def fetch_history(
    api,
    code: str,
    start: str,
    end: str,
    *,
    adjusted: bool,
    include_meta: bool = False,
) -> pd.DataFrame:
    bs_code = ts_code_to_baostock(code)
    start_date = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(end).strftime("%Y-%m-%d")
    is_index = code in {"000905.SH", "000300.SH", "000001.SH", "399001.SZ", "399006.SZ"}
    fields = INDEX_HISTORY_FIELDS if is_index else STOCK_HISTORY_FIELDS
    result = api.query_history_k_data_plus(
        bs_code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3" if is_index else ("2" if adjusted else "3"),
    )
    return _normalize_history(_result_frame(result), include_meta=include_meta and not is_index)


def _round_price(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def price_limit_rate(code: str, trade_date, is_st: bool) -> float:
    symbol = str(code).split(".")[0]
    ts = pd.Timestamp(trade_date)
    if symbol.startswith("688"):
        return 0.20
    if symbol.startswith(("300", "301")):
        return 0.20 if ts >= pd.Timestamp("2020-08-24") else 0.10
    return 0.05 if is_st else 0.10


def build_reference_tables(
    stock_basic: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    raw_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    st_rows: list[dict] = []
    limit_rows: list[dict] = []
    suspension_rows: list[dict] = []

    open_dates = set(
        pd.to_datetime(
            trade_calendar.loc[trade_calendar["is_open"].eq(1), "cal_date"],
            format="%Y%m%d",
            errors="coerce",
        ).dropna()
    )
    st_dates_with_rows: set[pd.Timestamp] = set()
    limit_dates_seen: set[pd.Timestamp] = set()

    for code, frame in raw_frames.items():
        if frame is None or frame.empty:
            continue
        work = frame.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"])
        for row in work.itertuples(index=False):
            day = pd.Timestamp(row.date).normalize()
            if day not in open_dates:
                continue
            is_st = str(getattr(row, "isST", "0")).strip() in {"1", "True", "true"}
            trade_status = pd.to_numeric(pd.Series([getattr(row, "tradestatus", "1")]), errors="coerce").iloc[0]
            pre_close = float(getattr(row, "preClose", np.nan))
            if is_st:
                st_rows.append({"trade_date": day.strftime("%Y%m%d"), "ts_code": code, "name": "", "type": "", "type_name": ""})
                st_dates_with_rows.add(day)
            if not np.isfinite(trade_status) or int(trade_status) != 1:
                suspension_rows.append(
                    {
                        "ts_code": code,
                        "trade_date": day.strftime("%Y%m%d"),
                        "suspend_type": "S",
                        "suspend_timing": "BaoStock tradestatus=0",
                    }
                )
            if np.isfinite(pre_close) and pre_close > 0:
                rate = price_limit_rate(code, day, is_st)
                limit_rows.append(
                    {
                        "trade_date": day.strftime("%Y%m%d"),
                        "ts_code": code,
                        "pre_close": pre_close,
                        "up_limit": _round_price(pre_close * (1.0 + rate)),
                        "down_limit": _round_price(pre_close * (1.0 - rate)),
                    }
                )
                limit_dates_seen.add(day)

    for day in sorted(open_dates):
        day_s = pd.Timestamp(day).strftime("%Y%m%d")
        if day not in st_dates_with_rows:
            st_rows.append({"trade_date": day_s, "ts_code": "__NONE__", "name": "", "type": "", "type_name": ""})
        if day not in limit_dates_seen:
            limit_rows.append(
                {
                    "trade_date": day_s,
                    "ts_code": "__NONE__",
                    "pre_close": np.nan,
                    "up_limit": np.nan,
                    "down_limit": np.nan,
                }
            )

    st = pd.DataFrame(st_rows, columns=["trade_date", "ts_code", "name", "type", "type_name"])
    limits = pd.DataFrame(limit_rows, columns=["trade_date", "ts_code", "pre_close", "up_limit", "down_limit"])
    susp = pd.DataFrame(
        suspension_rows,
        columns=["ts_code", "trade_date", "suspend_type", "suspend_timing"],
    )
    return st, limits, susp


def _write_qmt_cache(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["date"] + FIELDS
    out = frame.copy()
    for col in cols:
        if col not in out:
            out[col] = np.nan
    out[cols].to_parquet(path, index=False)


def prepare_baostock_cache(
    output_reference: str | Path,
    bar_cache_dir: str | Path,
    start: str,
    end: str,
    *,
    benchmark: str = "000905.SH",
    max_stocks: int = 0,
    sleep_seconds: float = 0.02,
    refresh: bool = False,
    api=None,
) -> dict:
    ref_root = Path(output_reference)
    cache_root = Path(bar_cache_dir)
    front_root = cache_root / f"front_{start}_{end}"
    raw_root = cache_root / f"none_limits_{start}_{end}"
    ref_root.mkdir(parents=True, exist_ok=True)
    front_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    own_session = api is None
    context = baostock_session(api) if own_session else _already_open(api)
    with context as bs:
        basic = fetch_stock_basic(bs)
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        list_ts = pd.to_datetime(basic["list_date"], format="%Y%m%d", errors="coerce")
        delist_ts = pd.to_datetime(basic["delist_date"], format="%Y%m%d", errors="coerce")
        overlap = list_ts.le(end_ts) & (delist_ts.isna() | delist_ts.ge(start_ts))
        basic = basic.loc[overlap].reset_index(drop=True)
        if max_stocks > 0:
            basic = basic.head(max_stocks).reset_index(drop=True)
        calendar = fetch_trade_calendar(bs, start, end)
        basic.to_parquet(ref_root / "stock_basic.parquet", index=False)
        calendar.to_parquet(ref_root / "trade_calendar.parquet", index=False)

        codes = list(basic["ts_code"])
        adjusted_loaded = 0
        raw_loaded = 0
        errors: list[dict[str, str]] = []
        raw_meta: dict[str, pd.DataFrame] = {}

        def fetch_with_retry(code: str, *, adjusted: bool, include_meta: bool) -> pd.DataFrame:
            last: Exception | None = None
            for attempt in range(4):
                try:
                    return fetch_history(
                        bs,
                        code,
                        start,
                        end,
                        adjusted=adjusted,
                        include_meta=include_meta,
                    )
                except Exception as exc:
                    last = exc
                    if attempt < 3:
                        time.sleep(max(0.5, sleep_seconds) * (2 ** attempt))
            raise RuntimeError(f"{code} history fetch failed after retries: {last}") from last

        for index, code in enumerate(codes, start=1):
            front_path = front_root / f"{code}.parquet"
            raw_path = raw_root / f"{code}.parquet"
            if refresh or not front_path.exists():
                try:
                    front = fetch_with_retry(code, adjusted=True, include_meta=False)
                    if not front.empty:
                        _write_qmt_cache(front, front_path)
                except Exception as exc:
                    errors.append({"code": code, "kind": "front", "error": str(exc)})
            if front_path.exists():
                adjusted_loaded += 1

            if refresh or not raw_path.exists():
                try:
                    raw = fetch_with_retry(code, adjusted=False, include_meta=True)
                    if not raw.empty:
                        _write_qmt_cache(raw, raw_path)
                        meta_path = raw_root / f"{code}.meta.parquet"
                        raw[["date", "isST", "tradestatus"]].to_parquet(meta_path, index=False)
                except Exception as exc:
                    errors.append({"code": code, "kind": "raw", "error": str(exc)})
            if raw_path.exists():
                raw_loaded += 1
                raw_cache = pd.read_parquet(raw_path)
                meta_path = raw_root / f"{code}.meta.parquet"
                if meta_path.exists():
                    meta = pd.read_parquet(meta_path)
                    raw_cache = raw_cache.merge(meta, on="date", how="left")
                raw_meta[code] = raw_cache
            if index == 1 or index % 50 == 0 or index == len(codes):
                print(
                    f"[BaoStock] {index}/{len(codes)} symbols; "
                    f"front={adjusted_loaded}, raw={raw_loaded}, errors={len(errors)}"
                )
            if sleep_seconds > 0 and index < len(codes):
                time.sleep(sleep_seconds)

        benchmark_path = front_root / f"{benchmark}.parquet"
        if refresh or not benchmark_path.exists():
            benchmark_frame = fetch_with_retry(benchmark, adjusted=True, include_meta=False)
            if benchmark_frame.empty:
                raise RuntimeError(f"BaoStock returned no benchmark data for {benchmark}.")
            _write_qmt_cache(benchmark_frame, benchmark_path)

    st, limits, susp = build_reference_tables(basic, calendar, raw_meta)
    st.to_parquet(ref_root / "stock_st.parquet", index=False)
    limits.to_parquet(ref_root / "stk_limit.parquet", index=False)
    susp.to_parquet(ref_root / "suspend_d.parquet", index=False)

    manifest = {
        "source": "baostock",
        "start": start,
        "end": end,
        "benchmark": benchmark,
        "symbols": len(basic),
        "adjusted_symbols_cached": adjusted_loaded,
        "raw_symbols_cached": raw_loaded,
        "reference_dir": str(ref_root),
        "bar_cache_dir": str(cache_root),
        "strict_ready": bool(adjusted_loaded == len(basic) and raw_loaded == len(basic)),
        "errors": errors,
    }
    pd.Series(manifest).to_json(ref_root / "free_data_manifest.json", force_ascii=False, indent=2)
    return manifest


@contextmanager
def _already_open(api):
    yield api


def verify_with_akshare(
    codes: Iterable[str],
    start: str,
    end: str,
    raw_cache_dir: str | Path,
    *,
    sample_size: int = 20,
    tolerance: float = 0.01,
    api=None,
) -> pd.DataFrame:
    ak = api or _import_akshare()
    cache_root = Path(raw_cache_dir)
    selected = list(dict.fromkeys(codes))[: max(int(sample_size), 0)]
    rows: list[dict] = []
    for code in selected:
        symbol = code.split(".")[0]
        path = cache_root / f"{code}.parquet"
        if not path.exists():
            rows.append({"code": code, "status": "missing_baostock_cache"})
            continue
        bs_frame = pd.read_parquet(path)
        bs_frame["date"] = pd.to_datetime(bs_frame["date"], errors="coerce")
        bs_frame = bs_frame.dropna(subset=["date"]).set_index("date")
        try:
            ak_frame = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=pd.Timestamp(start).strftime("%Y%m%d"),
                end_date=pd.Timestamp(end).strftime("%Y%m%d"),
                adjust="",
            )
        except Exception as exc:
            rows.append({"code": code, "status": "akshare_error", "error": str(exc)})
            continue
        if ak_frame is None or ak_frame.empty:
            rows.append({"code": code, "status": "akshare_empty"})
            continue
        date_col = "日期"
        close_col = "收盘"
        ak_frame = ak_frame.copy()
        ak_frame[date_col] = pd.to_datetime(ak_frame[date_col], errors="coerce")
        ak_frame[close_col] = pd.to_numeric(ak_frame[close_col], errors="coerce")
        ak_close = ak_frame.dropna(subset=[date_col]).set_index(date_col)[close_col]
        joined = pd.concat(
            [pd.to_numeric(bs_frame["close"], errors="coerce").rename("bs"), ak_close.rename("ak")],
            axis=1,
            join="inner",
        ).dropna()
        if joined.empty:
            rows.append({"code": code, "status": "no_overlap"})
            continue
        rel = (joined["bs"] - joined["ak"]).abs() / joined["ak"].abs().replace(0, np.nan)
        max_rel = float(rel.dropna().max()) if rel.notna().any() else float("nan")
        rows.append(
            {
                "code": code,
                "status": "pass" if np.isfinite(max_rel) and max_rel <= tolerance else "mismatch",
                "overlap_days": int(len(joined)),
                "max_close_relative_diff": max_rel,
            }
        )
    return pd.DataFrame(rows)

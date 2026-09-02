from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd


def _client(token: str | None = None):
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("tushare is not installed. Run: pip install -r requirements.txt") from exc
    resolved = token or os.environ.get("TUSHARE_TOKEN", "").strip()
    if not resolved:
        raise RuntimeError("Missing Tushare token. Set environment variable TUSHARE_TOKEN first.")
    return ts.pro_api(resolved)


def _save(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _merge_save(old: pd.DataFrame, new: pd.DataFrame, path: Path, keys: list[str]) -> Path:
    frames = [x for x in (old, new) if x is not None and not x.empty]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=keys)
    if not merged.empty:
        merged = merged.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    return _save(merged, path)


def sync_stock_basic(output_dir: str | Path, token: str | None = None) -> pd.DataFrame:
    pro = _client(token)
    fields = "ts_code,symbol,name,market,exchange,list_status,list_date,delist_date"
    frames = []
    for exchange in ("SSE", "SZSE"):
        for status in ("L", "D", "P"):
            frame = pro.stock_basic(exchange=exchange, list_status=status, fields=fields)
            if frame is not None and not frame.empty:
                frames.append(frame)
    if not frames:
        raise RuntimeError("Tushare stock_basic returned no rows.")
    basic = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="last")
    basic = basic.sort_values("ts_code").reset_index(drop=True)
    _save(basic, Path(output_dir) / "stock_basic.parquet")
    return basic


def sync_trade_calendar(
    output_dir: str | Path,
    start: str,
    end: str,
    token: str | None = None,
) -> pd.DataFrame:
    pro = _client(token)
    frame = pro.trade_cal(exchange="SSE", start_date=start, end_date=end, fields="exchange,cal_date,is_open,pretrade_date")
    if frame is None or frame.empty:
        raise RuntimeError("Tushare trade_cal returned no rows.")
    frame = frame.sort_values("cal_date").reset_index(drop=True)
    _save(frame, Path(output_dir) / "trade_calendar.parquet")
    return frame


def _retry_call(call, attempts: int = 5, base_sleep: float = 1.0):
    last = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # network / quota errors are SDK-defined
            last = exc
            if attempt + 1 < attempts:
                time.sleep(base_sleep * (2**attempt))
    raise RuntimeError(f"Tushare request failed after {attempts} attempts: {last}") from last


def sync_stock_st_dates(
    output_dir: str | Path,
    dates: Iterable[pd.Timestamp | str],
    token: str | None = None,
    sleep_seconds: float = 0.15,
) -> pd.DataFrame:
    pro = _client(token)
    path = Path(output_dir) / "stock_st.parquet"
    old = _load(path)
    done = set(old["trade_date"].astype(str)) if not old.empty and "trade_date" in old.columns else set()
    rows = []
    for value in dates:
        day = pd.Timestamp(value).strftime("%Y%m%d")
        if day in done:
            continue
        frame = _retry_call(lambda day=day: pro.stock_st(trade_date=day))
        if frame is not None and not frame.empty:
            rows.append(frame)
        else:
            # Keep a sentinel so resume knows this date was successfully queried.
            rows.append(pd.DataFrame({"trade_date": [day], "ts_code": ["__NONE__"], "name": [""], "type": [""], "type_name": [""]}))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    _merge_save(old, new, path, ["trade_date", "ts_code"])
    merged = _load(path)
    return merged[merged["ts_code"] != "__NONE__"].reset_index(drop=True) if not merged.empty else merged


def sync_limit_dates(
    output_dir: str | Path,
    dates: Iterable[pd.Timestamp | str],
    token: str | None = None,
    sleep_seconds: float = 0.15,
) -> pd.DataFrame:
    pro = _client(token)
    path = Path(output_dir) / "stk_limit.parquet"
    old = _load(path)
    done = set(old["trade_date"].astype(str)) if not old.empty and "trade_date" in old.columns else set()
    rows = []
    for value in dates:
        day = pd.Timestamp(value).strftime("%Y%m%d")
        if day in done:
            continue
        frame = _retry_call(lambda day=day: pro.stk_limit(trade_date=day))
        if frame is None or frame.empty:
            rows.append(pd.DataFrame({"trade_date": [day], "ts_code": ["__NONE__"], "pre_close": [float("nan")], "up_limit": [float("nan")], "down_limit": [float("nan")]}))
        else:
            if len(frame) >= 5800:
                print(f"WARNING: stk_limit {day} returned {len(frame)} rows (API cap). Coverage may be partial.")
            rows.append(frame)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    _merge_save(old, new, path, ["trade_date", "ts_code"])
    merged = _load(path)
    return merged[merged["ts_code"] != "__NONE__"].reset_index(drop=True) if not merged.empty else merged


def sync_suspension_range(
    output_dir: str | Path,
    start: str,
    end: str,
    token: str | None = None,
    sleep_seconds: float = 0.15,
) -> pd.DataFrame:
    """Cache daily suspension rows in month-sized chunks.

    Tushare `suspend_d` represents full-day suspension dates explicitly. Month chunking
    avoids relying on a large undocumented all-history response size.
    """
    import json

    pro = _client(token)
    root = Path(output_dir)
    path = root / "suspend_d.parquet"
    state_path = root / "suspend_d_months.json"
    old = _load(path)
    done: set[str] = set()
    if state_path.exists():
        try:
            done = set(json.loads(state_path.read_text(encoding="utf-8")))
        except Exception:
            done = set()

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    months = pd.period_range(start_ts, end_ts, freq="M")
    rows: list[pd.DataFrame] = []
    for period in months:
        key = str(period)
        if key in done:
            continue
        left = max(start_ts, period.start_time.normalize())
        right = min(end_ts, period.end_time.normalize())
        left_s = left.strftime("%Y%m%d")
        right_s = right.strftime("%Y%m%d")
        frame = _retry_call(
            lambda left_s=left_s, right_s=right_s: pro.suspend_d(
                start_date=left_s,
                end_date=right_s,
                suspend_type="S",
            )
        )
        if frame is not None and not frame.empty:
            rows.append(frame)
        done.add(key)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(sorted(done), ensure_ascii=False, indent=2), encoding="utf-8")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not new.empty:
        keys = [x for x in ["ts_code", "trade_date", "suspend_type"] if x in new.columns]
        if not keys:
            keys = ["ts_code", "trade_date"]
        _merge_save(old, new, path, keys)
    elif not path.exists():
        _save(pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type", "suspend_timing"]), path)
    return _load(path)

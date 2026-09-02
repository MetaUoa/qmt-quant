from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


FIELDS = ["open", "high", "low", "close", "volume", "amount", "preClose", "suspendFlag"]


def _xtdata():
    try:
        from xtquant import xtdata
    except ImportError as exc:
        raise RuntimeError(
            "xtquant is not importable. Run this inside the Python environment shipped with QMT/MiniQMT."
        ) from exc
    return xtdata


def get_sector_universe(sector: str = "沪深A股") -> List[str]:
    xtdata = _xtdata()
    codes = list(xtdata.get_stock_list_in_sector(sector) or [])
    return sorted({str(code).strip() for code in codes if code})


def read_universe_file(path: str | Path) -> List[str]:
    frame = pd.read_csv(path)
    if "code" not in frame.columns:
        raise ValueError("Universe CSV must contain a 'code' column.")
    return sorted({str(code).strip() for code in frame["code"].dropna() if str(code).strip()})


def download_daily_history(codes: Iterable[str], start: str, end: str) -> None:
    xtdata = _xtdata()
    code_list = list(dict.fromkeys(codes))
    if not code_list:
        return
    batch_api = getattr(xtdata, "download_history_data2", None)
    if batch_api is not None:
        batch_api(code_list, "1d", start, end)
        return
    for code in code_list:
        xtdata.download_history_data(code, "1d", start, end)


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index.tz_localize(None) if index.tz is not None else index
    values = pd.Series(index)
    numeric = pd.to_numeric(values, errors="coerce")
    valid_numeric = numeric.dropna()
    if not valid_numeric.empty:
        median = float(valid_numeric.median())
        if median > 1e11:
            return pd.to_datetime(numeric, unit="ms", errors="coerce")
        if median > 1e9:
            return pd.to_datetime(numeric, unit="s", errors="coerce")
    text = values.astype(str).str.replace(r"\.0$", "", regex=True)
    if text.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if text.str.fullmatch(r"\d{14}").all():
        return pd.to_datetime(text, format="%Y%m%d%H%M%S", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def _normalize_frame(frame: pd.DataFrame, fields: list[str] | None = None) -> pd.DataFrame:
    wanted = fields or FIELDS
    if frame is None or frame.empty:
        return pd.DataFrame(columns=wanted)
    out = frame.copy()
    out.index = _normalize_index(out.index)
    # Keep a canonical, source-independent index shape. Parquet round-trips
    # otherwise restore the serialized ``date`` column as a named index while
    # fresh QMT frames usually have an unnamed index.
    out.index.name = None
    out = out.loc[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for field in wanted:
        if field not in out.columns:
            out[field] = np.nan
        out[field] = pd.to_numeric(out[field], errors="coerce")
    return out[wanted]


def _cache_path(cache_dir: Path, code: str) -> Path:
    return cache_dir / f"{code.replace('/', '_')}.parquet"


def _read_cached(path: Path, start: str, end: str, fields: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=fields)
    frame = pd.read_parquet(path)
    if "date" not in frame.columns:
        return pd.DataFrame(columns=fields)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return _normalize_frame(frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)], fields)


def _write_cached(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy().reset_index().rename(columns={frame.index.name or "index": "date"})
    out.to_parquet(path, index=False)


def load_market_fields(
    codes: Iterable[str],
    start: str,
    end: str,
    fields: list[str],
    dividend_type: str = "front",
    batch_size: int = 200,
    cache_dir: str | Path | None = None,
    refresh_cache: bool = False,
) -> Dict[str, pd.DataFrame]:
    code_list = list(dict.fromkeys(codes))
    result: Dict[str, pd.DataFrame] = {}
    unresolved: list[str] = []
    cache_root = Path(cache_dir) if cache_dir else None

    for code in code_list:
        if cache_root is not None and not refresh_cache:
            frame = _read_cached(_cache_path(cache_root, code), start, end, fields)
            if not frame.empty:
                result[code] = frame
                continue
        unresolved.append(code)

    if not unresolved:
        return result

    xtdata = _xtdata()
    for offset in range(0, len(unresolved), batch_size):
        batch = unresolved[offset : offset + batch_size]
        raw = xtdata.get_market_data_ex(
            field_list=fields,
            stock_list=batch,
            period="1d",
            start_time=start,
            end_time=end,
            count=-1,
            dividend_type=dividend_type,
            fill_data=False,
        )
        for code in batch:
            frame = raw.get(code) if raw else None
            norm = _normalize_frame(frame, fields)
            if norm.empty:
                continue
            result[code] = norm
            if cache_root is not None:
                _write_cached(_cache_path(cache_root, code), norm)
    return result


def load_daily_bars(
    codes: Iterable[str],
    start: str,
    end: str,
    dividend_type: str = "front",
    batch_size: int = 200,
    cache_dir: str | Path | None = None,
    refresh_cache: bool = False,
) -> Dict[str, pd.DataFrame]:
    return load_market_fields(
        codes, start, end, FIELDS, dividend_type, batch_size, cache_dir, refresh_cache
    )


def load_limit_reference_bars(
    codes: Iterable[str],
    start: str,
    end: str,
    batch_size: int = 200,
    cache_dir: str | Path | None = None,
    refresh_cache: bool = False,
) -> Dict[str, pd.DataFrame]:
    return load_market_fields(
        codes,
        start,
        end,
        ["open", "close", "preClose"],
        dividend_type="none",
        batch_size=batch_size,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
    )


def coverage_report(codes: Iterable[str], bars: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for code in codes:
        frame = bars.get(code)
        valid = pd.Series(dtype=float) if frame is None or frame.empty else pd.to_numeric(frame["close"], errors="coerce").dropna()
        rows.append(
            {
                "code": code,
                "loaded": bool(len(valid)),
                "rows": int(len(valid)),
                "start": valid.index.min() if len(valid) else pd.NaT,
                "end": valid.index.max() if len(valid) else pd.NaT,
            }
        )
    return pd.DataFrame(rows)

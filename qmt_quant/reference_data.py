from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _find_table(root: Path, stem: str) -> Path | None:
    for suffix in (".parquet", ".csv"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _date(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _normalize_date_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    out = frame.copy()
    if column in out.columns:
        values = out[column]
        if pd.api.types.is_datetime64_any_dtype(values):
            out[column] = pd.to_datetime(values, errors="coerce").dt.normalize()
        else:
            text = values.astype("string").str.replace(r"[^0-9]", "", regex=True).str.slice(0, 8)
            out[column] = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return out


@dataclass(frozen=True)
class ReferenceAudit:
    basic_symbols: int
    st_dates: int
    limit_dates: int
    calendar_sessions: int


class ReferenceData:
    """Point-in-time A-share membership and execution constraints.

    `stock_basic` supplies listing/delisting dates. ST and daily price-limit tables are
    intentionally date keyed so the backtest never asks a future date while building
    the T close signal.
    """

    def __init__(
        self,
        stock_basic: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        st: pd.DataFrame | None = None,
        limits: pd.DataFrame | None = None,
    ) -> None:
        basic = stock_basic.copy()
        for col in ("list_date", "delist_date"):
            basic = _normalize_date_column(basic, col)
        if "ts_code" not in basic.columns:
            raise ValueError("stock_basic must contain ts_code")
        if "exchange" in basic.columns:
            basic = basic[basic["exchange"].isin(["SSE", "SZSE"])]
        basic = basic.drop_duplicates("ts_code", keep="last").reset_index(drop=True)
        self.stock_basic = basic
        self.calendar = pd.DatetimeIndex(calendar).normalize().drop_duplicates().sort_values()

        self._basic: Dict[str, Tuple[pd.Timestamp, pd.Timestamp | None]] = {}
        for row in basic.itertuples(index=False):
            code = str(getattr(row, "ts_code"))
            list_date = getattr(row, "list_date", pd.NaT)
            delist_date = getattr(row, "delist_date", pd.NaT)
            if pd.isna(list_date):
                continue
            self._basic[code] = (
                _date(list_date),
                None if pd.isna(delist_date) else _date(delist_date),
            )

        st_required = ["trade_date", "ts_code"]
        self.st = pd.DataFrame(columns=st_required) if st is None else st.copy()
        missing_st = [col for col in st_required if col not in self.st.columns]
        if missing_st:
            if self.st.empty:
                for col in missing_st:
                    self.st[col] = pd.Series(dtype="object")
            else:
                raise ValueError(f"stock_st must contain columns: {', '.join(st_required)}")
        if not self.st.empty:
            self.st = _normalize_date_column(self.st, "trade_date")
        self._st_map: Dict[pd.Timestamp, Set[str]] = {}
        for trade_date, group in self.st.dropna(subset=st_required).groupby("trade_date"):
            self._st_map[_date(trade_date)] = set(group["ts_code"].astype(str))

        limit_required = ["trade_date", "ts_code", "pre_close", "up_limit", "down_limit"]
        self.limits = pd.DataFrame(columns=limit_required) if limits is None else limits.copy()
        missing_limits = [col for col in limit_required if col not in self.limits.columns]
        if missing_limits:
            if self.limits.empty:
                for col in missing_limits:
                    self.limits[col] = pd.Series(dtype="object")
            else:
                raise ValueError(f"stk_limit must contain columns: {', '.join(limit_required)}")
        if not self.limits.empty:
            self.limits = _normalize_date_column(self.limits, "trade_date")
        self._limit_map: Dict[Tuple[pd.Timestamp, str], Tuple[float, float, float]] = {}
        for row in self.limits.dropna(subset=["trade_date", "ts_code"]).itertuples(index=False):
            try:
                values = (
                    float(getattr(row, "pre_close")),
                    float(getattr(row, "up_limit")),
                    float(getattr(row, "down_limit")),
                )
            except (TypeError, ValueError):
                continue
            self._limit_map[(_date(getattr(row, "trade_date")), str(getattr(row, "ts_code")))] = values

    @classmethod
    def from_dir(cls, directory: str | Path) -> "ReferenceData":
        root = Path(directory)
        basic_path = _find_table(root, "stock_basic")
        cal_path = _find_table(root, "trade_calendar")
        if basic_path is None or cal_path is None:
            raise FileNotFoundError(
                f"Reference data is incomplete under {root}. Run prepare_reference_data.py first."
            )
        basic = _read_table(basic_path)
        cal_frame = _read_table(cal_path)
        if "cal_date" not in cal_frame.columns:
            raise ValueError("trade_calendar must contain cal_date")
        cal = pd.to_datetime(cal_frame["cal_date"].astype("string"), format="%Y%m%d", errors="coerce")
        if "is_open" in cal_frame.columns:
            cal = cal[pd.to_numeric(cal_frame["is_open"], errors="coerce").fillna(0).astype(int).eq(1)]
        st_path = _find_table(root, "stock_st")
        limit_path = _find_table(root, "stk_limit")
        st = _read_table(st_path) if st_path else None
        limits = _read_table(limit_path) if limit_path else None
        return cls(basic, pd.DatetimeIndex(cal.dropna()), st, limits)

    def audit(self) -> ReferenceAudit:
        return ReferenceAudit(
            basic_symbols=len(self._basic),
            st_dates=len(self._st_map),
            limit_dates=len({key[0] for key in self._limit_map}),
            calendar_sessions=len(self.calendar),
        )

    @property
    def st_dates(self) -> Set[pd.Timestamp]:
        return set(self._st_map)

    @property
    def limit_dates(self) -> Set[pd.Timestamp]:
        return {key[0] for key in self._limit_map}

    def codes_ever_active(self, start, end) -> list[str]:
        start_ts, end_ts = _date(start), _date(end)
        result = []
        for code, (list_date, delist_date) in self._basic.items():
            if list_date <= end_ts and (delist_date is None or delist_date >= start_ts):
                result.append(code)
        return sorted(result)

    def listed_sessions(self, code: str, date) -> int:
        item = self._basic.get(code)
        if item is None:
            return 0
        list_date, _ = item
        ts = _date(date)
        right = int(self.calendar.searchsorted(ts, side="right"))
        left = int(self.calendar.searchsorted(list_date, side="left"))
        return max(right - left, 0)

    def is_member(self, code: str, date, min_listing_sessions: int = 0) -> bool:
        item = self._basic.get(code)
        if item is None:
            return False
        list_date, delist_date = item
        ts = _date(date)
        if ts < list_date or (delist_date is not None and ts > delist_date):
            return False
        return self.listed_sessions(code, ts) >= int(min_listing_sessions)

    def filter_members(self, codes: Iterable[str], date, min_listing_sessions: int = 0) -> list[str]:
        return [code for code in codes if self.is_member(code, date, min_listing_sessions)]

    def st_codes(self, date) -> Set[str]:
        return self._st_map.get(_date(date), set())

    def is_st(self, code: str, date) -> bool:
        return code in self.st_codes(date)

    def limit_prices(self, code: str, date) -> Tuple[float, float, float] | None:
        return self._limit_map.get((_date(date), code))

    def limit_blocked(self, code: str, date, open_px: float, side: str, tolerance: float = 0.001) -> bool:
        """Conservatively block execution when the opening auction is at the daily limit.

        Tushare limit prices are unadjusted while the signal bars may be front-adjusted.
        We therefore compare *ratios* (limit/pre-close) against open/preClose in the
        backtester. This method accepts that already-computed opening ratio via open_px
        when called with a synthetic pre-close of 1.0; use `limit_ratio` for clarity.
        """
        values = self.limit_prices(code, date)
        if values is None or not np.isfinite(open_px):
            return False
        pre_close, up_limit, down_limit = values
        if not np.isfinite(pre_close) or pre_close <= 0:
            return False
        ratio = float(open_px)
        up_ratio = up_limit / pre_close if np.isfinite(up_limit) else np.inf
        down_ratio = down_limit / pre_close if np.isfinite(down_limit) else -np.inf
        tol = abs(float(tolerance))
        side = side.upper()
        if side == "BUY":
            return ratio >= up_ratio * (1.0 - tol)
        if side == "SELL":
            return ratio <= down_ratio * (1.0 + tol)
        raise ValueError(f"Unknown side: {side}")

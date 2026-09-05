from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarQuality:
    label: str
    symbols_checked: int
    rows_checked: int
    missing_price_rows: int
    nonpositive_price_rows: int
    invalid_ohlc_rows: int
    negative_volume_rows: int
    negative_amount_rows: int

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.missing_price_rows,
                self.nonpositive_price_rows,
                self.invalid_ohlc_rows,
                self.negative_volume_rows,
                self.negative_amount_rows,
            )
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "passed": self.passed}


@dataclass(frozen=True)
class LimitQuality:
    rows_checked: int
    missing_rows: int
    nonpositive_rows: int
    inverted_rows: int

    @property
    def passed(self) -> bool:
        return not any((self.missing_rows, self.nonpositive_rows, self.inverted_rows))

    def to_dict(self) -> dict:
        return {**asdict(self), "passed": self.passed}


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in frame.columns:
            out[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            out[column] = np.nan
    return out


def audit_bar_collection(
    bars: Mapping[str, pd.DataFrame],
    *,
    label: str,
    require_ohlc: bool,
) -> tuple[BarQuality, pd.DataFrame]:
    details: list[dict] = []
    price_columns = ["open", "high", "low", "close"] if require_ohlc else ["open", "close", "preClose"]
    totals = {
        "symbols_checked": 0,
        "rows_checked": 0,
        "missing_price_rows": 0,
        "nonpositive_price_rows": 0,
        "invalid_ohlc_rows": 0,
        "negative_volume_rows": 0,
        "negative_amount_rows": 0,
    }

    for code, frame in sorted(bars.items()):
        if frame is None or frame.empty:
            continue
        numeric = _numeric(frame, price_columns + ["volume", "amount"])
        active = numeric[price_columns].notna().any(axis=1)
        active_frame = numeric.loc[active]
        if active_frame.empty:
            continue
        missing = active_frame[price_columns].isna().any(axis=1)
        nonpositive = active_frame[price_columns].le(0.0).any(axis=1)
        invalid_ohlc = pd.Series(False, index=active_frame.index)
        if require_ohlc:
            max_body = active_frame[["open", "close", "low"]].max(axis=1)
            min_body = active_frame[["open", "close", "high"]].min(axis=1)
            invalid_ohlc = (
                active_frame["high"].lt(max_body)
                | active_frame["low"].gt(min_body)
                | active_frame["high"].lt(active_frame["low"])
            )
        negative_volume = active_frame["volume"].lt(0.0) if "volume" in active_frame else pd.Series(False, index=active_frame.index)
        negative_amount = active_frame["amount"].lt(0.0) if "amount" in active_frame else pd.Series(False, index=active_frame.index)

        row = {
            "label": label,
            "code": str(code),
            "rows_checked": int(len(active_frame)),
            "missing_price_rows": int(missing.sum()),
            "nonpositive_price_rows": int(nonpositive.sum()),
            "invalid_ohlc_rows": int(invalid_ohlc.sum()),
            "negative_volume_rows": int(negative_volume.sum()),
            "negative_amount_rows": int(negative_amount.sum()),
        }
        row["passed"] = not any(row[key] for key in row if key.endswith("_rows") and key != "rows_checked")
        details.append(row)
        totals["symbols_checked"] += 1
        totals["rows_checked"] += row["rows_checked"]
        for key in (
            "missing_price_rows",
            "nonpositive_price_rows",
            "invalid_ohlc_rows",
            "negative_volume_rows",
            "negative_amount_rows",
        ):
            totals[key] += row[key]

    quality = BarQuality(label=label, **totals)
    detail = pd.DataFrame(
        details,
        columns=[
            "label",
            "code",
            "rows_checked",
            "missing_price_rows",
            "nonpositive_price_rows",
            "invalid_ohlc_rows",
            "negative_volume_rows",
            "negative_amount_rows",
            "passed",
        ],
    )
    return quality, detail


def audit_limit_reference_table(limits: pd.DataFrame) -> LimitQuality:
    required = ["pre_close", "up_limit", "down_limit"]
    if limits is None or limits.empty:
        return LimitQuality(rows_checked=0, missing_rows=0, nonpositive_rows=0, inverted_rows=0)
    numeric = _numeric(limits, required)
    missing = numeric[required].isna().any(axis=1)
    nonpositive = numeric[required].le(0.0).any(axis=1)
    inverted = (
        numeric["down_limit"].gt(numeric["pre_close"])
        | numeric["up_limit"].lt(numeric["pre_close"])
        | numeric["down_limit"].ge(numeric["up_limit"])
    )
    return LimitQuality(
        rows_checked=int(len(numeric)),
        missing_rows=int(missing.sum()),
        nonpositive_rows=int(nonpositive.sum()),
        inverted_rows=int(inverted.sum()),
    )

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import CostConfig
from .reference_data import ReferenceData


def deterministic_fill(
    cost: CostConfig,
    ts: pd.Timestamp,
    code: str,
    side: str,
) -> bool:
    probability = min(max(float(cost.fill_probability), 0.0), 1.0)
    if probability >= 1.0:
        return True
    token = f"{cost.fill_seed}|{ts.date()}|{code}|{side}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / float(2**64 - 1)
    return value <= probability


def commission(cost: CostConfig, notional: float) -> float:
    if notional <= 0:
        return 0.0
    return max(float(cost.min_commission), float(notional) * float(cost.commission_rate))


def equal_weight_target_shares(
    *,
    selected: Sequence[str],
    open_px: pd.DataFrame,
    execution_date: pd.Timestamp,
    portfolio_value: float,
    exposure: float,
    slippage_bps: float,
    lot_size: int,
) -> dict[str, int]:
    """Pure equal-weight target sizing used by the daily-bar backtest loop.

    This intentionally mirrors the current engine: target value is divided equally
    across already-tradable selected names, BUY slippage is applied before lot
    flooring, and targets are rounded down to whole board lots.
    """
    names = list(selected)
    if not names:
        return {}
    target_value = float(portfolio_value) * float(exposure) / len(names)
    slip = float(slippage_bps) / 10_000.0
    desired: dict[str, int] = {}
    for code in names:
        px = float(open_px.at[execution_date, code]) * (1.0 + slip)
        lots = int(target_value // (px * int(lot_size)))
        desired[str(code)] = max(lots, 0) * int(lot_size)
    return desired


def affordable_buy_quantity(
    *,
    requested_shares: int,
    execution_price: float,
    cash: float,
    cost: CostConfig,
) -> int:
    """Scale a BUY down by board lots until notional plus commission fits cash."""
    qty = max(int(requested_shares), 0)
    lot = int(cost.lot_size)
    while qty >= lot:
        notional = qty * float(execution_price)
        fee = commission(cost, notional)
        if notional + fee <= float(cash):
            break
        qty -= lot
    return qty if qty >= lot else 0


def mark_portfolio_value(
    *,
    cash: float,
    positions: Mapping[str, int],
    matrix: pd.DataFrame,
    close_px: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    index: int,
    reference: ReferenceData | None,
) -> float:
    value = float(cash)
    ts = calendar[index]
    for code, shares in positions.items():
        px = matrix.at[ts, code] if code in matrix.columns else np.nan
        if not np.isfinite(px) or px <= 0:
            if reference is not None and not reference.is_member(code, ts, 0):
                px = 0.0
            else:
                previous = (
                    close_px[code].iloc[: index + 1].dropna()
                    if code in close_px.columns
                    else pd.Series(dtype=float)
                )
                px = float(previous.iloc[-1]) if len(previous) else 0.0
        value += int(shares) * float(px)
    return float(value)


@dataclass
class TradabilityGuard:
    calendar: pd.DatetimeIndex
    open_px: pd.DataFrame
    high_px: pd.DataFrame
    low_px: pd.DataFrame
    close_px: pd.DataFrame
    suspend: pd.DataFrame
    limit_open_px: pd.DataFrame
    limit_preclose_px: pd.DataFrame
    reference: ReferenceData | None
    strict_reference: bool
    raw_limit_reference_supplied: bool
    limit_tolerance: float
    missing_suspend_rows: int = 0
    missing_limit_rows: int = 0

    def is_halted(self, ts: pd.Timestamp, code: str) -> bool:
        if code not in self.open_px.columns:
            return True
        op = self.open_px.at[ts, code]
        if code not in self.suspend.columns:
            if self.strict_reference:
                self.missing_suspend_rows += 1
                return True
            return (not np.isfinite(op)) or op <= 0
        flag = self.suspend.at[ts, code]
        if not np.isfinite(flag):
            if self.strict_reference:
                self.missing_suspend_rows += 1
                return True
            return (not np.isfinite(op)) or op <= 0
        return (not np.isfinite(op)) or op <= 0 or float(flag) == 1.0

    def opening_ratio(self, ts: pd.Timestamp, code: str) -> float | None:
        if code not in self.limit_open_px.columns or code not in self.limit_preclose_px.columns:
            return None
        op = self.limit_open_px.at[ts, code]
        prev = self.limit_preclose_px.at[ts, code]
        if (not np.isfinite(prev) or prev <= 0) and not self.raw_limit_reference_supplied:
            loc = self.calendar.get_loc(ts)
            if isinstance(loc, (int, np.integer)) and loc > 0:
                prev = self.close_px.at[self.calendar[loc - 1], code]
        if not np.isfinite(op) or not np.isfinite(prev) or prev <= 0:
            return None
        return float(op / prev)

    def bar_locked(self, ts: pd.Timestamp, code: str, side: str) -> bool:
        # Daily bars cannot reconstruct the intraday path. When exact daily limit data
        # are unavailable, reject only a one-price board rather than invent touch timing.
        if code not in self.open_px.columns:
            return False
        vals = [
            self.open_px.at[ts, code],
            self.high_px.at[ts, code],
            self.low_px.at[ts, code],
            self.close_px.at[ts, code],
        ]
        if not all(np.isfinite(value) and value > 0 for value in vals):
            return False
        spread = (max(vals) - min(vals)) / max(abs(float(vals[0])), 1e-12)
        ratio = self.opening_ratio(ts, code)
        if ratio is None or spread > 0.0005:
            return False
        return ratio > 1.045 if side == "BUY" else ratio < 0.955

    def limit_blocked(self, ts: pd.Timestamp, code: str, side: str) -> bool:
        ratio = self.opening_ratio(ts, code)
        if ratio is None:
            if self.reference is not None and self.strict_reference:
                self.missing_limit_rows += 1
                return True
            return self.bar_locked(ts, code, side)
        if self.reference is None:
            return self.bar_locked(ts, code, side)
        values = self.reference.limit_prices(code, ts)
        if values is None:
            self.missing_limit_rows += 1
            return True if self.strict_reference else self.bar_locked(ts, code, side)
        return self.reference.limit_blocked(
            code,
            ts,
            ratio,
            side,
            tolerance=float(self.limit_tolerance),
        )

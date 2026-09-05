from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import CostConfig, StrategyConfig
from .reference_data import ReferenceData


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    config: dict


def _stamp_tax_rate(ts: pd.Timestamp) -> float:
    # A-share sell-side stamp duty: 0.10% before 2023-08-28, 0.05% after.
    return 0.0005 if ts >= pd.Timestamp("2023-08-28") else 0.0010


def _t1_sell_allowed(last_buy_date: pd.Timestamp | None, execution_date: pd.Timestamp) -> bool:
    if last_buy_date is None:
        return True
    return pd.Timestamp(last_buy_date).normalize() < pd.Timestamp(execution_date).normalize()


def _panel(bars: Dict[str, pd.DataFrame], field: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Build one aligned field panel without reindexing every symbol in Python."""
    series = {
        code: frame[field]
        for code, frame in bars.items()
        if field in frame.columns
    }
    if not series:
        return pd.DataFrame(index=calendar)
    panel = pd.concat(series, axis=1, copy=False)
    panel.columns = panel.columns.astype(str)
    return panel.reindex(pd.DatetimeIndex(calendar)).sort_index()


def _build_features(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    cfg: StrategyConfig,
    eligibility_price: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r_short = close.pct_change(cfg.mom_short, fill_method=None)
    r_mid = close.pct_change(cfg.mom_mid, fill_method=None)
    r_long = close.pct_change(cfg.mom_long, fill_method=None)
    ma_fast = close.rolling(cfg.ma_fast, min_periods=cfg.ma_fast).mean()
    ma_slow = close.rolling(cfg.ma_slow, min_periods=cfg.ma_slow).mean()
    daily_ret = close.pct_change(fill_method=None)
    vol = daily_ret.rolling(cfg.vol_window, min_periods=cfg.vol_window).std()
    avg_amount = amount.rolling(cfg.amount_window, min_periods=cfg.amount_window).mean()

    score = (
        cfg.weight_short * r_short
        + cfg.weight_mid * r_mid
        + cfg.weight_long * r_long
        - cfg.vol_penalty * vol
    )
    price_for_filter = close if eligibility_price is None else eligibility_price.reindex_like(close)
    eligible = (
        (price_for_filter >= cfg.min_price)
        & (close > ma_fast)
        & (ma_fast > ma_slow)
        & (r_long >= cfg.min_momentum)
        & (vol <= cfg.max_daily_vol)
        & (avg_amount >= cfg.min_amount)
    )
    return score.where(eligible), eligible


def calculate_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna().sort_index()
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1.0 / 252.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    max_drawdown = float(drawdown.min())
    ann_vol = float(rets.std(ddof=0) * np.sqrt(252)) if len(rets) else 0.0
    std = float(rets.std(ddof=0)) if len(rets) else 0.0
    sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "ending_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "multiple": float(equity.iloc[-1] / equity.iloc[0]),
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "calmar": calmar,
        "positive_day_ratio": float((rets > 0).mean()) if len(rets) else 0.0,
    }


def rebalance_schedule(calendar: pd.DatetimeIndex, cfg: StrategyConfig) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    calendar = pd.DatetimeIndex(calendar).sort_values()
    if len(calendar) < 2:
        return []
    delay = max(int(cfg.execution_delay_sessions), 1)
    start_i = min(cfg.warmup + delay - 1, max(len(calendar) - 1, 0))
    schedule: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(start_i, len(calendar)):
        if ((i - start_i) % cfg.rebalance_days) == 0 and i >= delay:
            schedule.append((calendar[i - delay], calendar[i]))
    return schedule


def run_backtest(
    bars: Dict[str, pd.DataFrame],
    benchmark_code: str,
    strategy: StrategyConfig | None = None,
    costs: CostConfig | None = None,
    reference: ReferenceData | None = None,
    strict_reference: bool = False,
    limit_reference_bars: Dict[str, pd.DataFrame] | None = None,
    *,
    score_override: pd.DataFrame | None = None,
    risk_on_override: pd.Series | None = None,
) -> BacktestResult:
    cfg = strategy or StrategyConfig()
    cost = costs or CostConfig()
    if strict_reference and reference is None:
        raise ValueError("strict_reference requires point-in-time ReferenceData.")
    if strict_reference and limit_reference_bars is None:
        raise ValueError("strict_reference requires unadjusted QMT limit_reference_bars.")
    if benchmark_code not in bars:
        raise ValueError(f"Benchmark {benchmark_code} is missing from bars.")

    benchmark_index = bars[benchmark_code].index
    calendar = pd.DatetimeIndex(sorted(pd.unique(benchmark_index)))
    stock_bars = {k: v for k, v in bars.items() if k != benchmark_code}
    if not stock_bars:
        raise ValueError("No stock bars were supplied.")

    open_px = _panel(stock_bars, "open", calendar)
    high_px = _panel(stock_bars, "high", calendar)
    low_px = _panel(stock_bars, "low", calendar)
    close_px = _panel(stock_bars, "close", calendar)
    preclose_px = _panel(stock_bars, "preClose", calendar)
    if limit_reference_bars is not None:
        raw_stock_bars = {k: v for k, v in limit_reference_bars.items() if k != benchmark_code}
        limit_open_px = _panel(raw_stock_bars, "open", calendar)
        raw_close_px = _panel(raw_stock_bars, "close", calendar)
        limit_preclose_px = _panel(raw_stock_bars, "preClose", calendar)
    else:
        limit_open_px = open_px
        raw_close_px = close_px
        limit_preclose_px = preclose_px
    amount = _panel(stock_bars, "amount", calendar)
    # Do not fill missing suspension flags with zero. In strict PIT mode an unknown
    # suspension state is non-tradable; non-strict legacy backtests retain the open-price fallback.
    suspend = _panel(stock_bars, "suspendFlag", calendar)
    if score_override is None:
        score, _ = _build_features(close_px, amount, cfg, eligibility_price=raw_close_px)
    else:
        score = score_override.reindex(index=calendar, columns=close_px.columns).apply(
            pd.to_numeric, errors="coerce"
        )

    benchmark_close = bars[benchmark_code]["close"].reindex(calendar).ffill()
    benchmark_ma = benchmark_close.rolling(cfg.benchmark_ma, min_periods=cfg.benchmark_ma).mean()
    benchmark_mom = benchmark_close.pct_change(cfg.benchmark_mom_days, fill_method=None)
    breadth_ma = close_px.rolling(cfg.breadth_ma, min_periods=cfg.breadth_ma).mean()
    breadth = (close_px > breadth_ma).sum(axis=1) / close_px.notna().sum(axis=1).replace(0, np.nan)
    breadth_ok = breadth.fillna(0.0) >= float(cfg.min_breadth)
    risk_on = (benchmark_close > benchmark_ma) & (benchmark_mom > cfg.benchmark_mom_floor) & breadth_ok
    if risk_on_override is not None:
        risk_on = risk_on_override.reindex(calendar).fillna(False).astype(bool)

    cash = float(cost.initial_cash)
    positions: Dict[str, int] = {}
    last_buy_date: Dict[str, pd.Timestamp] = {}
    trade_rows: List[dict] = []
    equity_rows: List[dict] = []
    rebalance_count = 0
    blocked_st = 0
    blocked_limit_buy = 0
    blocked_limit_sell = 0
    blocked_suspend = 0
    blocked_t1_sell = 0
    missing_suspend_rows = 0
    missing_limit_rows = 0
    missing_st_dates = 0
    missing_limit_dates = 0
    blocked_random_fill = 0

    def mark_value(i: int, use_open: bool = False) -> float:
        matrix = open_px if use_open else close_px
        value = cash
        ts = calendar[i]
        for code, shares in positions.items():
            px = matrix.at[ts, code] if code in matrix.columns else np.nan
            if not np.isfinite(px) or px <= 0:
                if reference is not None and not reference.is_member(code, ts, 0):
                    px = 0.0  # conservative post-delist write-down when no executable quote exists
                else:
                    prev = close_px[code].iloc[: i + 1].dropna() if code in close_px.columns else pd.Series(dtype=float)
                    px = float(prev.iloc[-1]) if len(prev) else 0.0
            value += shares * float(px)
        return float(value)

    def should_fill(ts: pd.Timestamp, code: str, side: str) -> bool:
        probability = min(max(float(cost.fill_probability), 0.0), 1.0)
        if probability >= 1.0:
            return True
        token = f"{cost.fill_seed}|{ts.date()}|{code}|{side}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / float(2**64 - 1)
        return value <= probability

    def commission(notional: float) -> float:
        if notional <= 0:
            return 0.0
        return max(cost.min_commission, notional * cost.commission_rate)

    def is_halted(ts: pd.Timestamp, code: str) -> bool:
        nonlocal missing_suspend_rows
        if code not in open_px.columns:
            return True
        op = open_px.at[ts, code]
        if code not in suspend.columns:
            if strict_reference:
                missing_suspend_rows += 1
                return True
            return (not np.isfinite(op)) or op <= 0
        flag = suspend.at[ts, code]
        if not np.isfinite(flag):
            if strict_reference:
                missing_suspend_rows += 1
                return True
            return (not np.isfinite(op)) or op <= 0
        return (not np.isfinite(op)) or op <= 0 or float(flag) == 1.0

    def opening_ratio(ts: pd.Timestamp, code: str) -> float | None:
        if code not in limit_open_px.columns or code not in limit_preclose_px.columns:
            return None
        op = limit_open_px.at[ts, code]
        prev = limit_preclose_px.at[ts, code]
        if (not np.isfinite(prev) or prev <= 0) and limit_reference_bars is None:
            # Compatibility fallback only when raw unadjusted reference bars were not supplied.
            loc = calendar.get_loc(ts)
            if isinstance(loc, (int, np.integer)) and loc > 0:
                prev = close_px.at[calendar[loc - 1], code]
        if not np.isfinite(op) or not np.isfinite(prev) or prev <= 0:
            return None
        return float(op / prev)

    def bar_locked(ts: pd.Timestamp, code: str, side: str) -> bool:
        # Daily bars cannot reconstruct the intraday path. When exact daily limit data
        # are unavailable, reject only a one-price board rather than invent touch timing.
        if code not in open_px.columns:
            return False
        vals = [
            open_px.at[ts, code],
            high_px.at[ts, code],
            low_px.at[ts, code],
            close_px.at[ts, code],
        ]
        if not all(np.isfinite(v) and v > 0 for v in vals):
            return False
        spread = (max(vals) - min(vals)) / max(abs(float(vals[0])), 1e-12)
        ratio = opening_ratio(ts, code)
        if ratio is None or spread > 0.0005:
            return False
        return ratio > 1.045 if side == "BUY" else ratio < 0.955

    def limit_blocked(ts: pd.Timestamp, code: str, side: str) -> bool:
        nonlocal missing_limit_rows
        ratio = opening_ratio(ts, code)
        if ratio is None:
            if reference is not None and strict_reference:
                missing_limit_rows += 1
                return True
            return bar_locked(ts, code, side)
        if reference is None:
            return bar_locked(ts, code, side)
        values = reference.limit_prices(code, ts)
        if values is None:
            missing_limit_rows += 1
            return True if strict_reference else bar_locked(ts, code, side)
        return reference.limit_blocked(code, ts, ratio, side, tolerance=cost.limit_tolerance)

    delay = max(int(cfg.execution_delay_sessions), 1)
    start_i = min(cfg.warmup + delay - 1, max(len(calendar) - 1, 0))
    for i, ts in enumerate(calendar):
        if i < start_i:
            equity_rows.append(
                {
                    "date": ts,
                    "equity": cash,
                    "cash": cash,
                    "positions": 0,
                    "risk_on": False,
                }
            )
            continue

        do_rebalance = ((i - start_i) % cfg.rebalance_days) == 0 and i > 0
        if do_rebalance:
            signal_ts = calendar[i - delay]
            if reference is not None:
                if signal_ts not in reference.st_dates:
                    missing_st_dates += 1
                    if strict_reference:
                        raise ValueError(
                            f"Missing historical ST snapshot for signal date {signal_ts.date()}"
                        )
                if ts not in reference.limit_dates:
                    missing_limit_dates += 1
                    if strict_reference:
                        raise ValueError(
                            f"Missing daily price-limit snapshot for execution date {ts.date()}"
                        )

            row = score.loc[signal_ts].dropna().sort_values(ascending=False)
            if reference is not None:
                member_codes = set(
                    reference.filter_members(row.index, signal_ts, cfg.min_listing_sessions)
                )
                row = row[row.index.isin(member_codes)]
                st_codes = reference.st_codes(signal_ts)
                if st_codes:
                    before = len(row)
                    row = row.drop(
                        index=row.index.intersection(st_codes), errors="ignore"
                    )
                    blocked_st += before - len(row)

            selected = (
                list(row.head(cfg.top_n).index)
                if bool(risk_on.loc[signal_ts])
                else []
            )

            tradable = []
            for code in selected:
                if is_halted(ts, code):
                    blocked_suspend += 1
                    continue
                if limit_blocked(ts, code, "BUY"):
                    blocked_limit_buy += 1
                    continue
                tradable.append(code)
            selected = tradable

            pre_value = mark_value(i, use_open=True)
            exposure = (
                1.0 if bool(risk_on.loc[signal_ts]) else float(cfg.risk_off_exposure)
            )
            target_value = (
                pre_value * exposure / max(len(selected), 1) if selected else 0.0
            )
            slip = cost.slippage_bps / 10_000.0

            desired: Dict[str, int] = {}
            for code in selected:
                px = float(open_px.at[ts, code]) * (1.0 + slip)
                lots = int(target_value // (px * cost.lot_size))
                desired[code] = max(lots, 0) * cost.lot_size

            # Sell first so cash is available for buys. T+1 is explicit: shares whose
            # most recent acquisition date is today cannot be sold today, even if a
            # future scheduler is changed to permit multiple decisions in one session.
            for code in list(positions):
                current = positions.get(code, 0)
                target = desired.get(code, 0)
                qty = max(current - target, 0)
                if qty <= 0:
                    continue
                if not _t1_sell_allowed(last_buy_date.get(code), ts):
                    blocked_t1_sell += 1
                    continue
                if is_halted(ts, code):
                    blocked_suspend += 1
                    continue
                if limit_blocked(ts, code, "SELL"):
                    blocked_limit_sell += 1
                    continue
                if not should_fill(ts, code, "SELL"):
                    blocked_random_fill += 1
                    continue
                exec_px = float(open_px.at[ts, code]) * (1.0 - slip)
                notional = qty * exec_px
                fee = commission(notional)
                tax = notional * _stamp_tax_rate(ts)
                cash += notional - fee - tax
                positions[code] = current - qty
                if positions[code] <= 0:
                    del positions[code]
                    last_buy_date.pop(code, None)
                trade_rows.append(
                    {
                        "date": ts,
                        "code": code,
                        "side": "SELL",
                        "shares": qty,
                        "price": exec_px,
                        "notional": notional,
                        "commission": fee,
                        "stamp_tax": tax,
                        "signal_date": signal_ts,
                    }
                )

            # Buy second, scaling down each order if cash is insufficient.
            for code in selected:
                current = positions.get(code, 0)
                target = desired.get(code, 0)
                qty = max(target - current, 0)
                if qty <= 0:
                    continue
                # Tradability was fixed before order sizing; only cash can change after sells.
                if not should_fill(ts, code, "BUY"):
                    blocked_random_fill += 1
                    continue
                exec_px = float(open_px.at[ts, code]) * (1.0 + slip)
                lot = cost.lot_size
                while qty >= lot:
                    notional = qty * exec_px
                    fee = commission(notional)
                    if notional + fee <= cash:
                        break
                    qty -= lot
                if qty < lot:
                    continue
                notional = qty * exec_px
                fee = commission(notional)
                cash -= notional + fee
                positions[code] = current + qty
                last_buy_date[code] = pd.Timestamp(ts).normalize()
                trade_rows.append(
                    {
                        "date": ts,
                        "code": code,
                        "side": "BUY",
                        "shares": qty,
                        "price": exec_px,
                        "notional": notional,
                        "commission": fee,
                        "stamp_tax": 0.0,
                        "signal_date": signal_ts,
                    }
                )
            rebalance_count += 1

        eq = mark_value(i, use_open=False)
        equity_rows.append(
            {
                "date": ts,
                "equity": eq,
                "cash": cash,
                "positions": len(positions),
                "risk_on": bool(risk_on.loc[ts]) if pd.notna(risk_on.loc[ts]) else False,
            }
        )

    equity = pd.DataFrame(equity_rows).set_index("date")
    trades = pd.DataFrame(trade_rows)
    metrics = calculate_metrics(equity["equity"])
    metrics.update(
        {
            "trade_count": int(len(trades)),
            "rebalance_count": int(rebalance_count),
            "initial_cash": float(cost.initial_cash),
            "blocked_st_candidates": int(blocked_st),
            "blocked_limit_buys": int(blocked_limit_buy),
            "blocked_limit_sells": int(blocked_limit_sell),
            "blocked_suspended": int(blocked_suspend),
            "blocked_t1_sells": int(blocked_t1_sell),
            "missing_suspend_rows": int(missing_suspend_rows),
            "missing_limit_rows": int(missing_limit_rows),
            "missing_st_dates": int(missing_st_dates),
            "missing_limit_dates": int(missing_limit_dates),
            "point_in_time_universe": bool(reference is not None),
            "strict_reference": bool(strict_reference),
            "raw_limit_reference": bool(limit_reference_bars is not None),
            "blocked_random_fill": int(blocked_random_fill),
            "execution_delay_sessions": int(delay),
            "fill_probability": float(cost.fill_probability),
            "t_plus_one_enforced": True,
            "limit_model": "open_auction_reference_plus_one_price_daily_fallback",
            "intraday_limit_touch_modelled": False,
            "average_market_breadth": float(breadth.mean()) if len(breadth.dropna()) else 0.0,
        }
    )
    if score_override is not None:
        metrics["score_override"] = True
    if risk_on_override is not None:
        metrics["risk_on_override"] = True
    return BacktestResult(
        equity=equity,
        trades=trades,
        metrics=metrics,
        config={"strategy": asdict(cfg), "costs": asdict(cost)},
    )

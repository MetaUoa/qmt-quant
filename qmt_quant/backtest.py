from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .backtest_execution import (
    TradabilityGuard,
    affordable_buy_quantity,
    build_rebalance_order_plan,
    deterministic_fill,
    equal_weight_target_shares,
    mark_portfolio_value,
    settle_buy,
    settle_sell,
)
from .backtest_reporting import BacktestDiagnostics, assemble_backtest_metrics
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
    missing_st_dates = 0
    missing_limit_dates = 0
    blocked_random_fill = 0

    guard = TradabilityGuard(
        calendar=calendar,
        open_px=open_px,
        high_px=high_px,
        low_px=low_px,
        close_px=close_px,
        suspend=suspend,
        limit_open_px=limit_open_px,
        limit_preclose_px=limit_preclose_px,
        reference=reference,
        strict_reference=bool(strict_reference),
        raw_limit_reference_supplied=limit_reference_bars is not None,
        limit_tolerance=float(cost.limit_tolerance),
    )

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
                if guard.is_halted(ts, code):
                    blocked_suspend += 1
                    continue
                if guard.limit_blocked(ts, code, "BUY"):
                    blocked_limit_buy += 1
                    continue
                tradable.append(code)
            selected = tradable

            pre_value = mark_portfolio_value(
                cash=cash,
                positions=positions,
                matrix=open_px,
                close_px=close_px,
                calendar=calendar,
                index=i,
                reference=reference,
            )
            exposure = (
                1.0 if bool(risk_on.loc[signal_ts]) else float(cfg.risk_off_exposure)
            )
            slip = cost.slippage_bps / 10_000.0
            desired = equal_weight_target_shares(
                selected=selected,
                open_px=open_px,
                execution_date=ts,
                portfolio_value=pre_value,
                exposure=exposure,
                slippage_bps=cost.slippage_bps,
                lot_size=cost.lot_size,
            )
            order_plan = build_rebalance_order_plan(
                positions=positions,
                desired=desired,
                selected=selected,
            )

            # Sell first so cash is available for buys. T+1 is explicit: shares whose
            # most recent acquisition date is today cannot be sold today, even if a
            # future scheduler is changed to permit multiple decisions in one session.
            for intent in order_plan.sells:
                code = intent.code
                current = positions.get(code, 0)
                qty = intent.quantity
                if not _t1_sell_allowed(last_buy_date.get(code), ts):
                    blocked_t1_sell += 1
                    continue
                if guard.is_halted(ts, code):
                    blocked_suspend += 1
                    continue
                if guard.limit_blocked(ts, code, "SELL"):
                    blocked_limit_sell += 1
                    continue
                if not deterministic_fill(cost, ts, code, "SELL"):
                    blocked_random_fill += 1
                    continue
                exec_px = float(open_px.at[ts, code]) * (1.0 - slip)
                sell_settlement = settle_sell(
                    cash=cash,
                    current_shares=current,
                    quantity=qty,
                    execution_price=exec_px,
                    cost=cost,
                    stamp_tax_rate=_stamp_tax_rate(ts),
                )
                cash = sell_settlement.ending_cash
                positions[code] = sell_settlement.ending_shares
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
                        "notional": sell_settlement.notional,
                        "commission": sell_settlement.commission,
                        "stamp_tax": sell_settlement.stamp_tax,
                        "signal_date": signal_ts,
                    }
                )

            # Buy second, scaling down each order if cash is insufficient.
            for intent in order_plan.buys:
                code = intent.code
                current = positions.get(code, 0)
                qty = intent.quantity
                # Tradability was fixed before order sizing; only cash can change after sells.
                if not deterministic_fill(cost, ts, code, "BUY"):
                    blocked_random_fill += 1
                    continue
                exec_px = float(open_px.at[ts, code]) * (1.0 + slip)
                qty = affordable_buy_quantity(
                    requested_shares=qty,
                    execution_price=exec_px,
                    cash=cash,
                    cost=cost,
                )
                if qty <= 0:
                    continue
                buy_settlement = settle_buy(
                    cash=cash,
                    current_shares=current,
                    quantity=qty,
                    execution_price=exec_px,
                    cost=cost,
                )
                cash = buy_settlement.ending_cash
                positions[code] = buy_settlement.ending_shares
                last_buy_date[code] = pd.Timestamp(ts).normalize()
                trade_rows.append(
                    {
                        "date": ts,
                        "code": code,
                        "side": "BUY",
                        "shares": qty,
                        "price": exec_px,
                        "notional": buy_settlement.notional,
                        "commission": buy_settlement.commission,
                        "stamp_tax": 0.0,
                        "signal_date": signal_ts,
                    }
                )
            rebalance_count += 1

        eq = mark_portfolio_value(
            cash=cash,
            positions=positions,
            matrix=close_px,
            close_px=close_px,
            calendar=calendar,
            index=i,
            reference=reference,
        )
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
    metrics = assemble_backtest_metrics(
        calculate_metrics(equity["equity"]),
        BacktestDiagnostics(
            trade_count=len(trades),
            rebalance_count=rebalance_count,
            initial_cash=cost.initial_cash,
            blocked_st_candidates=blocked_st,
            blocked_limit_buys=blocked_limit_buy,
            blocked_limit_sells=blocked_limit_sell,
            blocked_suspended=blocked_suspend,
            blocked_t1_sells=blocked_t1_sell,
            missing_suspend_rows=guard.missing_suspend_rows,
            missing_limit_rows=guard.missing_limit_rows,
            missing_st_dates=missing_st_dates,
            missing_limit_dates=missing_limit_dates,
            point_in_time_universe=reference is not None,
            strict_reference=strict_reference,
            raw_limit_reference=limit_reference_bars is not None,
            blocked_random_fill=blocked_random_fill,
            execution_delay_sessions=delay,
            fill_probability=cost.fill_probability,
            average_market_breadth=(
                float(breadth.mean()) if len(breadth.dropna()) else 0.0
            ),
            score_override=score_override is not None,
            risk_on_override=risk_on_override is not None,
        ),
    )
    return BacktestResult(
        equity=equity,
        trades=trades,
        metrics=metrics,
        config={"strategy": asdict(cfg), "costs": asdict(cost)},
    )

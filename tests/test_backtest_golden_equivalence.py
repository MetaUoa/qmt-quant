from __future__ import annotations

import pandas as pd

from qmt_quant.backtest import run_backtest
from qmt_quant.config import CostConfig, StrategyConfig


def _flat_frame(index: pd.DatetimeIndex, price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1_000_000.0,
            "amount": 50_000_000.0,
            "preClose": price,
            "suspendFlag": 0.0,
        },
        index=index,
    )


def _strategy() -> StrategyConfig:
    return StrategyConfig(
        mom_short=1,
        mom_mid=1,
        mom_long=1,
        ma_fast=1,
        ma_slow=1,
        vol_window=1,
        amount_window=1,
        benchmark_ma=1,
        benchmark_mom_days=1,
        breadth_ma=1,
        top_n=1,
        rebalance_days=1,
        execution_delay_sessions=1,
        min_price=1.0,
        min_amount=1.0,
        min_momentum=-1.0,
        max_daily_vol=1.0,
        benchmark_mom_floor=-1.0,
        min_listing_sessions=1,
    )


def _rounded(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    return value


def test_whole_engine_public_outputs_match_frozen_golden_snapshot() -> None:
    index = pd.bdate_range("2025-01-02", periods=14)
    bars = {
        "AAA.SZ": _flat_frame(index, 10.0),
        "BBB.SZ": _flat_frame(index, 20.0),
        "000905.SH": _flat_frame(index, 100.0),
    }
    score = pd.DataFrame(index=index, columns=["AAA.SZ", "BBB.SZ"], dtype=float)
    score.loc[:, "AAA.SZ"] = 2.0
    score.loc[:, "BBB.SZ"] = 1.0
    score.loc[index[7] :, "AAA.SZ"] = 1.0
    score.loc[index[7] :, "BBB.SZ"] = 2.0
    risk_on = pd.Series(True, index=index)
    costs = CostConfig(
        initial_cash=100_000.0,
        commission_rate=0.001,
        min_commission=5.0,
        slippage_bps=10.0,
        lot_size=100,
        fill_probability=1.0,
    )

    result = run_backtest(
        bars,
        "000905.SH",
        _strategy(),
        costs,
        score_override=score,
        risk_on_override=risk_on,
    )

    trades = []
    for row in result.trades.itertuples(index=False):
        trades.append(
            {
                "date": str(pd.Timestamp(row.date).date()),
                "signal_date": str(pd.Timestamp(row.signal_date).date()),
                "code": row.code,
                "side": row.side,
                "shares": int(row.shares),
                "price": _rounded(float(row.price)),
                "notional": _rounded(float(row.notional)),
                "commission": _rounded(float(row.commission)),
                "stamp_tax": _rounded(float(row.stamp_tax)),
            }
        )

    equity = []
    for position in (0, 5, 6, 7, 8, 13):
        row = result.equity.iloc[position]
        equity.append(
            {
                "date": str(pd.Timestamp(result.equity.index[position]).date()),
                "equity": _rounded(float(row["equity"])),
                "cash": _rounded(float(row["cash"])),
                "positions": int(row["positions"]),
                "risk_on": bool(row["risk_on"]),
            }
        )

    metric_names = (
        "ending_equity",
        "trade_count",
        "rebalance_count",
        "blocked_st_candidates",
        "blocked_limit_buys",
        "blocked_limit_sells",
        "blocked_suspended",
        "blocked_t1_sells",
        "blocked_random_fill",
        "missing_suspend_rows",
        "missing_limit_rows",
        "execution_delay_sessions",
        "fill_probability",
        "t_plus_one_enforced",
        "intraday_limit_touch_modelled",
        "score_override",
        "risk_on_override",
    )
    metrics = {name: _rounded(result.metrics[name]) for name in metric_names}

    assert {"trades": trades, "equity": equity, "metrics": metrics} == {
        "trades": [
            {
                "date": "2025-01-10",
                "signal_date": "2025-01-09",
                "code": "AAA.SZ",
                "side": "BUY",
                "shares": 9900,
                "price": 10.01,
                "notional": 99099.0,
                "commission": 99.099,
                "stamp_tax": 0.0,
            },
            {
                "date": "2025-01-14",
                "signal_date": "2025-01-13",
                "code": "AAA.SZ",
                "side": "SELL",
                "shares": 9900,
                "price": 9.99,
                "notional": 98901.0,
                "commission": 98.901,
                "stamp_tax": 49.4505,
            },
            {
                "date": "2025-01-14",
                "signal_date": "2025-01-13",
                "code": "BBB.SZ",
                "side": "BUY",
                "shares": 4900,
                "price": 20.02,
                "notional": 98098.0,
                "commission": 98.098,
                "stamp_tax": 0.0,
            },
        ],
        "equity": [
            {
                "date": "2025-01-02",
                "equity": 100000.0,
                "cash": 100000.0,
                "positions": 0,
                "risk_on": False,
            },
            {
                "date": "2025-01-09",
                "equity": 100000.0,
                "cash": 100000.0,
                "positions": 0,
                "risk_on": False,
            },
            {
                "date": "2025-01-10",
                "equity": 99801.901,
                "cash": 801.901,
                "positions": 1,
                "risk_on": True,
            },
            {
                "date": "2025-01-13",
                "equity": 99801.901,
                "cash": 801.901,
                "positions": 1,
                "risk_on": True,
            },
            {
                "date": "2025-01-14",
                "equity": 99358.4515,
                "cash": 1358.4515,
                "positions": 1,
                "risk_on": True,
            },
            {
                "date": "2025-01-21",
                "equity": 99358.4515,
                "cash": 1358.4515,
                "positions": 1,
                "risk_on": True,
            },
        ],
        "metrics": {
            "ending_equity": 99358.4515,
            "trade_count": 3,
            "rebalance_count": 8,
            "blocked_st_candidates": 0,
            "blocked_limit_buys": 0,
            "blocked_limit_sells": 0,
            "blocked_suspended": 0,
            "blocked_t1_sells": 0,
            "blocked_random_fill": 0,
            "missing_suspend_rows": 0,
            "missing_limit_rows": 0,
            "execution_delay_sessions": 1,
            "fill_probability": 1.0,
            "t_plus_one_enforced": True,
            "intraday_limit_touch_modelled": False,
            "score_override": True,
            "risk_on_override": True,
        },
    }

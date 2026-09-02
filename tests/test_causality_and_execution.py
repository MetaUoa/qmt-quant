from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from qmt_quant.backtest import _stamp_tax_rate, rebalance_schedule, run_backtest
from qmt_quant.config import CostConfig, StrategyConfig
from qmt_quant.reference_data import ReferenceData


def _basic(codes: list[str]) -> pd.DataFrame:
    rows = []
    for code in codes:
        exchange = "SSE" if code.endswith(".SH") else "SZSE"
        rows.append(
            {
                "ts_code": code,
                "exchange": exchange,
                "list_date": "20100101",
                "delist_date": None,
            }
        )
    return pd.DataFrame(rows)


def _strict_reference(
    index: pd.DatetimeIndex,
    cfg: StrategyConfig,
    codes: list[str],
    *,
    up_ratio: float = 2.0,
    down_ratio: float = 0.5,
) -> ReferenceData:
    st_rows: list[dict] = []
    limit_rows: list[dict] = []
    for signal, execution in rebalance_schedule(index, cfg):
        st_rows.append({"trade_date": signal.strftime("%Y%m%d"), "ts_code": "__NONE__"})
        for code in codes:
            limit_rows.append(
                {
                    "trade_date": execution.strftime("%Y%m%d"),
                    "ts_code": code,
                    "pre_close": 10.0,
                    "up_limit": 10.0 * up_ratio,
                    "down_limit": 10.0 * down_ratio,
                }
            )
    return ReferenceData(_basic(codes), index, pd.DataFrame(st_rows), pd.DataFrame(limit_rows))


def test_future_price_mutation_cannot_change_past_results(
    synthetic_bars: dict[str, pd.DataFrame],
    permissive_strategy: StrategyConfig,
    low_costs: CostConfig,
):
    cutoff = pd.Timestamp("2021-07-01")
    original = run_backtest(synthetic_bars, "000905.SH", permissive_strategy, low_costs)

    mutated = deepcopy(synthetic_bars)
    for code, frame in mutated.items():
        mask = frame.index >= cutoff
        if code == "000905.SH":
            factor = 0.15
        else:
            factor = 7.0 if code == "CCC.SZ" else 0.20
        for field in ("open", "high", "low", "close", "preClose"):
            frame.loc[mask, field] = frame.loc[mask, field] * factor
        frame.loc[mask, "amount"] = frame.loc[mask, "amount"] * 3.0

    changed = run_backtest(mutated, "000905.SH", permissive_strategy, low_costs)

    pd.testing.assert_frame_equal(
        original.equity.loc[original.equity.index < cutoff],
        changed.equity.loc[changed.equity.index < cutoff],
        check_exact=True,
    )

    def past_trades(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return frame.loc[pd.to_datetime(frame["date"]) < cutoff].reset_index(drop=True)

    pd.testing.assert_frame_equal(past_trades(original.trades), past_trades(changed.trades), check_exact=True)


def test_strict_reference_rejects_missing_st_snapshot(
    synthetic_bars: dict[str, pd.DataFrame],
    permissive_strategy: StrategyConfig,
    low_costs: CostConfig,
):
    idx = synthetic_bars["000905.SH"].index
    codes = [c for c in synthetic_bars if c != "000905.SH"]
    ref = ReferenceData(_basic(codes), idx, st=pd.DataFrame(), limits=pd.DataFrame())
    with pytest.raises(ValueError, match="Missing historical ST snapshot"):
        run_backtest(
            synthetic_bars,
            "000905.SH",
            permissive_strategy,
            low_costs,
            reference=ref,
            strict_reference=True,
            limit_reference_bars=synthetic_bars,
        )


def test_strict_reference_rejects_missing_limit_snapshot(
    synthetic_bars: dict[str, pd.DataFrame],
    permissive_strategy: StrategyConfig,
    low_costs: CostConfig,
):
    idx = synthetic_bars["000905.SH"].index
    codes = [c for c in synthetic_bars if c != "000905.SH"]
    st_rows = [
        {"trade_date": signal.strftime("%Y%m%d"), "ts_code": "__NONE__"}
        for signal, _ in rebalance_schedule(idx, permissive_strategy)
    ]
    ref = ReferenceData(_basic(codes), idx, st=pd.DataFrame(st_rows), limits=pd.DataFrame())
    with pytest.raises(ValueError, match="Missing daily price-limit snapshot"):
        run_backtest(
            synthetic_bars,
            "000905.SH",
            permissive_strategy,
            low_costs,
            reference=ref,
            strict_reference=True,
            limit_reference_bars=synthetic_bars,
        )


def test_strict_reference_requires_point_in_time_reference(
    synthetic_bars: dict[str, pd.DataFrame],
    permissive_strategy: StrategyConfig,
    low_costs: CostConfig,
):
    with pytest.raises(ValueError, match="requires point-in-time ReferenceData"):
        run_backtest(
            synthetic_bars,
            "000905.SH",
            permissive_strategy,
            low_costs,
            strict_reference=True,
            limit_reference_bars=synthetic_bars,
        )


def test_strict_reference_requires_raw_limit_bars(
    synthetic_bars: dict[str, pd.DataFrame],
    permissive_strategy: StrategyConfig,
    low_costs: CostConfig,
):
    idx = synthetic_bars["000905.SH"].index
    codes = [c for c in synthetic_bars if c != "000905.SH"]
    ref = _strict_reference(idx, permissive_strategy, codes)
    with pytest.raises(ValueError, match="requires unadjusted QMT limit_reference_bars"):
        run_backtest(
            synthetic_bars,
            "000905.SH",
            permissive_strategy,
            low_costs,
            reference=ref,
            strict_reference=True,
        )


def test_suspended_stock_is_never_bought(
    synthetic_bars: dict[str, pd.DataFrame],
    permissive_strategy: StrategyConfig,
    low_costs: CostConfig,
):
    bars = deepcopy(synthetic_bars)
    bars["AAA.SZ"]["suspendFlag"] = 1.0
    result = run_backtest(bars, "000905.SH", permissive_strategy, low_costs)
    buys = result.trades[result.trades["side"] == "BUY"] if not result.trades.empty else result.trades
    assert "AAA.SZ" not in set(buys.get("code", []))
    assert result.metrics["blocked_suspended"] > 0


def test_limit_up_reference_blocks_buy_in_strict_mode(
    synthetic_bars: dict[str, pd.DataFrame],
    permissive_strategy: StrategyConfig,
    low_costs: CostConfig,
):
    idx = synthetic_bars["000905.SH"].index
    codes = [c for c in synthetic_bars if c != "000905.SH"]
    ref = _strict_reference(idx, permissive_strategy, codes, up_ratio=1.10, down_ratio=0.90)

    raw = {code: frame[["open", "close", "preClose"]].copy() for code, frame in synthetic_bars.items()}
    schedule = rebalance_schedule(idx, permissive_strategy)
    for _, execution in schedule:
        for code in codes:
            prev = raw[code].at[execution, "preClose"]
            if np.isfinite(prev) and prev > 0:
                raw[code].at[execution, "open"] = prev * 1.10

    result = run_backtest(
        synthetic_bars,
        "000905.SH",
        permissive_strategy,
        low_costs,
        reference=ref,
        strict_reference=True,
        limit_reference_bars=raw,
    )
    assert result.metrics["blocked_limit_buys"] > 0
    assert result.trades.empty or not (result.trades["side"] == "BUY").any()


def test_trade_shares_are_board_lot_multiples_and_signal_precedes_trade(
    synthetic_bars: dict[str, pd.DataFrame],
    permissive_strategy: StrategyConfig,
    low_costs: CostConfig,
):
    result = run_backtest(synthetic_bars, "000905.SH", permissive_strategy, low_costs)
    assert not result.trades.empty
    assert (result.trades["shares"] % low_costs.lot_size == 0).all()
    assert (pd.to_datetime(result.trades["signal_date"]) < pd.to_datetime(result.trades["date"])).all()
    assert (result.equity["cash"] >= -1e-8).all()


def test_stamp_tax_rate_switch_date():
    assert _stamp_tax_rate(pd.Timestamp("2023-08-27")) == pytest.approx(0.0010)
    assert _stamp_tax_rate(pd.Timestamp("2023-08-28")) == pytest.approx(0.0005)
    assert _stamp_tax_rate(pd.Timestamp("2025-12-31")) == pytest.approx(0.0005)

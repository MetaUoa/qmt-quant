from __future__ import annotations

import pandas as pd

import qmt_quant.backtest as backtest
import qmt_quant.backtest_sell_execution as sell_execution
from qmt_quant.backtest_sell_execution import evaluate_sell_execution as real_evaluate_sell
from qmt_quant.config import CostConfig, StrategyConfig


class _FakeGuard:
    def __init__(self, *, halted: bool = False, limit_blocked: bool = False) -> None:
        self.halted = halted
        self.sell_limit_blocked = limit_blocked
        self.calls: list[tuple[str, str, str | None]] = []

    def is_halted(self, ts: pd.Timestamp, code: str) -> bool:
        self.calls.append(("halt", code, None))
        return self.halted

    def limit_blocked(self, ts: pd.Timestamp, code: str, side: str) -> bool:
        self.calls.append(("limit", code, side))
        return self.sell_limit_blocked


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


def test_sell_decision_preserves_t1_halt_limit_fill_short_circuit(monkeypatch) -> None:
    ts = pd.Timestamp("2025-01-06")
    cost = CostConfig(fill_probability=1.0)

    guard = _FakeGuard()
    decision = real_evaluate_sell(
        last_buy_date=ts,
        execution_date=ts,
        code="AAA.SZ",
        cost=cost,
        guard=guard,  # type: ignore[arg-type]
    )
    assert decision.blocked_t1_sells == 1
    assert not decision.ready
    assert guard.calls == []

    guard = _FakeGuard(halted=True)
    decision = real_evaluate_sell(
        last_buy_date=pd.Timestamp("2025-01-03"),
        execution_date=ts,
        code="AAA.SZ",
        cost=cost,
        guard=guard,  # type: ignore[arg-type]
    )
    assert decision.blocked_suspended == 1
    assert guard.calls == [("halt", "AAA.SZ", None)]

    guard = _FakeGuard(limit_blocked=True)
    decision = real_evaluate_sell(
        last_buy_date=pd.Timestamp("2025-01-03"),
        execution_date=ts,
        code="AAA.SZ",
        cost=cost,
        guard=guard,  # type: ignore[arg-type]
    )
    assert decision.blocked_limit_sells == 1
    assert guard.calls == [
        ("halt", "AAA.SZ", None),
        ("limit", "AAA.SZ", "SELL"),
    ]

    monkeypatch.setattr(sell_execution, "deterministic_fill", lambda *args, **kwargs: False)
    guard = _FakeGuard()
    decision = real_evaluate_sell(
        last_buy_date=pd.Timestamp("2025-01-03"),
        execution_date=ts,
        code="AAA.SZ",
        cost=cost,
        guard=guard,  # type: ignore[arg-type]
    )
    assert decision.blocked_random_fill == 1
    assert not decision.ready
    assert guard.calls == [
        ("halt", "AAA.SZ", None),
        ("limit", "AAA.SZ", "SELL"),
    ]

    monkeypatch.setattr(sell_execution, "deterministic_fill", lambda *args, **kwargs: True)
    decision = real_evaluate_sell(
        last_buy_date=pd.Timestamp("2025-01-03"),
        execution_date=ts,
        code="AAA.SZ",
        cost=cost,
        guard=_FakeGuard(),  # type: ignore[arg-type]
    )
    assert decision.ready
    assert decision.blocked_t1_sells == 0
    assert decision.blocked_suspended == 0
    assert decision.blocked_limit_sells == 0
    assert decision.blocked_random_fill == 0


def test_run_backtest_routes_sell_gate_through_decision_helper(monkeypatch) -> None:
    index = pd.bdate_range("2025-01-02", periods=14)
    bars = {
        "AAA.SZ": _flat_frame(index, 10.0),
        "000905.SH": _flat_frame(index, 100.0),
    }
    score = pd.DataFrame(1.0, index=index, columns=["AAA.SZ"])
    risk_on = pd.Series(False, index=index)
    risk_on.iloc[:8] = True
    costs = CostConfig(
        initial_cash=100_000.0,
        commission_rate=0.00025,
        min_commission=5.0,
        slippage_bps=0.0,
        fill_probability=1.0,
    )
    calls: list[tuple[str, pd.Timestamp | None, pd.Timestamp]] = []

    def tracked_sell_decision(**kwargs):
        calls.append(
            (
                str(kwargs["code"]),
                kwargs["last_buy_date"],
                pd.Timestamp(kwargs["execution_date"]),
            )
        )
        return real_evaluate_sell(**kwargs)

    monkeypatch.setattr(backtest, "evaluate_sell_execution", tracked_sell_decision)
    result = backtest.run_backtest(
        bars,
        "000905.SH",
        _strategy(),
        costs,
        score_override=score,
        risk_on_override=risk_on,
    )

    assert calls
    assert all(code == "AAA.SZ" for code, _, _ in calls)
    assert "SELL" in set(result.trades["side"])
    assert result.metrics["blocked_t1_sells"] == 0

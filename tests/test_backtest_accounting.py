from __future__ import annotations

import pytest

from qmt_quant.backtest_execution import settle_buy, settle_sell
from qmt_quant.config import CostConfig


def test_sell_settlement_matches_current_inline_accounting_formula() -> None:
    cost = CostConfig(commission_rate=0.001, min_commission=5.0)
    settlement = settle_sell(
        cash=1000.0,
        current_shares=500,
        quantity=200,
        execution_price=10.0,
        cost=cost,
        stamp_tax_rate=0.001,
    )
    notional = 200 * 10.0
    fee = max(5.0, notional * 0.001)
    tax = notional * 0.001
    assert settlement.notional == pytest.approx(notional)
    assert settlement.commission == pytest.approx(fee)
    assert settlement.stamp_tax == pytest.approx(tax)
    assert settlement.ending_cash == pytest.approx(1000.0 + notional - fee - tax)
    assert settlement.ending_shares == 300


def test_buy_settlement_matches_current_inline_accounting_formula() -> None:
    cost = CostConfig(commission_rate=0.001, min_commission=5.0)
    settlement = settle_buy(
        cash=5000.0,
        current_shares=100,
        quantity=200,
        execution_price=10.0,
        cost=cost,
    )
    notional = 200 * 10.0
    fee = max(5.0, notional * 0.001)
    assert settlement.notional == pytest.approx(notional)
    assert settlement.commission == pytest.approx(fee)
    assert settlement.ending_cash == pytest.approx(5000.0 - notional - fee)
    assert settlement.ending_shares == 300


def test_accounting_helpers_do_not_apply_execution_or_t1_policy() -> None:
    cost = CostConfig(commission_rate=0.0, min_commission=0.0)
    sell = settle_sell(
        cash=0.0,
        current_shares=100,
        quantity=100,
        execution_price=8.0,
        cost=cost,
        stamp_tax_rate=0.0005,
    )
    buy = settle_buy(
        cash=1000.0,
        current_shares=0,
        quantity=100,
        execution_price=8.0,
        cost=cost,
    )
    assert sell.ending_shares == 0
    assert buy.ending_shares == 100

from __future__ import annotations

from datetime import date

import pytest

from qmt_quant.raw_ledger import (
    CorporateActionEvent,
    RawExecutionLedger,
    RawTradeFill,
    require_unadjusted_price_provenance,
)


SOURCE = "a" * 64
PRICE_SOURCE = "unadjusted-daily-bars"


def _fill(**overrides) -> RawTradeFill:
    values = {
        "fill_id": "fill-1",
        "trade_date": date(2024, 1, 2),
        "code": "000001.SZ",
        "side": "BUY",
        "shares": 100,
        "raw_price": 10.0,
        "source_sha256": SOURCE,
        "price_source": PRICE_SOURCE,
        "adjustment_mode": "raw",
    }
    values.update(overrides)
    return RawTradeFill(**values)


def test_raw_trade_uses_historical_fee_regime_and_preserves_cash_shares() -> None:
    ledger = RawExecutionLedger(initial_cash=100_000.0)
    row = ledger.apply_trade(
        _fill(
            fill_id="buy-1",
            trade_date=date(2022, 4, 28),
            shares=1000,
        ),
        broker_commission_rate=0.00025,
        min_broker_commission=5.0,
    )
    assert row["fee_regime"]["effective_from"] == "2015-08-01"
    assert row["fees"]["transfer_fee"] == pytest.approx(0.2)
    assert row["price_provenance"]["adjustment_mode"] == "raw"
    assert ledger.positions == {"000001.SZ": 1000}
    assert ledger.cash == pytest.approx(89_994.8)


def test_sell_boundary_uses_reduced_2023_stamp_and_exchange_rates() -> None:
    ledger = RawExecutionLedger(initial_cash=0.0, positions={"600000.SH": 1000})
    row = ledger.apply_trade(
        _fill(
            fill_id="sell-1",
            trade_date=date(2023, 8, 28),
            code="600000.SH",
            side="SELL",
            shares=1000,
        ),
        broker_commission_rate=0.00025,
        min_broker_commission=5.0,
    )
    assert row["fees"]["stamp_tax"] == pytest.approx(5.0)
    assert row["fees"]["exchange_handling_fee"] == pytest.approx(0.341)
    assert row["fees"]["embedded_regulatory_fee"] == pytest.approx(0.541)
    assert ledger.positions == {}
    assert ledger.cash == pytest.approx(9_989.9)


def test_trade_fill_rejects_adjusted_execution_price_provenance() -> None:
    ledger = RawExecutionLedger(initial_cash=10_000.0)
    with pytest.raises(RuntimeError, match="adjusted prices are forbidden"):
        ledger.apply_trade(
            _fill(adjustment_mode="front"),
            broker_commission_rate=0.00025,
            min_broker_commission=5.0,
        )
    assert ledger.positions == {}
    assert ledger.cash == pytest.approx(10_000.0)


def test_corporate_action_applies_exact_share_and_cash_delta_once() -> None:
    ledger = RawExecutionLedger(initial_cash=100.0, positions={"000001.SZ": 1000})
    event = CorporateActionEvent(
        event_id="ca-2024-1",
        action_date=date(2024, 6, 1),
        code="000001.SZ",
        share_multiplier=1.2,
        cash_per_pre_action_share=0.1,
        cash_in_lieu=0.0,
        source_sha256=SOURCE,
    )
    ledger.apply_corporate_action(event)
    assert ledger.positions == {"000001.SZ": 1200}
    assert ledger.cash == pytest.approx(200.0)
    with pytest.raises(RuntimeError, match="duplicate"):
        ledger.apply_corporate_action(event)


def test_fractional_corporate_action_requires_explicit_registrar_settlement() -> None:
    ledger = RawExecutionLedger(initial_cash=0.0, positions={"000001.SZ": 101})
    with pytest.raises(RuntimeError, match="fractional"):
        ledger.apply_corporate_action(
            CorporateActionEvent(
                event_id="ca-frac",
                action_date=date(2024, 6, 1),
                code="000001.SZ",
                share_multiplier=1.1,
                cash_per_pre_action_share=0.0,
                cash_in_lieu=0.0,
                source_sha256=SOURCE,
            )
        )


def test_adjusted_price_provenance_is_rejected_for_execution_ledger() -> None:
    with pytest.raises(RuntimeError, match="adjusted prices are forbidden"):
        require_unadjusted_price_provenance(
            {"adjustment_mode": "front", "source_sha256": SOURCE, "source": "fixture"}
        )
    accepted = require_unadjusted_price_provenance(
        {"adjustment_mode": "none", "source_sha256": SOURCE, "source": "fixture"}
    )
    assert accepted["adjustment_mode"] == "none"


def test_mark_to_market_and_state_reconciliation_use_raw_prices_only() -> None:
    ledger = RawExecutionLedger(initial_cash=1000.0, positions={"600000.SH": 100})
    equity = ledger.mark_to_market(
        {"600000.SH": 8.5},
        price_provenance={
            "adjustment_mode": "raw",
            "source_sha256": SOURCE,
            "source": PRICE_SOURCE,
        },
    )
    assert equity == pytest.approx(1850.0)
    report = ledger.reconcile_state(
        observed_cash=1000.0,
        observed_positions={"600000.SH": 100},
    )
    assert report["passed"] is True
    drifted = ledger.reconcile_state(
        observed_cash=1000.0,
        observed_positions={"600000.SH": 200},
    )
    assert drifted["passed"] is False

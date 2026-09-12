from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import math
import re
from typing import Mapping

from .historical_fees import fee_regime_for, historical_fee_schedule
from .transaction_costs import fee_breakdown


RAW_LEDGER_SCHEMA = "qmt-raw-execution-ledger-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RAW_ADJUSTMENT_MODES = {"none", "raw", "unadjusted"}


def _require_sha256(value: object, *, name: str) -> str:
    token = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(token):
        raise ValueError(f"{name} must be an exact lowercase 64-hex SHA256")
    return token


def _finite(value: object, *, name: str, non_negative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if non_negative and number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _strict_shares(value: object, *, name: str, allow_zero: bool = True) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        shares = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        shares = int(value)
    else:
        raise ValueError(f"{name} must be an integer")
    if shares < 0 or (shares == 0 and not allow_zero):
        qualifier = "positive" if not allow_zero else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return shares


def require_unadjusted_price_provenance(metadata: Mapping[str, object]) -> dict[str, str]:
    mode = str(metadata.get("adjustment_mode", metadata.get("adjustment", ""))).strip().lower()
    if mode not in _RAW_ADJUSTMENT_MODES:
        raise RuntimeError(
            "raw execution ledger requires unadjusted/raw price provenance; adjusted prices are forbidden"
        )
    source_sha256 = _require_sha256(metadata.get("source_sha256", ""), name="price_source.source_sha256")
    source = str(metadata.get("source", "")).strip()
    if not source:
        raise RuntimeError("raw execution ledger requires named price source provenance")
    return {"adjustment_mode": mode, "source_sha256": source_sha256, "source": source}


@dataclass(frozen=True)
class RawTradeFill:
    fill_id: str
    trade_date: date
    code: str
    side: str
    shares: int
    raw_price: float
    source_sha256: str

    def validate(self) -> None:
        if not str(self.fill_id).strip():
            raise ValueError("fill_id is required")
        if not str(self.code).strip():
            raise ValueError("code is required")
        if str(self.side).upper() not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        _strict_shares(self.shares, name="shares", allow_zero=False)
        if _finite(self.raw_price, name="raw_price", non_negative=True) <= 0:
            raise ValueError("raw_price must be positive")
        _require_sha256(self.source_sha256, name="fill.source_sha256")


@dataclass(frozen=True)
class CorporateActionEvent:
    event_id: str
    action_date: date
    code: str
    share_multiplier: float
    cash_per_pre_action_share: float
    cash_in_lieu: float
    source_sha256: str

    def validate(self) -> None:
        if not str(self.event_id).strip():
            raise ValueError("event_id is required")
        if not str(self.code).strip():
            raise ValueError("code is required")
        multiplier = _finite(self.share_multiplier, name="share_multiplier", non_negative=True)
        if multiplier <= 0:
            raise ValueError("share_multiplier must be positive")
        _finite(
            self.cash_per_pre_action_share,
            name="cash_per_pre_action_share",
            non_negative=True,
        )
        _finite(self.cash_in_lieu, name="cash_in_lieu", non_negative=True)
        _require_sha256(self.source_sha256, name="corporate_action.source_sha256")


class RawExecutionLedger:
    """Cash/share ledger that never uses adjusted prices for execution accounting.

    Corporate-action inputs are already-settled registrar/accounting facts: cash per
    pre-action share is the cash actually credited by the source ledger, and a share
    multiplier must produce an integer settled share count. Ambiguous fractional
    entitlements fail closed instead of inventing rounding or tax treatment.
    """

    def __init__(self, *, initial_cash: float, positions: Mapping[str, int] | None = None) -> None:
        self._initial_cash = _finite(initial_cash, name="initial_cash", non_negative=True)
        self.cash = self._initial_cash
        self._initial_positions = {
            str(code): _strict_shares(shares, name=f"positions.{code}")
            for code, shares in (positions or {}).items()
            if _strict_shares(shares, name=f"positions.{code}") > 0
        }
        self.positions = dict(self._initial_positions)
        self.entries: list[dict[str, object]] = []
        self._applied_event_ids: set[str] = set()

    def _reserve_event_id(self, event_id: str) -> None:
        token = str(event_id).strip()
        if not token:
            raise ValueError("event_id is required")
        if token in self._applied_event_ids:
            raise RuntimeError(f"duplicate raw-ledger event id: {token}")
        self._applied_event_ids.add(token)

    def apply_trade(
        self,
        fill: RawTradeFill,
        *,
        broker_commission_rate: float,
        min_broker_commission: float,
        commission_includes_exchange_and_management: bool = True,
    ) -> dict[str, object]:
        fill.validate()
        self._reserve_event_id(fill.fill_id)
        code = str(fill.code)
        side = str(fill.side).upper()
        shares = _strict_shares(fill.shares, name="shares", allow_zero=False)
        price = _finite(fill.raw_price, name="raw_price", non_negative=True)
        before_shares = int(self.positions.get(code, 0))
        cash_before = float(self.cash)
        notional = float(shares * price)
        schedule = historical_fee_schedule(
            fill.trade_date,
            broker_commission_rate=broker_commission_rate,
            min_broker_commission=min_broker_commission,
            commission_includes_exchange_and_management=commission_includes_exchange_and_management,
        )
        fees = fee_breakdown(side=side, notional=notional, schedule=schedule)
        if side == "BUY":
            required = notional + fees.total_cash_fee
            if required > cash_before + 1e-9:
                raise RuntimeError("raw ledger BUY would create negative cash")
            after_shares = before_shares + shares
            cash_after = cash_before - required
        else:
            if shares > before_shares:
                raise RuntimeError("raw ledger SELL exceeds settled shares")
            after_shares = before_shares - shares
            cash_after = cash_before + notional - fees.total_cash_fee

        self.cash = float(cash_after)
        if after_shares > 0:
            self.positions[code] = after_shares
        else:
            self.positions.pop(code, None)
        regime = fee_regime_for(fill.trade_date)
        entry: dict[str, object] = {
            "schema": RAW_LEDGER_SCHEMA,
            "event_type": "TRADE",
            "event_id": fill.fill_id,
            "event_date": fill.trade_date.isoformat(),
            "code": code,
            "side": side,
            "shares": shares,
            "raw_price": price,
            "notional": notional,
            "source_sha256": _require_sha256(fill.source_sha256, name="fill.source_sha256"),
            "cash_before": cash_before,
            "cash_after": float(cash_after),
            "shares_before": before_shares,
            "shares_after": after_shares,
            "fee_regime": {
                "effective_from": regime.effective_from.isoformat(),
                "authority_key": regime.authority_key,
            },
            "fees": asdict(fees),
        }
        self.entries.append(entry)
        return dict(entry)

    def apply_corporate_action(self, event: CorporateActionEvent) -> dict[str, object]:
        event.validate()
        self._reserve_event_id(event.event_id)
        code = str(event.code)
        before_shares = int(self.positions.get(code, 0))
        cash_before = float(self.cash)
        multiplier = _finite(event.share_multiplier, name="share_multiplier", non_negative=True)
        calculated_shares = before_shares * multiplier
        rounded_shares = int(round(calculated_shares))
        if not math.isclose(calculated_shares, rounded_shares, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                "corporate action produces fractional settled shares; explicit registrar settlement is required"
            )
        cash_delta = (
            before_shares
            * _finite(
                event.cash_per_pre_action_share,
                name="cash_per_pre_action_share",
                non_negative=True,
            )
            + _finite(event.cash_in_lieu, name="cash_in_lieu", non_negative=True)
        )
        self.cash = cash_before + cash_delta
        if rounded_shares > 0:
            self.positions[code] = rounded_shares
        else:
            self.positions.pop(code, None)
        entry: dict[str, object] = {
            "schema": RAW_LEDGER_SCHEMA,
            "event_type": "CORPORATE_ACTION",
            "event_id": event.event_id,
            "event_date": event.action_date.isoformat(),
            "code": code,
            "source_sha256": _require_sha256(
                event.source_sha256, name="corporate_action.source_sha256"
            ),
            "share_multiplier": multiplier,
            "cash_per_pre_action_share": float(event.cash_per_pre_action_share),
            "cash_in_lieu": float(event.cash_in_lieu),
            "cash_delta": float(cash_delta),
            "cash_before": cash_before,
            "cash_after": float(self.cash),
            "shares_before": before_shares,
            "shares_after": rounded_shares,
        }
        self.entries.append(entry)
        return dict(entry)

    def mark_to_market(
        self,
        raw_close: Mapping[str, float],
        *,
        price_provenance: Mapping[str, object],
    ) -> float:
        require_unadjusted_price_provenance(price_provenance)
        market_value = 0.0
        for code, shares in self.positions.items():
            if code not in raw_close:
                raise RuntimeError(f"missing raw close for held position {code}")
            price = _finite(raw_close[code], name=f"raw_close.{code}", non_negative=True)
            if price <= 0:
                raise RuntimeError(f"invalid raw close for held position {code}")
            market_value += shares * price
        return float(self.cash + market_value)

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": RAW_LEDGER_SCHEMA,
            "cash": float(self.cash),
            "positions": {code: self.positions[code] for code in sorted(self.positions)},
            "entry_count": len(self.entries),
            "ledger_sha256": self.ledger_sha256(),
        }

    def ledger_sha256(self) -> str:
        payload = {
            "schema": RAW_LEDGER_SCHEMA,
            "initial_cash": self._initial_cash,
            "initial_positions": {
                code: self._initial_positions[code] for code in sorted(self._initial_positions)
            },
            "entries": self.entries,
        }
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def reconcile_state(
        self,
        *,
        observed_cash: float,
        observed_positions: Mapping[str, int],
        cash_tolerance: float = 0.01,
    ) -> dict[str, object]:
        tolerance = _finite(cash_tolerance, name="cash_tolerance", non_negative=True)
        cash_value = _finite(observed_cash, name="observed_cash", non_negative=True)
        normalized_positions = {
            str(code): _strict_shares(shares, name=f"observed_positions.{code}")
            for code, shares in observed_positions.items()
            if _strict_shares(shares, name=f"observed_positions.{code}") > 0
        }
        cash_delta = cash_value - float(self.cash)
        position_match = normalized_positions == self.positions
        return {
            "passed": abs(cash_delta) <= tolerance and position_match,
            "expected_cash": float(self.cash),
            "observed_cash": cash_value,
            "cash_delta": float(cash_delta),
            "cash_tolerance": tolerance,
            "expected_positions": {code: self.positions[code] for code in sorted(self.positions)},
            "observed_positions": {
                code: normalized_positions[code] for code in sorted(normalized_positions)
            },
            "position_match": position_match,
            "ledger_sha256": self.ledger_sha256(),
        }

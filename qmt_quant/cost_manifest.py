from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .config import CostConfig
from .historical_fees import historical_fee_manifest


COST_MANIFEST_SCHEMA = "qmt-research-cost-manifest-v1"


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    raw = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_cost_manifest(
    cost: CostConfig,
    *,
    execution_price_mode: str = "raw_unadjusted",
    commission_includes_exchange_and_management: bool = True,
) -> dict[str, object]:
    mode = str(execution_price_mode).strip().lower()
    if mode != "raw_unadjusted":
        raise RuntimeError("research cost manifest requires raw_unadjusted execution accounting")
    core: dict[str, object] = {
        "schema": COST_MANIFEST_SCHEMA,
        "execution_price_mode": mode,
        "cost_config": asdict(cost),
        "historical_a_share_fees": historical_fee_manifest(),
        "commission_includes_exchange_and_management": bool(
            commission_includes_exchange_and_management
        ),
    }
    return {**core, "manifest_sha256": _canonical_sha256(core)}


def validate_cost_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    if str(payload.get("schema", "")) != COST_MANIFEST_SCHEMA:
        raise RuntimeError(f"cost manifest requires schema {COST_MANIFEST_SCHEMA}")
    if str(payload.get("execution_price_mode", "")) != "raw_unadjusted":
        raise RuntimeError("cost manifest must use raw_unadjusted execution accounting")
    cost_config = payload.get("cost_config")
    historical = payload.get("historical_a_share_fees")
    if not isinstance(cost_config, Mapping) or not isinstance(historical, Mapping):
        raise RuntimeError("cost manifest requires cost_config and historical fee manifest")
    core = {
        "schema": COST_MANIFEST_SCHEMA,
        "execution_price_mode": "raw_unadjusted",
        "cost_config": dict(cost_config),
        "historical_a_share_fees": dict(historical),
        "commission_includes_exchange_and_management": bool(
            payload.get("commission_includes_exchange_and_management", True)
        ),
    }
    expected = _canonical_sha256(core)
    observed = str(payload.get("manifest_sha256", "")).strip().lower()
    if observed != expected:
        raise RuntimeError("cost manifest SHA256 is invalid")
    return {**core, "manifest_sha256": observed}


def write_cost_manifest(
    path: str | Path,
    cost: CostConfig,
    *,
    commission_includes_exchange_and_management: bool = True,
) -> dict[str, object]:
    payload = build_cost_manifest(
        cost,
        commission_includes_exchange_and_management=commission_includes_exchange_and_management,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

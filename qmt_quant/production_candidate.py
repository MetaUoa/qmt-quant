from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .config import StrategyConfig
from .holdout import FrozenCandidate, verify_candidate_manifest


PRODUCTION_SCHEMA = "qmt-v5-production-candidate-v1"


@dataclass(frozen=True)
class StrategySource:
    kind: str
    sha256: str
    strategy: StrategyConfig | None = None
    candidate: FrozenCandidate | None = None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_legacy_strategy_config(path: str | Path) -> StrategySource:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("strategy config must be a JSON object")
    allowed = set(StrategyConfig.__dataclass_fields__)
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"strategy config contains unknown fields: {', '.join(unknown)}")
    return StrategySource(
        kind="legacy_strategy_config",
        sha256=_sha256_bytes(raw),
        strategy=StrategyConfig(**dict(payload)),
    )


def load_production_candidate_bundle(path: str | Path) -> StrategySource:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("production candidate bundle must be a JSON object")
    if str(payload.get("schema", "")) != PRODUCTION_SCHEMA:
        raise RuntimeError(f"production candidate schema must be {PRODUCTION_SCHEMA}")
    for key in ("basic_alpha_gate_passed", "holdout_unlocked", "holdout_passed"):
        if payload.get(key) is not True:
            raise RuntimeError(f"production candidate is not eligible: {key} is not true")
    frozen = payload.get("frozen")
    if not isinstance(frozen, Mapping):
        raise ValueError("production candidate bundle requires frozen manifest")
    candidate = verify_candidate_manifest(frozen)
    declared = str(payload.get("candidate_sha256", ""))
    if declared != candidate.fingerprint():
        raise RuntimeError("production candidate SHA256 does not match frozen candidate")
    return StrategySource(
        kind="v5_frozen_candidate",
        sha256=candidate.fingerprint(),
        candidate=candidate,
    )


def strategy_source_manifest(source: StrategySource) -> dict[str, object]:
    payload: dict[str, object] = {"kind": source.kind, "sha256": source.sha256}
    if source.strategy is not None:
        payload["strategy"] = asdict(source.strategy)
    if source.candidate is not None:
        payload["candidate"] = source.candidate.canonical_payload()
    return payload

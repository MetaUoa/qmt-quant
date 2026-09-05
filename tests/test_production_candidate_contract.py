from __future__ import annotations

import json
import sys

import pytest

import generate_live_targets
from qmt_quant.holdout import FrozenCandidate, freeze_candidate_manifest
from qmt_quant.production_candidate import (
    PRODUCTION_SCHEMA,
    load_legacy_strategy_config,
    load_production_candidate_bundle,
)


def _candidate() -> FrozenCandidate:
    return FrozenCandidate(
        name="unit-test",
        research_data_end="2025-12-31",
        neutralization_variant="industry_size_liquidity",
        weights={"short_reversal": 1.0},
        top_n=8,
        rebalance_days=5,
        execution_delay_sessions=1,
        min_price=3.0,
        min_amount=20_000_000.0,
        min_listing_sessions=120,
    )


def test_live_target_cli_has_no_implicit_v3_strategy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate_live_targets.py"])
    with pytest.raises(SystemExit):
        generate_live_targets.parse_args()


def test_missing_legacy_strategy_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_legacy_strategy_config(tmp_path / "missing.json")


def test_unknown_legacy_strategy_fields_are_rejected(tmp_path):
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps({"top_n": 8, "typo_field": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_legacy_strategy_config(path)


def test_production_candidate_requires_all_release_gates(tmp_path):
    candidate = _candidate()
    path = tmp_path / "candidate.json"
    payload = {
        "schema": PRODUCTION_SCHEMA,
        "basic_alpha_gate_passed": True,
        "holdout_unlocked": False,
        "holdout_passed": False,
        "candidate_sha256": candidate.fingerprint(),
        "frozen": freeze_candidate_manifest(candidate),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="holdout_unlocked"):
        load_production_candidate_bundle(path)


def test_production_candidate_sha_is_verified(tmp_path):
    candidate = _candidate()
    path = tmp_path / "candidate.json"
    payload = {
        "schema": PRODUCTION_SCHEMA,
        "basic_alpha_gate_passed": True,
        "holdout_unlocked": True,
        "holdout_passed": True,
        "candidate_sha256": candidate.fingerprint(),
        "frozen": freeze_candidate_manifest(candidate),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    source = load_production_candidate_bundle(path)
    assert source.kind == "v5_frozen_candidate"
    assert source.sha256 == candidate.fingerprint()
    assert source.candidate == candidate
    assert source.strategy is None

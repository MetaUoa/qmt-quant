from __future__ import annotations

from datetime import date
import hashlib
import json

import pandas as pd
import pytest

import qmt_quant.live_safety as live_safety
from qmt_quant.acceptance_lineage import lineage_binding_sha256
from qmt_quant.run_manifest import RUN_MANIFEST_SCHEMA, canonical_sha256


SHA = "a" * 64
EVIDENCE_SHA = "c" * 64
ARTIFACT_SHA = "d" * 64


def _write_targets(
    tmp_path,
    *,
    signal_date="2026-09-04",
    expected_execution_session="2026-09-07",
    expires_after_session="2026-09-07",
    source=True,
    csv_signal_date=None,
    csv_expected_execution_session=None,
    csv_expires_after_session=None,
    csv_sha=SHA,
):
    targets = tmp_path / "target_weights.csv"
    diagnostics = tmp_path / "signal_diagnostics.json"
    pd.DataFrame(
        {
            "signal_date": [csv_signal_date or signal_date],
            "expected_execution_session": [
                csv_expected_execution_session or expected_execution_session
            ],
            "expires_after_session": [csv_expires_after_session or expires_after_session],
            "strategy_sha256": [csv_sha],
            "code": ["000001.SZ"],
            "target_weight": [1.0],
        }
    ).to_csv(targets, index=False)
    payload = {
        "signal_date": signal_date,
        "expected_execution_session": expected_execution_session,
        "expires_after_session": expires_after_session,
        "selected_count": 1,
    }
    if source:
        payload["strategy_source"] = {"kind": "legacy_strategy_config", "sha256": SHA}
    diagnostics.write_text(json.dumps(payload), encoding="utf-8")
    return targets, diagnostics


def _acceptance_payload(*, grade="B", strategy_sha=SHA, binding_valid=True):
    evidence = {
        "backtest": EVIDENCE_SHA,
        "walk_forward": EVIDENCE_SHA,
        "folds": EVIDENCE_SHA,
        "stress": EVIDENCE_SHA,
    }
    artifacts = {
        "strategy_source": ARTIFACT_SHA,
        "config": ARTIFACT_SHA,
        "data_lineage": ARTIFACT_SHA,
        "engine_manifest": ARTIFACT_SHA,
        "dependency_lock": ARTIFACT_SHA,
        "cost_manifest": ARTIFACT_SHA,
    }
    run_core = {
        "schema": RUN_MANIFEST_SCHEMA,
        "git": {"commit_sha1": "1" * 40, "tree_sha1": "2" * 40},
        "python": {
            "implementation": "CPython",
            "version": "3.12.10",
            "major_minor": "3.12",
        },
        "strategy_sha256": strategy_sha,
        "evidence_sha256": evidence,
        "artifact_sha256": artifacts,
    }
    run_manifest = {**run_core, "run_id": canonical_sha256(run_core)}
    run_id = str(run_manifest["run_id"])
    binding = lineage_binding_sha256(
        strategy_sha256=strategy_sha,
        evidence_sha256=evidence,
        artifact_sha256=artifacts,
        run_id=run_id,
    )
    if not binding_valid:
        binding = "e" * 64
    lineage = {
        "strategy_sha256": strategy_sha,
        "evidence_sha256": evidence,
        "artifact_sha256": artifacts,
        "run_id": run_id,
        "run_manifest": run_manifest,
        "binding_sha256": binding,
    }
    return {
        "schema": "qmt-acceptance-v4",
        "grade": grade,
        "strategy_sha256": strategy_sha,
        "evidence_sha256": evidence,
        "artifact_sha256": artifacts,
        "run_id": run_id,
        "run_manifest": run_manifest,
        "lineage": lineage,
    }


def test_next_trading_session_skips_non_session_dates():
    calendar = pd.DatetimeIndex(["2026-09-04", "2026-09-07", "2026-09-08"])
    assert live_safety.next_trading_session(calendar, "2026-09-04") == date(2026, 9, 7)


def test_live_targets_are_valid_on_expected_execution_session(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(tmp_path)
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 7))
    bundle = live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)
    assert bundle.signal_date == date(2026, 9, 4)
    assert bundle.expected_execution_session == date(2026, 9, 7)
    assert bundle.expires_after_session == date(2026, 9, 7)


def test_live_targets_reject_session_after_expiry(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(tmp_path)
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 8))
    with pytest.raises(RuntimeError, match="outside their execution window"):
        live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)


def test_live_targets_reject_execution_session_not_after_signal(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(
        tmp_path,
        expected_execution_session="2026-09-04",
        expires_after_session="2026-09-04",
    )
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 4))
    with pytest.raises(RuntimeError, match="must be after signal_date"):
        live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)


def test_live_targets_require_strategy_fingerprint(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(tmp_path, source=False)
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 7))
    with pytest.raises(RuntimeError, match="fingerprinted strategy_source"):
        live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)


def test_live_target_csv_sha_must_match_diagnostics(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(tmp_path, csv_sha="b" * 64)
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 7))
    with pytest.raises(RuntimeError, match="CSV strategy SHA256"):
        live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)


def test_live_target_csv_signal_date_must_match_diagnostics(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(tmp_path, csv_signal_date="2026-09-03")
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 7))
    with pytest.raises(RuntimeError, match="CSV signal_date"):
        live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)


def test_live_target_csv_execution_session_must_match_diagnostics(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(
        tmp_path,
        csv_expected_execution_session="2026-09-08",
    )
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 7))
    with pytest.raises(RuntimeError, match="CSV expected_execution_session"):
        live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)


def test_valid_live_target_bundle_returns_exact_sha_and_file_digest(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(tmp_path)
    expected_digest = hashlib.sha256(targets.read_bytes()).hexdigest()
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 7))
    bundle = live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)
    assert bundle.strategy_sha256 == SHA
    assert bundle.target_file_sha256 == expected_digest


def test_acceptance_must_bind_exact_strategy_sha(tmp_path):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(_acceptance_payload(grade="A", strategy_sha="b" * 64)),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="target strategy SHA256"):
        live_safety.validate_acceptance_for_strategy(acceptance, "C", SHA)


def test_live_acceptance_rejects_v2_report(tmp_path):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps({"schema": "qmt-acceptance-v2", "grade": "A", "strategy_sha256": SHA}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="schema qmt-acceptance-v4"):
        live_safety.validate_acceptance_for_strategy(acceptance, "C", SHA)


def test_live_acceptance_rejects_v3_report(tmp_path):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps({"schema": "qmt-acceptance-v3", "grade": "A", "strategy_sha256": SHA}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="schema qmt-acceptance-v4"):
        live_safety.validate_acceptance_for_strategy(acceptance, "C", SHA)


def test_live_acceptance_rejects_invalid_lineage_binding(tmp_path):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps(_acceptance_payload(binding_valid=False)), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding SHA256 is invalid"):
        live_safety.validate_acceptance_for_strategy(acceptance, "C", SHA)


def test_acceptance_matching_strategy_sha_and_lineage_passes(tmp_path):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps(_acceptance_payload()), encoding="utf-8")
    report = live_safety.validate_acceptance_for_strategy(acceptance, "C", SHA)
    assert report["grade"] == "B"

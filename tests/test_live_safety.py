from __future__ import annotations

from datetime import date
import hashlib
import json

import pandas as pd
import pytest

import qmt_quant.live_safety as live_safety


SHA = "a" * 64
EVIDENCE_SHA = "c" * 64


def _write_targets(
    tmp_path,
    *,
    signal_date="2026-09-06",
    source=True,
    csv_signal_date=None,
    csv_sha=SHA,
):
    targets = tmp_path / "target_weights.csv"
    diagnostics = tmp_path / "signal_diagnostics.json"
    pd.DataFrame(
        {
            "signal_date": [csv_signal_date or signal_date],
            "strategy_sha256": [csv_sha],
            "code": ["000001.SZ"],
            "target_weight": [1.0],
        }
    ).to_csv(targets, index=False)
    payload = {"signal_date": signal_date, "selected_count": 1}
    if source:
        payload["strategy_source"] = {"kind": "legacy_strategy_config", "sha256": SHA}
    diagnostics.write_text(json.dumps(payload), encoding="utf-8")
    return targets, diagnostics


def _acceptance_payload(*, grade="B", strategy_sha=SHA):
    return {
        "schema": "qmt-acceptance-v2",
        "grade": grade,
        "strategy_sha256": strategy_sha,
        "evidence_sha256": {
            "backtest": EVIDENCE_SHA,
            "walk_forward": EVIDENCE_SHA,
            "folds": EVIDENCE_SHA,
            "stress": EVIDENCE_SHA,
        },
    }


def test_live_targets_must_match_current_china_market_date(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(tmp_path, signal_date="2026-09-05")
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 6))
    with pytest.raises(RuntimeError, match="stale live targets"):
        live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)


def test_live_targets_require_strategy_fingerprint(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(tmp_path, source=False)
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 6))
    with pytest.raises(RuntimeError, match="fingerprinted strategy_source"):
        live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)


def test_live_target_csv_sha_must_match_diagnostics(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(tmp_path, csv_sha="b" * 64)
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 6))
    with pytest.raises(RuntimeError, match="CSV strategy SHA256"):
        live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)


def test_live_target_csv_signal_date_must_match_diagnostics(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(tmp_path, csv_signal_date="2026-09-05")
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 6))
    with pytest.raises(RuntimeError, match="CSV signal_date"):
        live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)


def test_valid_live_target_bundle_returns_exact_sha_and_file_digest(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(tmp_path)
    expected_digest = hashlib.sha256(targets.read_bytes()).hexdigest()
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 6))
    bundle = live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)
    assert bundle.signal_date == date(2026, 9, 6)
    assert bundle.strategy_sha256 == SHA
    assert bundle.target_file_sha256 == expected_digest


def test_acceptance_must_bind_exact_strategy_sha(tmp_path):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(_acceptance_payload(grade="A", strategy_sha="b" * 64)),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="exact target strategy SHA256"):
        live_safety.validate_acceptance_for_strategy(acceptance, "C", SHA)


def test_live_acceptance_rejects_unhashed_legacy_report(tmp_path):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps({"grade": "A", "strategy_sha256": SHA}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="schema qmt-acceptance-v2"):
        live_safety.validate_acceptance_for_strategy(acceptance, "C", SHA)


def test_live_acceptance_requires_all_evidence_hashes(tmp_path):
    payload = _acceptance_payload()
    del payload["evidence_sha256"]["stress"]
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stress"):
        live_safety.validate_acceptance_for_strategy(acceptance, "C", SHA)


def test_acceptance_matching_strategy_sha_and_evidence_hashes_passes(tmp_path):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps(_acceptance_payload()), encoding="utf-8")
    report = live_safety.validate_acceptance_for_strategy(acceptance, "C", SHA)
    assert report["grade"] == "B"

from __future__ import annotations

from datetime import date
import json

import pandas as pd
import pytest

import qmt_quant.live_safety as live_safety


SHA = "a" * 64


def _write_targets(tmp_path, *, signal_date="2026-09-06", source=True):
    targets = tmp_path / "target_weights.csv"
    diagnostics = tmp_path / "signal_diagnostics.json"
    pd.DataFrame({"code": ["000001.SZ"], "target_weight": [1.0]}).to_csv(targets, index=False)
    payload = {"signal_date": signal_date, "selected_count": 1}
    if source:
        payload["strategy_source"] = {"kind": "legacy_strategy_config", "sha256": SHA}
    diagnostics.write_text(json.dumps(payload), encoding="utf-8")
    return targets, diagnostics


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


def test_valid_live_target_bundle_returns_exact_sha(tmp_path, monkeypatch):
    targets, diagnostics = _write_targets(tmp_path)
    monkeypatch.setattr(live_safety, "china_market_date", lambda: date(2026, 9, 6))
    bundle = live_safety.validate_target_bundle(targets, diagnostics, require_current_session=True)
    assert bundle.signal_date == date(2026, 9, 6)
    assert bundle.strategy_sha256 == SHA


def test_acceptance_must_bind_exact_strategy_sha(tmp_path):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps({"grade": "A", "strategy_sha256": "b" * 64}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact target strategy SHA256"):
        live_safety.validate_acceptance_for_strategy(acceptance, "C", SHA)


def test_acceptance_matching_strategy_sha_passes(tmp_path):
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps({"grade": "B", "strategy_sha256": SHA}), encoding="utf-8")
    report = live_safety.validate_acceptance_for_strategy(acceptance, "C", SHA)
    assert report["grade"] == "B"

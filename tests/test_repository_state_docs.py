from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_uses_current_v5c_state_not_legacy_linear_version_claim():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "QMT Quant Research Suite V3-V7" not in text
    assert "V5-C pre-2026" in text
    assert "2026" in text and "盲化" in text
    assert "v5-c-pre2026-frozen-lineage-v1" in text
    assert "baostock==0.9.3" in text
    assert "V5 production scorer" in text


def test_release_manifest_has_dynamic_ci_claim_and_holdout_locked():
    payload = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "qmt-repository-state-v1"
    assert payload["holdout_2026_blinded"] is True
    assert payload["basic_alpha_gate"]["production_promotion_allowed"] is False
    assert payload["frozen_lineage"]["historical_shards"] == 20
    assert payload["frozen_lineage"]["max_parallel"] == 5
    assert payload["frozen_lineage"]["baostock"] == "0.9.3"
    assert payload["offline_verification"]["test_count"] == "dynamic; do not hardcode"
    assert "tests_passed" not in payload.get("offline_verification", {})

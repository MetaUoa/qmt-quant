from __future__ import annotations

import pytest

from qmt_quant.config import CostConfig
from qmt_quant.cost_manifest import build_cost_manifest, validate_cost_manifest


def test_cost_manifest_is_content_addressed_and_raw_execution_only() -> None:
    manifest = build_cost_manifest(CostConfig())
    assert manifest["execution_price_mode"] == "raw_unadjusted"
    assert len(str(manifest["manifest_sha256"])) == 64
    validated = validate_cost_manifest(manifest)
    assert validated["manifest_sha256"] == manifest["manifest_sha256"]


def test_cost_manifest_rejects_adjusted_execution_mode() -> None:
    with pytest.raises(RuntimeError, match="raw_unadjusted"):
        build_cost_manifest(CostConfig(), execution_price_mode="front_adjusted")


def test_cost_manifest_rejects_tampering() -> None:
    manifest = build_cost_manifest(CostConfig())
    tampered = dict(manifest)
    tampered["manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA256"):
        validate_cost_manifest(tampered)

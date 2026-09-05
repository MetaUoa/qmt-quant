from __future__ import annotations

import pytest

from qmt_quant.holdout_artifacts import (
    HoldoutArtifactLineage,
    assert_observed_artifacts_exact,
    freeze_holdout_lineage,
    verify_holdout_lineage_manifest,
)


def _lineage() -> HoldoutArtifactLineage:
    return HoldoutArtifactLineage(
        bar_run_id=33963234789,
        exposure_run_id=33963542253,
        exposure_recovery_run_id=99999999999,
        industry_run_id=33963542253,
    )


def test_holdout_lineage_requires_exact_20_and_replaces_only_shard12():
    lineage = _lineage()
    bars = lineage.expected_bar_artifacts()
    exposures = lineage.expected_exposure_artifacts()

    assert len(bars) == 20
    assert len(exposures) == 20
    assert bars[0] == "holdout-2026-shard-0"
    assert bars[19] == "holdout-2026-shard-19"
    assert exposures[11] == "holdout-2026-exposure-shard-11"
    assert exposures[12] == "holdout-2026-exposure-shard-12-recovery"
    assert exposures[13] == "holdout-2026-exposure-shard-13"
    assert "holdout-2026-exposure-shard-12" not in exposures


def test_holdout_lineage_manifest_is_immutable_and_verifiable():
    lineage = _lineage()
    manifest = freeze_holdout_lineage(lineage)
    restored = verify_holdout_lineage_manifest(manifest)
    assert restored == lineage
    assert manifest["policy"] == "exact_artifacts_one_shot_no_refit"

    tampered = dict(manifest)
    tampered["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        verify_holdout_lineage_manifest(tampered)


def test_artifact_set_fails_closed_on_missing_extra_or_duplicate():
    expected = ("a", "b", "c")
    assert_observed_artifacts_exact(expected, ["a", "b", "c"], label="test")
    with pytest.raises(RuntimeError, match="artifact set mismatch"):
        assert_observed_artifacts_exact(expected, ["a", "b"], label="test")
    with pytest.raises(RuntimeError, match="artifact set mismatch"):
        assert_observed_artifacts_exact(expected, ["a", "b", "c", "d"], label="test")
    with pytest.raises(RuntimeError, match="duplicate test artifacts"):
        assert_observed_artifacts_exact(expected, ["a", "b", "b"], label="test")


def test_lineage_rejects_any_non_20_shard_contract():
    lineage = HoldoutArtifactLineage(
        bar_run_id=1,
        exposure_run_id=2,
        exposure_recovery_run_id=3,
        industry_run_id=2,
        shard_count=5,
    )
    with pytest.raises(RuntimeError, match="exactly 20 shards"):
        freeze_holdout_lineage(lineage)

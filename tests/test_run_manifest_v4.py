from __future__ import annotations

import pytest

from qmt_quant import run_manifest


STRATEGY = "a" * 64
EVIDENCE = {"backtest": "b" * 64, "walk_forward": "c" * 64}
ARTIFACTS = {"config": "d" * 64, "cost": "e" * 64}


def _git_identity() -> dict[str, str]:
    return {"commit_sha1": "1" * 40, "tree_sha1": "2" * 40}


def test_run_manifest_is_deterministic_and_content_addressed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_manifest, "resolve_git_identity", lambda _: _git_identity())
    first = run_manifest.build_run_manifest(
        strategy_sha256=STRATEGY,
        evidence_sha256=EVIDENCE,
        artifact_sha256=ARTIFACTS,
        evidence_keys=("backtest", "walk_forward"),
        artifact_keys=("config", "cost"),
        repo_root=".",
    )
    second = run_manifest.build_run_manifest(
        strategy_sha256=STRATEGY,
        evidence_sha256=EVIDENCE,
        artifact_sha256=ARTIFACTS,
        evidence_keys=("backtest", "walk_forward"),
        artifact_keys=("config", "cost"),
        repo_root=".",
    )
    assert first == second
    assert len(str(first["run_id"])) == 64
    assert first["git"] == _git_identity()


def test_run_manifest_validation_rejects_tampering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_manifest, "resolve_git_identity", lambda _: _git_identity())
    manifest = run_manifest.build_run_manifest(
        strategy_sha256=STRATEGY,
        evidence_sha256=EVIDENCE,
        artifact_sha256=ARTIFACTS,
        evidence_keys=("backtest", "walk_forward"),
        artifact_keys=("config", "cost"),
        repo_root=".",
    )
    tampered = dict(manifest)
    tampered["run_id"] = "f" * 64
    with pytest.raises(RuntimeError, match="run_id"):
        run_manifest.validate_run_manifest(
            tampered,
            strategy_sha256=STRATEGY,
            evidence_sha256=EVIDENCE,
            artifact_sha256=ARTIFACTS,
            evidence_keys=("backtest", "walk_forward"),
            artifact_keys=("config", "cost"),
        )


def test_run_manifest_hash_maps_are_exact_not_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_manifest, "resolve_git_identity", lambda _: _git_identity())
    with pytest.raises(RuntimeError, match="keys mismatch"):
        run_manifest.build_run_manifest(
            strategy_sha256=STRATEGY,
            evidence_sha256={**EVIDENCE, "unexpected": "f" * 64},
            artifact_sha256=ARTIFACTS,
            evidence_keys=("backtest", "walk_forward"),
            artifact_keys=("config", "cost"),
            repo_root=".",
        )

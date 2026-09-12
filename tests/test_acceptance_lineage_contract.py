from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

import run_acceptance
from qmt_quant.acceptance_lineage import build_acceptance_lineage
from qmt_quant.config import CostConfig
from qmt_quant.cost_manifest import build_cost_manifest


ROOT = Path(__file__).resolve().parents[1]


def _artifact_paths(tmp_path: Path, strategy_source: Path) -> dict[str, Path]:
    artifacts = {"strategy_source": strategy_source}
    for key in ("config", "data_lineage", "engine_manifest", "dependency_lock"):
        path = tmp_path / f"{key}.txt"
        path.write_text(key, encoding="utf-8")
        artifacts[key] = path
    cost_path = tmp_path / "cost_manifest.json"
    cost_path.write_text(
        json.dumps(build_cost_manifest(CostConfig()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifacts["cost_manifest"] = cost_path
    return artifacts


def test_acceptance_cli_has_no_legacy_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_acceptance.py"])
    with pytest.raises(SystemExit):
        run_acceptance.parse_args()


def test_acceptance_requires_exact_lowercase_sha256():
    assert run_acceptance._require_strategy_sha("a" * 64) == "a" * 64
    with pytest.raises(ValueError, match="64-hex"):
        run_acceptance._require_strategy_sha("abc")
    with pytest.raises(ValueError, match="64-hex"):
        run_acceptance._require_strategy_sha("G" * 64)


def test_acceptance_hashes_evidence_file_bytes(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_bytes(b'{"value":1}\n')
    assert run_acceptance._sha256_path(evidence) == hashlib.sha256(evidence.read_bytes()).hexdigest()


def test_acceptance_v4_binds_strategy_source_cost_and_run_manifest(tmp_path):
    strategy_source = tmp_path / "strategy.json"
    strategy_source.write_bytes(b"{}")
    strategy_sha = hashlib.sha256(strategy_source.read_bytes()).hexdigest()

    evidence_paths = {}
    for key in ("backtest", "walk_forward", "folds", "stress"):
        path = tmp_path / f"{key}.txt"
        path.write_text(key, encoding="utf-8")
        evidence_paths[key] = path
    artifact_paths = _artifact_paths(tmp_path, strategy_source)

    lineage = build_acceptance_lineage(
        strategy_sha256=strategy_sha,
        evidence_paths=evidence_paths,
        artifact_paths=artifact_paths,
        repo_root=ROOT,
    )
    assert lineage["strategy_sha256"] == strategy_sha
    assert len(lineage["evidence_sha256"]) == 4
    assert len(lineage["artifact_sha256"]) == 6
    assert len(lineage["binding_sha256"]) == 64
    assert len(lineage["run_id"]) == 64
    assert lineage["run_manifest"]["run_id"] == lineage["run_id"]
    assert len(lineage["run_manifest"]["git"]["commit_sha1"]) == 40
    assert len(lineage["run_manifest"]["git"]["tree_sha1"]) == 40


def test_acceptance_v4_rejects_invalid_cost_manifest(tmp_path):
    strategy_source = tmp_path / "strategy.json"
    strategy_source.write_bytes(b"{}")
    strategy_sha = hashlib.sha256(strategy_source.read_bytes()).hexdigest()
    evidence_paths = {}
    for key in ("backtest", "walk_forward", "folds", "stress"):
        path = tmp_path / f"{key}.txt"
        path.write_text(key, encoding="utf-8")
        evidence_paths[key] = path
    artifact_paths = _artifact_paths(tmp_path, strategy_source)
    artifact_paths["cost_manifest"].write_text(
        json.dumps({"schema": "not-a-cost-manifest"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="cost manifest requires schema"):
        build_acceptance_lineage(
            strategy_sha256=strategy_sha,
            evidence_paths=evidence_paths,
            artifact_paths=artifact_paths,
            repo_root=ROOT,
        )


def test_acceptance_v4_rejects_user_sha_not_derived_from_strategy_source(tmp_path):
    strategy_source = tmp_path / "strategy.json"
    strategy_source.write_bytes(b"{}")
    evidence_paths = {}
    for key in ("backtest", "walk_forward", "folds", "stress"):
        path = tmp_path / f"{key}.txt"
        path.write_text(key, encoding="utf-8")
        evidence_paths[key] = path
    artifact_paths = _artifact_paths(tmp_path, strategy_source)
    with pytest.raises(RuntimeError, match="strategy source identity"):
        build_acceptance_lineage(
            strategy_sha256="a" * 64,
            evidence_paths=evidence_paths,
            artifact_paths=artifact_paths,
            repo_root=ROOT,
        )


def test_acceptance_source_requires_full_v4_lineage_inputs():
    source = Path(run_acceptance.__file__).read_text(encoding="utf-8")
    assert "output/v3_research" not in source
    for arg in (
        "--backtest",
        "--walk-forward",
        "--folds",
        "--stress",
        "--strategy-sha256",
        "--strategy-source",
        "--config",
        "--data-lineage",
        "--engine-manifest",
        "--dependency-lock",
        "--cost-manifest",
    ):
        assert f'p.add_argument("{arg}", required=True)' in source
    assert 'report["lineage"]' in source
    assert 'report["evidence_sha256"]' in source
    assert 'report["artifact_sha256"]' in source
    assert 'report["run_id"]' in source
    assert 'report["run_manifest"]' in source
    assert 'out / "run_manifest.json"' in source


def test_legacy_acceptance_batch_is_fail_closed():
    batch = (ROOT / "run_acceptance.bat").read_text(encoding="utf-8")
    assert "output\\v3_research" not in batch
    assert "run_acceptance.py --backtest" not in batch
    assert "no longer supplies implicit V3/V4 evidence paths" in batch
    assert "exit /b 2" in batch

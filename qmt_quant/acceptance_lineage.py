from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from .holdout import verify_candidate_manifest
from .production_candidate import load_legacy_strategy_config
from .run_manifest import build_run_manifest, validate_run_manifest


ACCEPTANCE_SCHEMA = "qmt-acceptance-v4"
EVIDENCE_KEYS = ("backtest", "walk_forward", "folds", "stress")
LINEAGE_ARTIFACT_KEYS = (
    "strategy_source",
    "config",
    "data_lineage",
    "engine_manifest",
    "dependency_lock",
    "cost_manifest",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def require_sha256(value: str, *, name: str) -> str:
    sha = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(sha):
        raise ValueError(f"{name} must be an exact lowercase 64-hex SHA256")
    return sha


def sha256_path(path: str | Path) -> str:
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_strategy_source_sha256(path: str | Path) -> str:
    """Derive strategy identity from a known strategy/candidate source file."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("strategy source must be a JSON strategy/candidate manifest") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("strategy source must be a JSON object")

    frozen = payload.get("frozen")
    if isinstance(frozen, Mapping):
        return verify_candidate_manifest(frozen).fingerprint()
    if "candidate" in payload and "sha256" in payload:
        return verify_candidate_manifest(payload).fingerprint()

    return load_legacy_strategy_config(source).sha256


def lineage_binding_sha256(
    *,
    strategy_sha256: str,
    evidence_sha256: Mapping[str, str],
    artifact_sha256: Mapping[str, str],
    run_id: str,
) -> str:
    strategy = require_sha256(strategy_sha256, name="strategy_sha256")
    evidence = {
        key: require_sha256(str(evidence_sha256.get(key, "")), name=f"evidence_sha256.{key}")
        for key in EVIDENCE_KEYS
    }
    artifacts = {
        key: require_sha256(str(artifact_sha256.get(key, "")), name=f"artifact_sha256.{key}")
        for key in LINEAGE_ARTIFACT_KEYS
    }
    payload = {
        "strategy_sha256": strategy,
        "evidence_sha256": evidence,
        "artifact_sha256": artifacts,
        "run_id": require_sha256(run_id, name="run_id"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def build_acceptance_lineage(
    *,
    strategy_sha256: str,
    evidence_paths: Mapping[str, str | Path],
    artifact_paths: Mapping[str, str | Path],
    repo_root: str | Path = ".",
) -> dict[str, object]:
    strategy = require_sha256(strategy_sha256, name="strategy_sha256")
    observed_strategy = resolve_strategy_source_sha256(artifact_paths["strategy_source"])
    if observed_strategy != strategy:
        raise RuntimeError(
            "strategy source identity does not match --strategy-sha256; refusing mixed acceptance evidence"
        )

    evidence_hashes = {key: sha256_path(evidence_paths[key]) for key in EVIDENCE_KEYS}
    artifact_hashes = {key: sha256_path(artifact_paths[key]) for key in LINEAGE_ARTIFACT_KEYS}
    run_manifest = build_run_manifest(
        strategy_sha256=strategy,
        evidence_sha256=evidence_hashes,
        artifact_sha256=artifact_hashes,
        evidence_keys=EVIDENCE_KEYS,
        artifact_keys=LINEAGE_ARTIFACT_KEYS,
        repo_root=repo_root,
    )
    run_id = require_sha256(str(run_manifest["run_id"]), name="run_manifest.run_id")
    binding = lineage_binding_sha256(
        strategy_sha256=strategy,
        evidence_sha256=evidence_hashes,
        artifact_sha256=artifact_hashes,
        run_id=run_id,
    )
    return {
        "strategy_sha256": strategy,
        "evidence_sha256": evidence_hashes,
        "artifact_sha256": artifact_hashes,
        "run_id": run_id,
        "run_manifest": run_manifest,
        "binding_sha256": binding,
    }


def validate_acceptance_lineage(lineage: Mapping[str, object], *, strategy_sha256: str) -> dict:
    observed_strategy = require_sha256(
        str(lineage.get("strategy_sha256", "")), name="lineage.strategy_sha256"
    )
    if observed_strategy != require_sha256(strategy_sha256, name="strategy_sha256"):
        raise RuntimeError("acceptance lineage is not bound to the target strategy SHA256")
    evidence = lineage.get("evidence_sha256")
    artifacts = lineage.get("artifact_sha256")
    run_manifest = lineage.get("run_manifest")
    if not isinstance(evidence, Mapping) or not isinstance(artifacts, Mapping):
        raise RuntimeError("acceptance lineage requires evidence and artifact SHA256 maps")
    if not isinstance(run_manifest, Mapping):
        raise RuntimeError("acceptance v4 lineage requires immutable run_manifest")

    evidence_map = {str(k): str(v) for k, v in evidence.items()}
    artifact_map = {str(k): str(v) for k, v in artifacts.items()}
    validated_manifest = validate_run_manifest(
        run_manifest,
        strategy_sha256=observed_strategy,
        evidence_sha256=evidence_map,
        artifact_sha256=artifact_map,
        evidence_keys=EVIDENCE_KEYS,
        artifact_keys=LINEAGE_ARTIFACT_KEYS,
    )
    run_id = require_sha256(str(lineage.get("run_id", "")), name="lineage.run_id")
    if run_id != validated_manifest["run_id"]:
        raise RuntimeError("acceptance lineage run_id does not match run_manifest")

    expected = lineage_binding_sha256(
        strategy_sha256=observed_strategy,
        evidence_sha256=evidence_map,
        artifact_sha256=artifact_map,
        run_id=run_id,
    )
    observed_binding = require_sha256(
        str(lineage.get("binding_sha256", "")), name="lineage.binding_sha256"
    )
    if observed_binding != expected:
        raise RuntimeError("acceptance lineage binding SHA256 is invalid")
    return {
        "strategy_sha256": observed_strategy,
        "evidence_sha256": {key: str(evidence[key]) for key in EVIDENCE_KEYS},
        "artifact_sha256": {key: str(artifacts[key]) for key in LINEAGE_ARTIFACT_KEYS},
        "run_id": run_id,
        "run_manifest": validated_manifest,
        "binding_sha256": observed_binding,
    }

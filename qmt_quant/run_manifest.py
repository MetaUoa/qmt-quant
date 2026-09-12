from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Mapping, Sequence


RUN_MANIFEST_SCHEMA = "qmt-run-manifest-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


def _require_sha256(value: object, *, name: str) -> str:
    token = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(token):
        raise ValueError(f"{name} must be an exact lowercase 64-hex SHA256")
    return token


def _require_git_sha1(value: object, *, name: str) -> str:
    token = str(value).strip().lower()
    if not _GIT_SHA1_RE.fullmatch(token):
        raise RuntimeError(f"{name} must be an exact lowercase 40-hex Git SHA1")
    return token


def canonical_sha256(payload: Mapping[str, object]) -> str:
    raw = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git(repo_root: str | Path, *args: str) -> str:
    root = Path(repo_root).resolve()
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"git {' '.join(args)} failed for {root}: {stderr or completed.returncode}")
    return completed.stdout.strip()


def resolve_git_identity(repo_root: str | Path) -> dict[str, str]:
    root = Path(repo_root).resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"acceptance v4 requires a Git checkout: {root}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    if dirty:
        raise RuntimeError("acceptance v4 refuses a dirty tracked Git worktree")
    commit = _require_git_sha1(_git(root, "rev-parse", "HEAD"), name="git.commit_sha1")
    tree = _require_git_sha1(_git(root, "rev-parse", "HEAD^{tree}"), name="git.tree_sha1")
    return {"commit_sha1": commit, "tree_sha1": tree}


def _normalized_hash_map(
    values: Mapping[str, object],
    *,
    keys: Sequence[str],
    name: str,
) -> dict[str, str]:
    expected = tuple(str(key) for key in keys)
    missing = [key for key in expected if key not in values]
    extras = sorted(str(key) for key in values if str(key) not in expected)
    if missing or extras:
        raise RuntimeError(f"{name} keys mismatch; missing={missing} extras={extras}")
    return {
        key: _require_sha256(values[key], name=f"{name}.{key}")
        for key in expected
    }


def build_run_manifest(
    *,
    strategy_sha256: str,
    evidence_sha256: Mapping[str, object],
    artifact_sha256: Mapping[str, object],
    evidence_keys: Sequence[str],
    artifact_keys: Sequence[str],
    repo_root: str | Path,
) -> dict[str, object]:
    strategy = _require_sha256(strategy_sha256, name="strategy_sha256")
    evidence = _normalized_hash_map(
        evidence_sha256,
        keys=evidence_keys,
        name="evidence_sha256",
    )
    artifacts = _normalized_hash_map(
        artifact_sha256,
        keys=artifact_keys,
        name="artifact_sha256",
    )
    core: dict[str, object] = {
        "schema": RUN_MANIFEST_SCHEMA,
        "git": resolve_git_identity(repo_root),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "strategy_sha256": strategy,
        "evidence_sha256": evidence,
        "artifact_sha256": artifacts,
    }
    return {**core, "run_id": canonical_sha256(core)}


def validate_run_manifest(
    manifest: Mapping[str, object],
    *,
    strategy_sha256: str,
    evidence_sha256: Mapping[str, object],
    artifact_sha256: Mapping[str, object],
    evidence_keys: Sequence[str],
    artifact_keys: Sequence[str],
) -> dict[str, object]:
    if str(manifest.get("schema", "")) != RUN_MANIFEST_SCHEMA:
        raise RuntimeError(f"run manifest requires schema {RUN_MANIFEST_SCHEMA}")
    strategy = _require_sha256(strategy_sha256, name="strategy_sha256")
    observed_strategy = _require_sha256(
        manifest.get("strategy_sha256", ""), name="run_manifest.strategy_sha256"
    )
    if observed_strategy != strategy:
        raise RuntimeError("run manifest strategy SHA256 mismatch")

    evidence = _normalized_hash_map(
        evidence_sha256,
        keys=evidence_keys,
        name="evidence_sha256",
    )
    artifacts = _normalized_hash_map(
        artifact_sha256,
        keys=artifact_keys,
        name="artifact_sha256",
    )
    observed_evidence = manifest.get("evidence_sha256")
    observed_artifacts = manifest.get("artifact_sha256")
    if not isinstance(observed_evidence, Mapping) or not isinstance(observed_artifacts, Mapping):
        raise RuntimeError("run manifest requires evidence/artifact hash maps")
    if _normalized_hash_map(observed_evidence, keys=evidence_keys, name="run_manifest.evidence_sha256") != evidence:
        raise RuntimeError("run manifest evidence SHA256 mismatch")
    if _normalized_hash_map(observed_artifacts, keys=artifact_keys, name="run_manifest.artifact_sha256") != artifacts:
        raise RuntimeError("run manifest artifact SHA256 mismatch")

    git = manifest.get("git")
    python_runtime = manifest.get("python")
    if not isinstance(git, Mapping) or not isinstance(python_runtime, Mapping):
        raise RuntimeError("run manifest requires git and python runtime identity")
    normalized_git = {
        "commit_sha1": _require_git_sha1(git.get("commit_sha1", ""), name="run_manifest.git.commit_sha1"),
        "tree_sha1": _require_git_sha1(git.get("tree_sha1", ""), name="run_manifest.git.tree_sha1"),
    }
    implementation = str(python_runtime.get("implementation", "")).strip()
    version = str(python_runtime.get("version", "")).strip()
    major_minor = str(python_runtime.get("major_minor", "")).strip()
    if not implementation or not version or not re.fullmatch(r"\d+\.\d+", major_minor):
        raise RuntimeError("run manifest contains invalid Python runtime identity")

    core: dict[str, object] = {
        "schema": RUN_MANIFEST_SCHEMA,
        "git": normalized_git,
        "python": {
            "implementation": implementation,
            "version": version,
            "major_minor": major_minor,
        },
        "strategy_sha256": strategy,
        "evidence_sha256": evidence,
        "artifact_sha256": artifacts,
    }
    expected_run_id = canonical_sha256(core)
    observed_run_id = _require_sha256(manifest.get("run_id", ""), name="run_manifest.run_id")
    if observed_run_id != expected_run_id:
        raise RuntimeError("run manifest run_id is invalid")
    return {**core, "run_id": observed_run_id}

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping


@dataclass(frozen=True)
class HoldoutArtifactLineage:
    """Immutable provenance for a one-shot 2026 holdout evaluation.

    The recovery run is allowed to replace exactly one exposure shard.  It cannot
    replace bars, industry data, or any additional exposure shard without creating
    a new lineage fingerprint and therefore a different evaluation contract.
    """

    bar_run_id: int
    exposure_run_id: int
    exposure_recovery_run_id: int
    industry_run_id: int
    shard_count: int = 20
    exposure_recovery_shard: int = 12

    def canonical_payload(self) -> dict:
        payload = asdict(self)
        return {key: int(value) for key, value in payload.items()}

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def expected_bar_artifacts(self) -> tuple[str, ...]:
        self._validate_contract()
        return tuple(f"holdout-2026-shard-{index}" for index in range(self.shard_count))

    def expected_exposure_artifacts(self) -> tuple[str, ...]:
        self._validate_contract()
        names = []
        for index in range(self.shard_count):
            if index == self.exposure_recovery_shard:
                names.append(f"holdout-2026-exposure-shard-{index}-recovery")
            else:
                names.append(f"holdout-2026-exposure-shard-{index}")
        return tuple(names)

    def expected_industry_artifact(self) -> str:
        self._validate_contract()
        return "holdout-2026-industry-snapshots"

    def _validate_contract(self) -> None:
        if int(self.shard_count) != 20:
            raise RuntimeError("2026 holdout lineage must preserve exactly 20 shards")
        if not 0 <= int(self.exposure_recovery_shard) < int(self.shard_count):
            raise RuntimeError("exposure recovery shard is outside the 20-shard contract")
        for field in (
            self.bar_run_id,
            self.exposure_run_id,
            self.exposure_recovery_run_id,
            self.industry_run_id,
        ):
            if int(field) <= 0:
                raise RuntimeError("holdout artifact run IDs must be positive")


def freeze_holdout_lineage(lineage: HoldoutArtifactLineage) -> dict:
    lineage._validate_contract()
    return {
        "lineage": lineage.canonical_payload(),
        "sha256": lineage.fingerprint(),
        "bar_artifacts": list(lineage.expected_bar_artifacts()),
        "exposure_artifacts": list(lineage.expected_exposure_artifacts()),
        "industry_artifact": lineage.expected_industry_artifact(),
        "policy": "exact_artifacts_one_shot_no_refit",
    }


def verify_holdout_lineage_manifest(manifest: Mapping) -> HoldoutArtifactLineage:
    if "lineage" not in manifest or "sha256" not in manifest:
        raise ValueError("holdout lineage manifest requires lineage and sha256")
    lineage = HoldoutArtifactLineage(**dict(manifest["lineage"]))
    lineage._validate_contract()
    if str(manifest["sha256"]) != lineage.fingerprint():
        raise RuntimeError("holdout artifact lineage fingerprint mismatch")

    expected_bars = list(lineage.expected_bar_artifacts())
    expected_exposures = list(lineage.expected_exposure_artifacts())
    if list(manifest.get("bar_artifacts", [])) != expected_bars:
        raise RuntimeError("holdout bar artifact lineage does not match exact 20-shard contract")
    if list(manifest.get("exposure_artifacts", [])) != expected_exposures:
        raise RuntimeError("holdout exposure artifact lineage does not match 19+1 recovery contract")
    if str(manifest.get("industry_artifact", "")) != lineage.expected_industry_artifact():
        raise RuntimeError("holdout industry artifact lineage mismatch")
    if str(manifest.get("policy", "")) != "exact_artifacts_one_shot_no_refit":
        raise RuntimeError("holdout artifact lineage policy mismatch")
    return lineage


def assert_observed_artifacts_exact(
    expected: tuple[str, ...],
    observed: list[str] | tuple[str, ...],
    *,
    label: str,
) -> None:
    """Fail closed on missing, duplicate, or unexpected artifact names."""
    observed_list = [str(name) for name in observed]
    if len(observed_list) != len(set(observed_list)):
        raise RuntimeError(f"duplicate {label} artifacts observed")
    expected_set = set(expected)
    observed_set = set(observed_list)
    missing = sorted(expected_set.difference(observed_set))
    extra = sorted(observed_set.difference(expected_set))
    if missing or extra:
        raise RuntimeError(
            f"{label} artifact set mismatch: missing={missing}, extra={extra}"
        )

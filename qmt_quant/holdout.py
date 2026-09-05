from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class FrozenCandidate:
    name: str
    research_data_end: str
    neutralization_variant: str
    weights: Mapping[str, float]
    top_n: int
    rebalance_days: int
    execution_delay_sessions: int
    min_price: float
    min_amount: float
    min_listing_sessions: int

    def canonical_payload(self) -> dict:
        payload = asdict(self)
        payload["weights"] = {
            str(key): float(value) for key, value in sorted(dict(self.weights).items())
        }
        return payload

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def freeze_candidate_manifest(candidate: FrozenCandidate) -> dict:
    return {
        "candidate": candidate.canonical_payload(),
        "sha256": candidate.fingerprint(),
        "holdout_policy": "one_shot_no_refit",
    }


def verify_candidate_manifest(manifest: Mapping) -> FrozenCandidate:
    if "candidate" not in manifest or "sha256" not in manifest:
        raise ValueError("frozen candidate manifest requires candidate and sha256")
    candidate = FrozenCandidate(**dict(manifest["candidate"]))
    expected = candidate.fingerprint()
    if str(manifest["sha256"]) != expected:
        raise RuntimeError("frozen candidate fingerprint mismatch")
    return candidate


def assert_holdout_boundary(candidate: FrozenCandidate, *, holdout_start) -> None:
    research_end = pd.Timestamp(candidate.research_data_end).normalize()
    start = pd.Timestamp(holdout_start).normalize()
    if research_end >= start:
        raise RuntimeError(
            f"candidate research_data_end {research_end.date()} overlaps holdout start {start.date()}"
        )


def assert_one_shot_result(result_manifest: Mapping, candidate: FrozenCandidate) -> None:
    if str(result_manifest.get("candidate_sha256", "")) != candidate.fingerprint():
        raise RuntimeError("holdout result was not produced by the frozen candidate")
    if int(result_manifest.get("evaluation_count", 0)) != 1:
        raise RuntimeError("holdout must be evaluated exactly once per frozen candidate")

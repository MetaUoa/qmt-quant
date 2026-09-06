from __future__ import annotations

from typing import Mapping


SCHEMA_VERSION = 1


def front_adjustment_provenance(*, provider: str, requested_end: str, legacy_inferred: bool = False) -> dict:
    """Describe the semantics of a frozen provider-side front-adjusted cache.

    This records provenance only. It never recomputes adjustment factors and never
    reacquires bars. The important invariant is that provider-side front-adjusted
    values are a snapshot of the requested acquisition window and must not be mixed
    across independently acquired lineages as though they shared one stable base.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "provider": str(provider),
        "adjustment": "front",
        "requested_end": str(requested_end),
        "baseline_semantics": "provider_snapshot_for_requested_window",
        "frozen_artifact_snapshot": True,
        "cross_lineage_mixing_allowed": False,
        "legacy_inferred": bool(legacy_inferred),
    }


def raw_reference_provenance(*, provider: str, requested_end: str, legacy_inferred: bool = False) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "provider": str(provider),
        "adjustment": "none",
        "requested_end": str(requested_end),
        "baseline_semantics": "raw_reference_snapshot_for_requested_window",
        "frozen_artifact_snapshot": True,
        "cross_lineage_mixing_allowed": False,
        "legacy_inferred": bool(legacy_inferred),
    }


def validate_shard_adjustment_provenance(manifests: list[Mapping]) -> tuple[dict, dict]:
    """Return one coherent adjusted/raw provenance pair or fail closed.

    Historical frozen shards predate this schema. A complete all-legacy set can be
    represented explicitly as inferred provenance from its common end date. Mixing
    newly annotated and legacy shards is rejected because their baseline contract is
    not demonstrably identical.
    """
    if not manifests:
        raise ValueError("at least one shard manifest is required")
    present = ["adjustment_provenance" in row for row in manifests]
    if any(present) and not all(present):
        raise RuntimeError("cannot mix legacy and annotated adjustment provenance")

    provider_values = {str(row.get("source", "baostock")) for row in manifests}
    end_values = {str(row["end"]) for row in manifests}
    if len(end_values) != 1:
        raise RuntimeError("shard manifests disagree on adjustment requested_end")
    requested_end = next(iter(end_values))

    if not any(present):
        provider = "baostock" if provider_values == {"baostock"} else "+".join(sorted(provider_values))
        return (
            front_adjustment_provenance(
                provider=provider,
                requested_end=requested_end,
                legacy_inferred=True,
            ),
            raw_reference_provenance(
                provider=provider,
                requested_end=requested_end,
                legacy_inferred=True,
            ),
        )

    adjusted = [dict(row["adjustment_provenance"]["adjusted"]) for row in manifests]
    raw = [dict(row["adjustment_provenance"]["raw"]) for row in manifests]
    if any(row != adjusted[0] for row in adjusted[1:]):
        raise RuntimeError("adjusted-cache provenance differs across shards")
    if any(row != raw[0] for row in raw[1:]):
        raise RuntimeError("raw-cache provenance differs across shards")
    if str(adjusted[0].get("requested_end")) != requested_end:
        raise RuntimeError("adjusted provenance requested_end does not match shard manifest")
    if str(raw[0].get("requested_end")) != requested_end:
        raise RuntimeError("raw provenance requested_end does not match shard manifest")
    if bool(adjusted[0].get("cross_lineage_mixing_allowed", True)):
        raise RuntimeError("front-adjusted cache must forbid cross-lineage mixing")
    return adjusted[0], raw[0]

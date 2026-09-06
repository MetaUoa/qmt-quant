from __future__ import annotations

import pytest

from qmt_quant.adjustment_provenance import (
    front_adjustment_provenance,
    validate_shard_adjustment_provenance,
)


def _legacy(index: int) -> dict:
    return {
        "source": "baostock",
        "start": "20170101",
        "end": "20251231",
        "benchmark": "000905.SH",
        "shard_index": index,
        "shard_count": 20,
    }


def test_legacy_frozen_set_is_explicitly_marked_inferred_and_nonmixable():
    adjusted, raw = validate_shard_adjustment_provenance([_legacy(0), _legacy(1)])
    assert adjusted["adjustment"] == "front"
    assert adjusted["requested_end"] == "20251231"
    assert adjusted["legacy_inferred"] is True
    assert adjusted["frozen_artifact_snapshot"] is True
    assert adjusted["cross_lineage_mixing_allowed"] is False
    assert raw["adjustment"] == "none"


def test_mixed_legacy_and_annotated_shards_fail_closed():
    legacy = _legacy(0)
    annotated = _legacy(1)
    annotated["adjustment_provenance"] = {
        "adjusted": front_adjustment_provenance(
            provider="baostock", requested_end="20251231"
        ),
        "raw": {
            "schema_version": 1,
            "provider": "baostock",
            "adjustment": "none",
            "requested_end": "20251231",
            "baseline_semantics": "raw_reference_snapshot_for_requested_window",
            "frozen_artifact_snapshot": True,
            "cross_lineage_mixing_allowed": False,
            "legacy_inferred": False,
        },
    }
    with pytest.raises(RuntimeError, match="cannot mix legacy and annotated"):
        validate_shard_adjustment_provenance([legacy, annotated])


def test_front_adjustment_contract_forbids_cross_lineage_mixing():
    row = front_adjustment_provenance(provider="baostock", requested_end="20251231")
    assert row["baseline_semantics"] == "provider_snapshot_for_requested_window"
    assert row["cross_lineage_mixing_allowed"] is False

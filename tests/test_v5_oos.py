from __future__ import annotations

import pandas as pd

from qmt_quant.v5_oos import purged_evidence_end


def test_purge_keeps_forward_label_fully_before_validation():
    calendar = pd.bdate_range("2020-10-01", "2021-02-01")
    validation_start = pd.Timestamp("2021-01-01")
    evidence_end = purged_evidence_end(
        calendar,
        validation_start,
        max_forward_horizon=20,
    )
    first_validation_i = int(calendar.searchsorted(validation_start, side="left"))
    evidence_i = int(calendar.get_loc(evidence_end))
    assert evidence_i + 20 < first_validation_i


def test_purge_rejects_non_positive_horizon():
    calendar = pd.bdate_range("2020-01-01", "2021-01-01")
    try:
        purged_evidence_end(calendar, "2021-01-01", max_forward_horizon=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")

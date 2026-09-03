from __future__ import annotations

import pandas as pd

from run_data_audit import count_observed_sessions


def test_count_observed_sessions_uses_same_denominator_sessions():
    frame = pd.DataFrame(
        {"close": [10.0, 10.1, 10.2]},
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )
    expected = pd.DatetimeIndex(["2025-01-02", "2025-01-06"])

    observed = count_observed_sessions(frame, expected, "close")

    assert observed == 2
    assert observed / len(expected) <= 1.0


def test_count_observed_sessions_ignores_nan_and_duplicate_rows():
    frame = pd.DataFrame(
        {"open": [10.0, float("nan"), 10.5, 10.6]},
        index=pd.to_datetime(
            ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-06"]
        ),
    )
    expected = pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06"])

    assert count_observed_sessions(frame, expected, "open") == 2

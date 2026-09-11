from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from qmt_quant.freshness import (
    FreshnessError,
    FreshnessPolicy,
    require_fresh,
    validate_query_window,
    validate_tick_freshness,
)


def test_tick_freshness_accepts_recent_epoch_milliseconds() -> None:
    now = datetime(2026, 9, 12, 1, 2, 3, tzinfo=timezone.utc)
    ticks = {"000001.SZ": {"time": int((now - timedelta(seconds=1)).timestamp() * 1000)}}
    report = validate_tick_freshness(
        ticks,
        ["000001.SZ"],
        policy=FreshnessPolicy(quote_max_age_seconds=5),
        now=now,
    )
    assert report["passed"] is True
    assert report["quotes"][0]["age_seconds"] == pytest.approx(1.0)


def test_tick_freshness_fails_stale_missing_and_future_clock() -> None:
    now = datetime(2026, 9, 12, 1, 2, 3, tzinfo=timezone.utc)
    policy = FreshnessPolicy(quote_max_age_seconds=5, clock_skew_tolerance_seconds=1)
    report = validate_tick_freshness(
        {
            "STALE": {"time": int((now - timedelta(seconds=6)).timestamp() * 1000)},
            "UNKNOWN": {"lastPrice": 10.0},
            "FUTURE": {"time": int((now + timedelta(seconds=2)).timestamp() * 1000)},
        },
        ["STALE", "UNKNOWN", "FUTURE", "MISSING"],
        policy=policy,
        now=now,
    )
    assert report["passed"] is False
    reasons = str(report["violations"])
    assert "stale_tick" in reasons
    assert "unknown_tick_clock" in reasons
    assert "tick_from_future" in reasons
    assert "missing_tick" in reasons
    with pytest.raises(FreshnessError):
        require_fresh(report, label="quotes")


def test_tick_stime_is_interpreted_as_china_standard_time() -> None:
    now = datetime(2026, 9, 12, 1, 30, 0, tzinfo=timezone.utc)
    report = validate_tick_freshness(
        {"000001.SZ": {"stime": "20260912 09:29:59"}},
        ["000001.SZ"],
        policy=FreshnessPolicy(quote_max_age_seconds=5),
        now=now,
    )
    assert report["passed"] is True
    assert report["quotes"][0]["age_seconds"] == pytest.approx(1.0)


def test_query_window_fails_slow_and_clock_anomalies() -> None:
    now = datetime(2026, 9, 12, 1, 0, 10, tzinfo=timezone.utc)
    policy = FreshnessPolicy(query_max_duration_seconds=5, clock_skew_tolerance_seconds=1)
    slow = validate_query_window(
        source="snapshot",
        started_at=now - timedelta(seconds=10),
        completed_at=now,
        policy=policy,
        now=now,
    )
    assert slow["passed"] is False
    assert "query_too_slow" in slow["violations"]

    backwards = validate_query_window(
        source="snapshot",
        started_at=now,
        completed_at=now - timedelta(seconds=2),
        policy=policy,
        now=now,
    )
    assert backwards["passed"] is False
    assert "clock_moved_backwards_during_query" in backwards["violations"]


def test_naive_timestamp_is_rejected() -> None:
    naive = datetime(2026, 9, 12, 1, 0, 0)
    with pytest.raises(FreshnessError, match="timezone"):
        validate_query_window(
            source="snapshot",
            started_at=naive,
            completed_at=naive,
            policy=FreshnessPolicy(),
        )

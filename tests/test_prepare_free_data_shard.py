from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from prepare_free_data_shard import (
    _bind_baostock_socket_timeout,
    _configure_socket_timeout,
    fetch_history_with_retry,
)


class FakeRawSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class FakeBaoStock:
    def __init__(self) -> None:
        self.logins = 0
        self.logouts = 0

    def login(self):
        self.logins += 1
        return SimpleNamespace(error_code="0", error_msg="")

    def logout(self):
        self.logouts += 1
        return SimpleNamespace(error_code="0", error_msg="")


def test_socket_timeout_bounds_network_stalls(monkeypatch):
    configured: list[float] = []
    monkeypatch.setattr("prepare_free_data_shard.socket.setdefaulttimeout", configured.append)

    _configure_socket_timeout(45.0)

    assert configured == [45.0]
    with pytest.raises(ValueError, match="socket_timeout must be > 0"):
        _configure_socket_timeout(0)


def test_context_socket_binding_sets_timeout(monkeypatch):
    raw_socket = FakeRawSocket()
    context = SimpleNamespace(default_socket=raw_socket)
    monkeypatch.setattr(
        "prepare_free_data_shard.importlib.import_module",
        lambda name: context,
    )

    _bind_baostock_socket_timeout(30.0)

    assert raw_socket.timeouts == [30.0]
    with pytest.raises(ValueError, match="socket_timeout must be > 0"):
        _bind_baostock_socket_timeout(0)


def test_context_socket_binding_fails_closed_without_active_socket(monkeypatch):
    context = SimpleNamespace(default_socket=None)
    monkeypatch.setattr(
        "prepare_free_data_shard.importlib.import_module",
        lambda name: context,
    )

    with pytest.raises(RuntimeError, match="default_socket is unavailable"):
        _bind_baostock_socket_timeout(30.0)


def test_retry_reconnects_after_transient_reset(monkeypatch):
    api = FakeBaoStock()
    calls = 0
    configured: list[float] = []
    bound: list[float] = []

    def fetcher(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError(10054, "remote host reset")
        return pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "close": [1.0]})

    monkeypatch.setattr("prepare_free_data_shard.time.sleep", lambda _: None)
    monkeypatch.setattr("prepare_free_data_shard._configure_socket_timeout", configured.append)
    monkeypatch.setattr("prepare_free_data_shard._bind_baostock_socket_timeout", bound.append)
    frame = fetch_history_with_retry(
        api,
        "000001.SZ",
        "20240101",
        "20240131",
        adjusted=True,
        include_meta=False,
        attempts=2,
        socket_timeout_seconds=12.0,
        fetcher=fetcher,
    )

    assert calls == 2
    assert api.logouts == 1
    assert api.logins == 1
    assert configured == [12.0]
    assert bound == [12.0]
    assert not frame.empty


def test_retry_stays_fail_closed_after_exhaustion(monkeypatch):
    api = FakeBaoStock()
    configured: list[float] = []
    bound: list[float] = []

    def fetcher(*args, **kwargs):
        raise ConnectionResetError(10054, "remote host reset")

    monkeypatch.setattr("prepare_free_data_shard.time.sleep", lambda _: None)
    monkeypatch.setattr("prepare_free_data_shard._configure_socket_timeout", configured.append)
    monkeypatch.setattr("prepare_free_data_shard._bind_baostock_socket_timeout", bound.append)
    with pytest.raises(RuntimeError, match="history fetch failed after retries"):
        fetch_history_with_retry(
            api,
            "000001.SZ",
            "20240101",
            "20240131",
            adjusted=True,
            include_meta=False,
            attempts=3,
            socket_timeout_seconds=9.0,
            fetcher=fetcher,
        )

    assert api.logouts == 2
    assert api.logins == 2
    assert configured == [9.0, 9.0]
    assert bound == [9.0, 9.0]

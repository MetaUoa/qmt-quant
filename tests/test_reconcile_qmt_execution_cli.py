from __future__ import annotations

import json
from pathlib import Path
import sys

import reconcile_qmt_execution
from qmt_quant.execution_state import (
    account_execution_key,
    load_account_execution_lock,
    read_execution_state,
    reserve_account_execution_lock,
    reserve_execution_batch,
)


BATCH = "a" * 64


class _FakeBroker:
    def __init__(self, *_args, **_kwargs):
        self.connected = False

    def connect(self, **_kwargs):
        self.connected = True

    def query_orders(self, *, cancelable_only=False):
        return []

    def query_trades(self):
        return []


def _prepare_state(tmp_path: Path) -> tuple[Path, str, Path]:
    state_root = tmp_path / "state"
    account_key = account_execution_key(account_id="A001", account_type="STOCK")
    journal = (state_root / "journals" / f"{BATCH}.jsonl").resolve()
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        json.dumps({"batch_id": BATCH, "event": "EXECUTION_START"}) + "\n",
        encoding="utf-8",
    )
    reserve_execution_batch(
        state_root / "execution_batches",
        batch_id=BATCH,
        metadata={"account_key": account_key, "journal_path": str(journal)},
    )
    reserve_account_execution_lock(
        state_root,
        account_key=account_key,
        batch_id=BATCH,
    )
    return state_root, account_key, journal


def test_recovery_cli_audits_without_releasing_lock(tmp_path, monkeypatch):
    state_root, account_key, _journal = _prepare_state(tmp_path)
    output = tmp_path / "recovery.json"
    monkeypatch.setattr(reconcile_qmt_execution, "QmtBroker", _FakeBroker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconcile_qmt_execution.py",
            "--userdata",
            "unused",
            "--account",
            "A001",
            "--state-dir",
            str(state_root),
            "--output",
            str(output),
        ],
    )
    assert reconcile_qmt_execution.main() == 2
    assert load_account_execution_lock(state_root, account_key=account_key) is not None
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["safe_to_acknowledge"] is True
    assert report["acknowledgement"] == "NOT_REQUESTED"


def test_recovery_cli_requires_exact_batch_acknowledgement_to_release(tmp_path, monkeypatch):
    state_root, account_key, _journal = _prepare_state(tmp_path)
    output = tmp_path / "recovery.json"
    monkeypatch.setattr(reconcile_qmt_execution, "QmtBroker", _FakeBroker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconcile_qmt_execution.py",
            "--userdata",
            "unused",
            "--account",
            "A001",
            "--state-dir",
            str(state_root),
            "--output",
            str(output),
            "--acknowledge-batch",
            BATCH,
        ],
    )
    assert reconcile_qmt_execution.main() == 0
    assert load_account_execution_lock(state_root, account_key=account_key) is None
    marker = read_execution_state(state_root / "execution_batches" / f"{BATCH}.json")
    assert marker["status"] == "RECOVERED_RELEASED"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["acknowledgement"] == "ACCEPTED_LOCK_RELEASED"

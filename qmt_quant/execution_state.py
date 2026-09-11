from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


_TERMINAL_BATCH_STATUSES = frozenset({"COMPLETED", "INCOMPLETE", "BLOCKED_ACCOUNT_LOCK"})


def account_execution_key(*, account_id: str, account_type: str) -> str:
    """Return a stable non-plaintext account key for persistent execution state."""
    payload = {
        "account_id": str(account_id).strip(),
        "account_type": str(account_type).strip().upper(),
    }
    if not payload["account_id"] or not payload["account_type"]:
        raise ValueError("account_id and account_type must be non-empty")
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def execution_batch_id(
    *,
    signal_date: str,
    strategy_sha256: str,
    target_weights: Mapping[str, float],
    account_key: str | None = None,
) -> str:
    payload = {
        "signal_date": str(signal_date),
        "strategy_sha256": str(strategy_sha256),
        "target_weights": {
            str(code): float(target_weights[code]) for code in sorted(target_weights)
        },
    }
    if account_key:
        payload["account_key"] = str(account_key)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json_create(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(path), flags, 0o600)
    try:
        raw = (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_parent(path)


def read_execution_state(path: str | Path) -> dict:
    source = Path(path)
    if not source.exists():
        raise RuntimeError(f"execution state is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"execution state is unreadable: {source}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"execution state is invalid: {source}")
    return payload


def reserve_execution_batch(
    root: str | Path,
    *,
    batch_id: str,
    metadata: Mapping[str, object],
) -> Path:
    """Atomically reserve one live batch.

    The marker is deliberately persistent after success or failure. Re-running the
    exact account/strategy/signal/target bundle therefore requires explicit operator
    review rather than silently submitting the same batch twice.
    """
    directory = Path(root)
    path = directory / f"{batch_id}.json"
    payload = {
        "batch_id": batch_id,
        "status": "RESERVED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **dict(metadata),
    }
    try:
        _atomic_json_create(path, payload)
    except FileExistsError as exc:
        raise RuntimeError(
            f"execution batch {batch_id} already exists; reconcile it before any retry"
        ) from exc
    return path


def update_execution_batch(
    path: str | Path,
    *,
    status: str,
    details: Mapping[str, object] | None = None,
) -> None:
    target = Path(path)
    payload = read_execution_state(target)
    payload["status"] = str(status)
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if details:
        payload["details"] = dict(details)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
    _fsync_parent(target)


def account_lock_path(root: str | Path, *, account_key: str) -> Path:
    return Path(root) / "account_locks" / f"{str(account_key)}.json"


def load_account_execution_lock(root: str | Path, *, account_key: str) -> dict | None:
    path = account_lock_path(root, account_key=account_key)
    if not path.exists():
        return None
    payload = read_execution_state(path)
    if str(payload.get("account_key", "")) != str(account_key):
        raise RuntimeError("account execution lock identity mismatch")
    batch_id = str(payload.get("batch_id", ""))
    if not batch_id:
        raise RuntimeError("account execution lock is missing batch_id")
    return payload


def reserve_account_execution_lock(
    root: str | Path,
    *,
    account_key: str,
    batch_id: str,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically permit only one active live batch for an account.

    The lock is independent of the output/report directory so two executor invocations
    cannot bypass one another by choosing different ``--output`` paths.
    """
    path = account_lock_path(root, account_key=account_key)
    payload = {
        "account_key": str(account_key),
        "batch_id": str(batch_id),
        "status": "ACTIVE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **dict(metadata or {}),
    }
    try:
        _atomic_json_create(path, payload)
    except FileExistsError as exc:
        raise RuntimeError(
            f"account execution lock already exists for {account_key}; recover it before any new batch"
        ) from exc
    return path


def release_account_execution_lock(path: str | Path, *, batch_id: str) -> None:
    target = Path(path)
    payload = read_execution_state(target)
    if str(payload.get("batch_id", "")) != str(batch_id):
        raise RuntimeError("refusing to release account execution lock owned by another batch")
    target.unlink()
    _fsync_parent(target)


def batch_status_requires_recovery(status: str) -> bool:
    return str(status).upper() not in _TERMINAL_BATCH_STATUSES

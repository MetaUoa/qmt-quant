from __future__ import annotations

from pathlib import Path
import sys

import pytest

import run_acceptance


ROOT = Path(__file__).resolve().parents[1]


def test_acceptance_cli_has_no_legacy_v3_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_acceptance.py"])
    with pytest.raises(SystemExit):
        run_acceptance.parse_args()


def test_acceptance_requires_exact_lowercase_sha256():
    assert run_acceptance._require_strategy_sha("a" * 64) == "a" * 64
    with pytest.raises(ValueError, match="64-hex"):
        run_acceptance._require_strategy_sha("abc")
    with pytest.raises(ValueError, match="64-hex"):
        run_acceptance._require_strategy_sha("G" * 64)


def test_acceptance_source_has_no_v3_implicit_path():
    source = Path(run_acceptance.__file__).read_text(encoding="utf-8")
    assert "output/v3_research" not in source
    assert 'p.add_argument("--backtest", required=True)' in source
    assert 'p.add_argument("--walk-forward", required=True)' in source
    assert 'p.add_argument("--folds", required=True)' in source
    assert 'p.add_argument("--stress", required=True)' in source
    assert 'p.add_argument("--strategy-sha256", required=True)' in source


def test_legacy_acceptance_batch_is_fail_closed():
    batch = (ROOT / "run_acceptance.bat").read_text(encoding="utf-8")
    assert "output\\v3_research" not in batch
    assert "run_acceptance.py --backtest" not in batch
    assert "no longer supplies implicit V3/V4 evidence paths" in batch
    assert "exit /b 2" in batch

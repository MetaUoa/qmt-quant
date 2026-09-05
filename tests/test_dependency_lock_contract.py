from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements.txt"
CI = ROOT / ".github" / "workflows" / "tests.yml"


def test_all_repository_requirements_are_exactly_pinned():
    lines = [
        line.strip()
        for line in REQ.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines
    assert all("==" in line for line in lines)
    assert all(not re.search(r"(?<![=])[<>~!]", line) for line in lines)
    assert "baostock==0.9.3" in lines


def test_ci_installs_the_same_locked_requirements_on_all_python_versions():
    text = CI.read_text(encoding="utf-8")
    assert 'python-version: ["3.10", "3.11", "3.12"]' in text
    assert "python -m pip install -r requirements.txt" in text
    assert "python -m pip check" in text
    assert "python -m pip install numpy pandas pyarrow pytest pytest-cov" not in text

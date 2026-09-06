from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


_RAW_WORKFLOW_READ = re.compile(
    r"(?:\.github[/\\][\"']?workflows|workflows[\"']?\s*/).*?read_text",
    re.DOTALL,
)


def test_workflow_contract_tests_use_structured_loader() -> None:
    offenders: list[str] = []
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        if ".github" not in text or "workflows" not in text or "read_text" not in text:
            continue
        if _RAW_WORKFLOW_READ.search(text) or ".github/workflows" in text:
            offenders.append(path.name)
    assert not offenders, (
        "workflow contracts must parse YAML structure instead of asserting raw wording; "
        f"migrate: {', '.join(offenders)}"
    )

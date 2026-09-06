from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def _assigned_expressions(tree: ast.AST) -> dict[str, ast.AST]:
    assigned: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assigned[node.targets[0].id] = node.value
    return assigned


def _is_workflow_path(expr: ast.AST, assigned: dict[str, ast.AST], seen: set[str] | None = None) -> bool:
    seen = set() if seen is None else seen
    if isinstance(expr, ast.Name):
        if expr.id in seen or expr.id not in assigned:
            return False
        return _is_workflow_path(assigned[expr.id], assigned, seen | {expr.id})
    strings = [
        node.value
        for node in ast.walk(expr)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    joined = "/".join(strings).replace("\\", "/")
    return ".github" in joined and "workflows" in joined


def _raw_workflow_reads(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assigned = _assigned_expressions(tree)
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "read_text":
            continue
        if _is_workflow_path(node.func.value, assigned):
            count += 1
    return count


def test_workflow_contract_tests_use_structured_loader() -> None:
    offenders = [
        path.name
        for path in sorted(TESTS.glob("test_*.py"))
        if path.name != Path(__file__).name and _raw_workflow_reads(path)
    ]
    assert not offenders, (
        "workflow contracts must parse YAML structure instead of asserting raw wording; "
        f"migrate: {', '.join(offenders)}"
    )

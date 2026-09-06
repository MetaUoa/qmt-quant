from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class _GitHubWorkflowLoader(yaml.SafeLoader):
    """YAML loader that preserves GitHub's ``on`` key as a string.

    PyYAML's YAML 1.1 resolver otherwise interprets words such as ``on``/``off`` as
    booleans. GitHub Actions uses YAML 1.2-style keys, so contract tests must not
    silently turn the workflow trigger key into ``True``.
    """


for first_char, resolvers in list(_GitHubWorkflowLoader.yaml_implicit_resolvers.items()):
    _GitHubWorkflowLoader.yaml_implicit_resolvers[first_char] = [
        (tag, regexp)
        for tag, regexp in resolvers
        if tag != "tag:yaml.org,2002:bool"
    ]


def load_workflow(path: str | Path) -> dict[str, Any]:
    payload = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_GitHubWorkflowLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"workflow {path} must decode to a mapping")
    return payload


def workflow_events(workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    events = workflow.get("on")
    if not isinstance(events, Mapping):
        raise ValueError("workflow trigger must be a mapping")
    return events


def job(workflow: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    jobs = workflow.get("jobs")
    if not isinstance(jobs, Mapping) or name not in jobs or not isinstance(jobs[name], Mapping):
        raise KeyError(f"workflow job not found: {name}")
    return jobs[name]


def step(workflow: Mapping[str, Any], job_name: str, step_name: str) -> Mapping[str, Any]:
    steps = job(workflow, job_name).get("steps")
    if not isinstance(steps, list):
        raise KeyError(f"workflow job has no steps: {job_name}")
    matches = [row for row in steps if isinstance(row, Mapping) and row.get("name") == step_name]
    if len(matches) != 1:
        raise KeyError(f"expected exactly one step {step_name!r} in job {job_name!r}, found {len(matches)}")
    return matches[0]


def normalized_run(workflow: Mapping[str, Any], job_name: str, step_name: str) -> str:
    run = step(workflow, job_name, step_name).get("run")
    if not isinstance(run, str):
        raise ValueError(f"step {step_name!r} in {job_name!r} has no run command")
    return " ".join(run.split())


def env_value(workflow: Mapping[str, Any], key: str, *, job_name: str | None = None) -> str:
    source: Mapping[str, Any]
    if job_name is None:
        source = workflow
    else:
        source = job(workflow, job_name)
    env = source.get("env")
    if not isinstance(env, Mapping) or key not in env:
        raise KeyError(f"environment key not found: {key}")
    return str(env[key])


def matrix_values(workflow: Mapping[str, Any], job_name: str, key: str) -> list[str]:
    strategy = job(workflow, job_name).get("strategy")
    if not isinstance(strategy, Mapping):
        raise KeyError(f"job {job_name!r} has no strategy")
    matrix = strategy.get("matrix")
    if not isinstance(matrix, Mapping) or key not in matrix or not isinstance(matrix[key], list):
        raise KeyError(f"matrix key not found: {job_name}.{key}")
    return [str(value) for value in matrix[key]]


def max_parallel(workflow: Mapping[str, Any], job_name: str) -> int:
    strategy = job(workflow, job_name).get("strategy")
    if not isinstance(strategy, Mapping) or "max-parallel" not in strategy:
        raise KeyError(f"job {job_name!r} has no max-parallel")
    return int(strategy["max-parallel"])

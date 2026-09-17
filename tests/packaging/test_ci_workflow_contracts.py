"""Contracts over the CI workflows: what every job that collects the suite must do first."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
FIXTURE_ACTION = "./.github/actions/format2-module-fixture"
FIXTURE_TEST = REPO_ROOT / "tests" / "core" / "training" / "test_checkpoint_format2.py"


def _jobs() -> dict[str, dict[str, Any]]:
    """Every job of every workflow, keyed by ``<workflow file>:<job id>``."""
    jobs: dict[str, dict[str, Any]] = {}
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        for job_id, job in document.get("jobs", {}).items():
            jobs[f"{workflow.name}:{job_id}"] = job
    return jobs


def _run_lines(job: dict[str, Any]) -> list[str]:
    return [str(step.get("run", "")) for step in job.get("steps", [])]


def _collects_the_fixture_test(job: dict[str, Any]) -> bool:
    """Whether a pytest invocation in ``job`` collects the format-2 test's directory."""
    for run in _run_lines(job):
        if "pytest" not in run:
            continue
        paths = [token for token in run.split() if token.startswith("tests")]
        if any(FIXTURE_TEST.is_relative_to(REPO_ROOT / path) for path in paths):
            return True
    return False


def test_the_fixture_test_exists_where_the_contract_looks() -> None:
    assert FIXTURE_TEST.is_file()


def test_every_job_collecting_the_format2_test_writes_the_fixture_first() -> None:
    collecting = {name: job for name, job in _jobs().items() if _collects_the_fixture_test(job)}
    assert collecting, "no workflow job runs pytest over the training tests"
    for name, job in collecting.items():
        steps = job["steps"]
        fixture_steps = [i for i, step in enumerate(steps) if step.get("uses") == FIXTURE_ACTION]
        assert fixture_steps, f"{name} collects the format-2 test without writing its fixture"
        pytest_steps = [i for i, step in enumerate(steps) if "pytest" in str(step.get("run", ""))]
        assert fixture_steps[0] < min(pytest_steps), f"{name} runs pytest before the fixture"

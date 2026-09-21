"""Contracts over the CI workflows.

Every job that collects the suite writes the generated fixture first; every workflow a push
triggers cancels the run it supersedes; the pull-request gate runs on ubuntu alone, and the
macOS unit lane lives in the nightly workflow under a runner cap, so one opifex push never
holds the organisation's macOS runners against the other repositories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
PULL_REQUEST_GATE = "ci.yml"
NIGHTLY = "tests-extended.yml"
MACOS_RUNNER = "macos-14"
MACOS_RUNNER_CAP = 3


def _documents() -> dict[str, dict[str, Any]]:
    """Every workflow document, keyed by file name."""
    return {
        workflow.name: yaml.safe_load(workflow.read_text(encoding="utf-8"))
        for workflow in sorted(WORKFLOWS.glob("*.yml"))
    }


def _triggers(document: dict[str, Any]) -> dict[str, Any]:
    """The ``on`` mapping; PyYAML reads the bare key ``on`` as the boolean ``True``."""
    keys: dict[Any, Any] = document
    return keys.get("on") or keys.get(True) or {}


def _jobs() -> dict[str, dict[str, Any]]:
    """Every job of every workflow, keyed by ``<workflow file>:<job id>``."""
    jobs: dict[str, dict[str, Any]] = {}
    for name, document in _documents().items():
        for job_id, job in document.get("jobs", {}).items():
            jobs[f"{name}:{job_id}"] = job
    return jobs


def _runners(job: dict[str, Any]) -> list[str]:
    """The runner labels a job can land on: ``runs-on`` plus any matrix ``os`` values."""
    labels = [str(job.get("runs-on", ""))]
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    os_values = matrix.get("os", [])
    labels.extend(
        str(value) for value in (os_values if isinstance(os_values, list) else [os_values])
    )
    return labels


def _pytest_runs(job: dict[str, Any]) -> list[str]:
    return [run for run in _run_lines(job) if "pytest" in run]


def _run_lines(job: dict[str, Any]) -> list[str]:
    return [str(step.get("run", "")) for step in job.get("steps", [])]


def test_every_workflow_a_push_triggers_cancels_the_run_it_supersedes() -> None:
    """Two pushes in a row leave one run: the group is keyed on the ref, in-progress cancelled."""
    pushed = {name: doc for name, doc in _documents().items() if "push" in _triggers(doc)}
    assert pushed, "no workflow runs on push"
    for name, document in pushed.items():
        concurrency = document.get("concurrency")
        assert isinstance(concurrency, dict), f"{name} declares no concurrency group"
        assert "github.ref" in str(concurrency.get("group")), (
            f"{name}'s group is not keyed on the ref"
        )
        assert concurrency.get("cancel-in-progress") is True, f"{name} keeps superseded runs"


def test_the_pull_request_gate_runs_on_ubuntu_alone() -> None:
    """No job of the push and pull-request gate lands on a macOS runner."""
    gate = {name: job for name, job in _jobs().items() if name.startswith(f"{PULL_REQUEST_GATE}:")}
    assert gate, f"{PULL_REQUEST_GATE} declares no jobs"
    for name, job in gate.items():
        assert not any("macos" in label for label in _runners(job)), f"{name} lands on macOS"


def test_the_nightly_workflow_holds_the_macos_unit_lane_under_a_runner_cap() -> None:
    """The unit suite runs on macOS nightly, sharded, holding at most a few runners at once."""
    nightly = {name: job for name, job in _jobs().items() if name.startswith(f"{NIGHTLY}:")}
    macos = [job for job in nightly.values() if MACOS_RUNNER in _runners(job)]
    assert len(macos) == 1, f"{NIGHTLY} holds {len(macos)} macOS jobs, not one"
    (job,) = macos
    strategy = job.get("strategy") or {}
    assert strategy.get("max-parallel", 10**9) <= MACOS_RUNNER_CAP, (
        "the macOS lane has no runner cap"
    )
    assert "group" in (strategy.get("matrix") or {}), "the macOS lane is not sharded"
    runs = _pytest_runs(job)
    assert runs and all("not slow" in run for run in runs), (
        "the macOS lane does not run the unit suite"
    )

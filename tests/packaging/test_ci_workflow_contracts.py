"""Contracts over the CI workflows.

Every job that collects the suite writes the generated fixture first; every workflow a push
triggers cancels the run it supersedes; the pull-request gate runs on ubuntu alone, and the
macOS unit lane lives in the nightly workflow under a runner cap, so one opifex push never
holds the organisation's macOS runners against the other repositories. A merge whose pull request
already passed over the same tree does not repeat the test and quality jobs: they consult
substrax's already-tested action, pinned by commit, which compares only on a push.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
ACTIONS = REPO_ROOT / ".github" / "actions"
PULL_REQUEST_GATE = "ci.yml"
NIGHTLY = "tests-extended.yml"
MACOS_RUNNER = "macos-14"
MACOS_RUNNER_CAP = 3
GATE_JOB = "already_tested"
GATE_CONDITION = f"needs.{GATE_JOB}.outputs.skip != 'true'"
GATE_ACTION = re.compile(r"^avitai/substrax/\.github/actions/already-tested@[0-9a-f]{40}$")
GATED_WORKFLOWS = ("ci.yml", "quality-checks.yml")
# A report job runs after its producer whatever that producer's result; it still stands down
# with the rest when the merge repeats the pull request.
GATED_CONDITIONS = (GATE_CONDITION, f"always() && {GATE_CONDITION}")


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


def test_every_uv_cache_is_pruned_before_it_is_saved() -> None:
    """A saved uv cache holds only what uv built, not every wheel it downloaded.

    setup-uv prunes only when asked (``prune-cache`` defaults to false from v9); unpruned, the
    caches of the heavy extras grow to gigabytes each and evict the repository's other caches.
    """
    documents = [*sorted(WORKFLOWS.glob("*.yml")), *sorted(ACTIONS.glob("*/action.yml"))]
    checked = 0
    unpruned: list[str] = []
    for path in documents:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        owners = {**document.get("jobs", {}), "runs": document.get("runs") or {}}
        for owner, body in owners.items():
            for step in body.get("steps", []):
                if not str(step.get("uses", "")).startswith("astral-sh/setup-uv@"):
                    continue
                checked += 1
                if (step.get("with") or {}).get("prune-cache") is not True:
                    unpruned.append(f"{path.relative_to(REPO_ROOT)}:{owner}")

    assert checked, "no setup-uv step found; the contract is reading the wrong files"
    assert unpruned == [], f"setup-uv steps saving an unpruned cache: {unpruned}"


def _workflow_jobs(name: str) -> dict[str, dict[str, Any]]:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))["jobs"]


def _compare_step(name: str) -> dict[str, Any]:
    steps = _workflow_jobs(name)[GATE_JOB]["steps"]
    return next(step for step in steps if step.get("id") == "compare")


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_the_gate_compares_only_on_a_push(name: str) -> None:
    """A manual run re-measures on purpose; only a merge repeats a pull request."""
    gate = _workflow_jobs(name)[GATE_JOB]
    steps = [step for step in gate["steps"] if "already-tested" in str(step.get("uses", ""))]

    assert [step.get("if") for step in steps] == ["github.event_name == 'push'"]
    assert steps[0]["id"] in gate["outputs"]["skip"]


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_the_gate_is_the_shared_action_pinned_to_a_commit(name: str) -> None:
    """The compare is substrax's already-tested action, pinned by a full commit SHA.

    The action finds the pull request a push merged (squash or rebase) and skips only when that
    pull request tested this tree and every one of its checks succeeded; its rules are tested in
    substrax. A full commit SHA pins exactly the code that runs.
    """
    compare = _compare_step(name)

    assert GATE_ACTION.match(compare.get("uses", "")), compare.get("uses")
    assert "run" not in compare, "the gate runs the shared action, not an inline script"


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_every_job_that_repeats_the_pull_request_consults_the_gate(name: str) -> None:
    jobs = _workflow_jobs(name)
    ungated = sorted(
        job_id
        for job_id, job in jobs.items()
        if job_id != GATE_JOB
        and (job.get("if") not in GATED_CONDITIONS or GATE_JOB not in job.get("needs", []))
    )

    assert ungated == [], f"{name}: these repeat the pull request without the gate: {ungated}"


@pytest.mark.parametrize("name", GATED_WORKFLOWS)
def test_an_unanswered_gate_leaves_the_work_running(name: str) -> None:
    """An empty output (no compare, or a lookup that failed) reads as "test it"."""
    for job_id, job in _workflow_jobs(name).items():
        if job_id == GATE_JOB:
            continue
        text = yaml.safe_dump(job)
        assert "outputs.skip == " not in text, f"{job_id} tests the gate for equality"
        assert "outputs.skip != 'false'" not in text, f"{job_id} runs only on an explicit false"

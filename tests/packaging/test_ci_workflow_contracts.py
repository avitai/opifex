"""Contracts over the CI workflows.

Every job that collects the suite writes the generated fixture first; every workflow a push
triggers cancels the run it supersedes; the pull-request gate runs on ubuntu alone, and the
macOS unit lane lives in the nightly workflow under a runner cap, so one opifex push never
holds the organisation's macOS runners against the other repositories.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
FIXTURE_ACTION = "./.github/actions/format2-module-fixture"
FIXTURE_TEST = REPO_ROOT / "tests" / "core" / "training" / "test_checkpoint_format2.py"
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


FIXTURE_INPUTS = "scripts/format2_fixture_requirements.in"
FIXTURE_LOCK = "scripts/format2_fixture_requirements.txt"
_PINNED = re.compile(r"^[A-Za-z0-9_.\-]+(\[[^\]]+\])?==\S+")
_FIXTURE_ROOT = Path(__file__).resolve().parents[2]


def _requirement_lines(relative_path: str) -> list[str]:
    text = (_FIXTURE_ROOT / relative_path).read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_the_fixture_action_runs_the_generator_under_the_committed_lock() -> None:
    """The fixture environment is a committed lock: it neither floats between two jobs of one
    run (jax 0.11.2 broke flax 0.12.9's import that way) nor drifts with the project's lock."""
    action = yaml.safe_load(
        (_FIXTURE_ROOT / ".github/actions/format2-module-fixture" / "action.yml").read_text(
            encoding="utf-8"
        )
    )
    runs = [str(step.get("run", "")) for step in action["runs"]["steps"]]
    assert any(f"--with-requirements {FIXTURE_LOCK}" in run for run in runs), (
        "the action resolves the environment"
    )
    assert not any(re.search(r"--with\s", run) for run in runs), (
        "the action adds a floating package"
    )


def test_every_package_of_the_fixture_lock_is_pinned_and_the_inputs_are_kept() -> None:
    locked = _requirement_lines(FIXTURE_LOCK)
    assert locked, "the fixture lock is empty"
    for line in locked:
        assert _PINNED.match(line), f"unpinned requirement in the fixture lock: {line}"
    bare = {entry.split(";")[0].strip().lower() for entry in locked}
    for line in _requirement_lines(FIXTURE_INPUTS):
        assert _PINNED.match(line), f"an input is not an exact pin: {line}"
        assert line.lower() in bare, line
    assert {"jax", "jaxlib", "flax", "orbax-checkpoint"} <= {
        re.split(r"[\[=]", line)[0].lower() for line in locked
    }


SMOKE_MODULE = "opifex.core.training.trainer"


def test_the_fresh_install_smoke_reads_pypi_and_imports_the_numerical_stack() -> None:
    """The build-verification smoke installs the wheel fresh and imports a module that loads
    flax.nnx: `import opifex` alone passed with a jax that broke flax's import."""
    job = _jobs()["build-verification.yml:build"]
    smoke = [run for run in _run_lines(job) if "dist/*.whl" in run]
    assert len(smoke) == 1, "build-verification has no fresh-install smoke"
    assert "uv pip install --refresh dist/*.whl" in smoke[0], "the smoke reads a restored cache"
    assert f"import opifex, {SMOKE_MODULE}" in smoke[0], (
        "the smoke never reaches the numerical stack"
    )

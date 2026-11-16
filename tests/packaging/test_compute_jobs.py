"""Repository contracts for the compute jobs in ``[tool.substrax.compute]``.

A job's table is read only when the job is submitted, so a key substrax refuses, or an example
whose budget its task limit cannot hold, stays hidden until a GPU run fails at submission. These
tests read every job as ``substrax-compute`` reads it.
"""

from pathlib import Path

import pytest
from substrax.compute import JobSpec
from substrax.compute.config import read_compute_config, resolve_job
from substrax.examples import discover_examples


REPO_ROOT = Path(__file__).resolve().parents[2]
JOB_NAMES = sorted(read_compute_config(REPO_ROOT).jobs)


def _resolve(name: str) -> JobSpec:
    return resolve_job(read_compute_config(REPO_ROOT), name, project=REPO_ROOT)


@pytest.mark.parametrize("name", JOB_NAMES)
def test_job_resolves_as_substrax_submits_it(name: str) -> None:
    """Each job resolves into at least one task, its examples held to budgets it can run."""
    assert _resolve(name).tasks


@pytest.mark.parametrize("name", JOB_NAMES)
def test_cuda_job_also_starts_the_cpu_backend(name: str) -> None:
    """A job on ``cuda`` lists ``cpu`` too: with ``cuda`` alone jax starts no CPU backend,
    and an example's first host callback fails.
    """
    platforms = _resolve(name).runtime.platforms or ()
    if "cuda" in platforms:
        assert "cpu" in platforms


def test_every_example_is_a_task_of_exactly_one_job() -> None:
    """The jobs together run each example once."""
    names = [task.name for name in JOB_NAMES for task in _resolve(name).tasks]
    assert len(names) == len(set(names))
    assert len(names) == len(discover_examples(REPO_ROOT / "examples"))

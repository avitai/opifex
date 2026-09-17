"""Benchmark output directories resolve through ``substrax.artifacts``.

With no directory given, a runner, evaluator or results manager writes under
``$AVITAI_OUTPUT_DIR/benchmarks`` when the variable is set and under a per-process
temporary directory otherwise; nothing defaults into the working tree. An explicit
directory is resolved against the working directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from opifex.benchmarking.benchmark_runner import BenchmarkRunner
from opifex.benchmarking.evaluation_engine import BenchmarkEvaluator
from opifex.benchmarking.results_manager import ResultsManager


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "outputs"
    root.mkdir()
    monkeypatch.setenv("AVITAI_OUTPUT_DIR", str(root))
    return root


def _runner(**kwargs: object) -> BenchmarkRunner:
    with patch("opifex.benchmarking.benchmark_runner.OperatorBenchmarkRegistry") as registry:
        registry.return_value.list_available_operators.return_value = []
        return BenchmarkRunner(**kwargs)  # pyright: ignore[reportArgumentType]


def test_results_manager_defaults_under_the_output_variable(output_root: Path) -> None:
    manager = ResultsManager()
    assert manager.storage_path == output_root / "benchmarks"
    assert manager.database_path == output_root / "benchmarks" / "benchmark_database.json"
    assert manager.raw_results_path.is_dir()


def test_evaluator_defaults_under_the_output_variable(output_root: Path) -> None:
    evaluator = BenchmarkEvaluator()
    assert evaluator.output_dir == output_root / "benchmarks"
    assert evaluator.raw_results_dir == output_root / "benchmarks" / "raw_evaluations"


def test_runner_defaults_under_the_output_variable_and_shares_it(output_root: Path) -> None:
    runner = _runner()
    assert runner.output_dir == output_root / "benchmarks"
    assert runner.evaluator.output_dir == runner.output_dir
    assert runner.results_manager.storage_path == runner.output_dir


def test_defaults_never_land_in_the_working_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AVITAI_OUTPUT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    for path in (
        ResultsManager().storage_path,
        BenchmarkEvaluator().output_dir,
        _runner().output_dir,
    ):
        assert path.is_absolute()
        assert tmp_path not in path.parents, path
        assert path.name == "benchmarks"


def test_explicit_directory_resolves_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert ResultsManager(storage_path="results").storage_path == tmp_path / "results"
    assert BenchmarkEvaluator(output_dir="results").output_dir == tmp_path / "results"
    assert _runner(output_dir=tmp_path / "runs").output_dir == tmp_path / "runs"

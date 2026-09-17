"""The example contract, checked on every script under ``examples/``.

Every example defines ``main() -> dict`` and guards its execution under
``if __name__ == "__main__":`` (``docs/development/example-documentation-design.md``).
The cheap test reads the source. The slow test runs the example in its own
interpreter through ``substrax.testing.run_example``, on the CPU backend, with its
outputs under a temporary directory, and asserts the returned metrics are finite.
The per-example budget is the suite's ``--timeout`` (``pyproject.toml``); the runner
enforces it and names it, so pytest-timeout is off for that test.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
from substrax.testing import discover_examples, run_example


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = discover_examples(ROOT / "examples")
IDS = [path.relative_to(ROOT / "examples").as_posix() for path in EXAMPLES]


@pytest.fixture
def output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh output directory, with ``AVITAI_OUTPUT_DIR`` pointing at it."""
    path = tmp_path / "outputs"
    path.mkdir()
    monkeypatch.setenv("AVITAI_OUTPUT_DIR", str(path))
    return path


def test_examples_are_discovered() -> None:
    assert len(EXAMPLES) > 50


@pytest.mark.parametrize("path", EXAMPLES, ids=IDS)
def test_example_defines_main_behind_a_guard(path: Path) -> None:
    """Each example defines ``main`` and runs it only under the ``__main__`` guard."""
    tree = ast.parse(path.read_text(), filename=str(path))
    defines_main = any(
        isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body
    )
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
    ]
    assert defines_main, f"{path} must define main()"
    assert guards, f"{path} must run main() under an if __name__ == '__main__' guard"


@pytest.mark.slow
@pytest.mark.timeout(0)
@pytest.mark.parametrize("path", EXAMPLES, ids=IDS)
def test_example_runs_and_returns_finite_metrics(
    path: Path, output_dir: Path, request: pytest.FixtureRequest
) -> None:
    """``main()`` completes in a child interpreter and returns a dict of finite metrics."""
    budget = float(request.config.getoption("timeout"))
    run = run_example(path, repo_root=ROOT, output_dir=output_dir, timeout=budget, call_main=True)
    assert run.result.check().returncode == 0
    summary = run.summary
    assert isinstance(summary, dict) and summary, f"{path}: main() must return a non-empty dict"
    for key, value in summary.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue  # labels, shapes and lists are allowed
        assert math.isfinite(value), f"{path}: metric {key!r} is not finite ({value})"

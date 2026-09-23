"""Every console command the package declares is callable from an installed wheel.

A ``[project.scripts]`` entry is part of the published interface: installing the package
puts the command on the user's PATH. The wheel carries ``src/opifex`` alone, so an entry
pointing anywhere else -- a repository ``scripts/`` directory, a module that was never
written -- installs a command that fails at start with ``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _console_scripts() -> dict[str, str]:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    return dict(project.get("scripts", {}))


def _packaged_roots() -> list[str]:
    wheel = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["build"][
        "targets"
    ]["wheel"]
    return [Path(package).name for package in wheel["packages"]]


@pytest.mark.parametrize(("command", "target"), sorted(_console_scripts().items()))
def test_the_target_is_inside_the_wheel(command: str, target: str) -> None:
    module = target.split(":", 1)[0]
    assert module.split(".", 1)[0] in _packaged_roots(), (
        f"{command} points at {module}, which the wheel does not contain"
    )


@pytest.mark.parametrize(("command", "target"), sorted(_console_scripts().items()))
def test_the_target_resolves_to_a_callable(command: str, target: str) -> None:
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attribute)), f"{command}: {target} is not callable"


def test_the_benchmark_command_is_the_benchmarking_cli() -> None:
    assert _console_scripts()["opifex-benchmark"] == "opifex.benchmarking.cli:main"

"""Repository contracts for local MkDocs tooling."""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def test_docs_extra_declares_mkdocs_toolchain() -> None:
    """The docs extra must expose the MkDocs toolchain used by local docs commands."""

    pyproject = _pyproject()

    optional_dependencies = pyproject["project"]["optional-dependencies"]
    docs_extra = set(optional_dependencies["docs"])

    required_docs_packages = {
        "mkdocs>=1.6.1",
        "mkdocs-material>=9.6.7",
        "mkdocstrings>=0.28.3",
        "mkdocstrings-python>=1.1.2",
        "pymdown-extensions>=10.14.3",
    }

    assert required_docs_packages <= docs_extra


def test_local_docs_commands_select_docs_extra() -> None:
    """Local MkDocs commands should select the existing docs extra explicitly."""

    installation = (
        REPO_ROOT / "docs" / "getting-started" / "installation.md"
    ).read_text()

    assert "uv run --extra docs mkdocs serve" in installation
    assert "uv run --extra docs mkdocs build" in installation


def test_mkdocs_config_does_not_load_polyfill_io() -> None:
    """The docs site must not load JavaScript from the unsafe polyfill.io domain."""

    mkdocs_config = (REPO_ROOT / "mkdocs.yml").read_text()

    assert "polyfill.io" not in mkdocs_config


def test_training_api_mkdocstrings_targets_are_importable() -> None:
    """Training API mkdocstrings directives must target real modules."""

    training_api = (REPO_ROOT / "docs" / "api" / "training.md").read_text()
    targets = re.findall(r"^:::\s+(opifex\.[\w.]+)$", training_api, re.MULTILINE)

    assert targets

    for target in targets:
        importlib.import_module(target)

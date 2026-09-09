"""The derived-status check fails when the README's operator bullet drifts from the registry."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from opifex.neural.operators import OPERATOR_REGISTRY


if TYPE_CHECKING:
    from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def derive_status() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "derive_status", REPO_ROOT / "scripts" / "derive_status.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["derive_status"] = module
    spec.loader.exec_module(module)
    return module


def test_the_readme_names_the_registered_operators(derive_status: ModuleType) -> None:
    readme = (REPO_ROOT / "README.md").read_text()

    assert derive_status.readme_drift(readme, OPERATOR_REGISTRY) == []
    assert derive_status.render_operator_line(OPERATOR_REGISTRY) in readme.splitlines()


def test_a_stale_count_or_name_is_drift(derive_status: ModuleType) -> None:
    line = derive_status.render_operator_line(OPERATOR_REGISTRY)
    stale_count = line.replace(
        f"({len(OPERATOR_REGISTRY)} registered", f"({len(OPERATOR_REGISTRY) + 1} registered"
    )
    stale_name = line.replace("FNO, ", "FNO, DISCO, ", 1)

    for stale in (stale_count, stale_name):
        (drift,) = derive_status.readme_drift(stale + "\n", OPERATOR_REGISTRY)
        assert "drifted" in drift
    (missing,) = derive_status.readme_drift("# nothing here\n", OPERATOR_REGISTRY)
    assert "no line" in missing


def test_every_registered_operator_has_a_row(derive_status: ModuleType) -> None:
    rows = derive_status.operator_rows()

    assert [row.key for row in rows] == list(OPERATOR_REGISTRY)
    assert all(row.categories for row in rows)
    assert all(row.uncertainty_strategy for row in rows)

"""The import-layers contract covers every top-level package.

import-linter's ``layers`` contract only constrains the modules it lists, so a package left
out of the list could import anything. This holds the list to the package tree.
"""

import re
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "opifex"


def _layered_names() -> set[str]:
    contracts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["importlinter"][
        "contracts"
    ]
    (layers,) = [c["layers"] for c in contracts if c["type"] == "layers"]
    return {name.strip() for layer in layers for name in re.split(r"[|:]", layer)}


def _top_level_modules() -> set[str]:
    packages = {p.name for p in PACKAGE_ROOT.iterdir() if (p / "__init__.py").is_file()}
    modules = {
        p.stem for p in PACKAGE_ROOT.glob("*.py") if p.name not in {"__init__.py", "__main__.py"}
    }
    return packages | modules


def test_the_tree_has_modules() -> None:
    """Control: the tree scan finds the packages, so an empty difference below means something."""
    assert {"core", "neural", "capabilities"} <= _top_level_modules()


def test_every_top_level_module_is_layered() -> None:
    unlisted = _top_level_modules() - _layered_names()
    assert not unlisted, f"top-level modules missing from the layers contract: {sorted(unlisted)}"


def test_the_contract_names_nothing_that_does_not_exist() -> None:
    stale = _layered_names() - _top_level_modules()
    assert not stale, f"layers contract names modules that do not exist: {sorted(stale)}"

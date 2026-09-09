#!/usr/bin/env python3
"""Derive the neural-operator catalogue from the registry and check the README against it.

Two run modes::

    uv run python scripts/derive_status.py            # print the derived catalogue
    uv run python scripts/derive_status.py --check     # exit 1 on README drift (CI)

``opifex.neural.operators.OPERATOR_REGISTRY`` is the one place an operator
architecture is registered; ``create_operator`` and ``list_operators`` read it.
The README's Neural Operators bullet names that registry's keys and count, so it
is rendered from the registry here and compared verbatim.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping


logger = logging.getLogger("derive_status")

REPO_ROOT = Path(__file__).resolve().parent.parent
BULLET_PREFIX = "- **Neural Operators**: "
REGISTRY_PATH = "opifex.neural.operators.OPERATOR_REGISTRY"


@dataclass(frozen=True, slots=True, kw_only=True)
class OperatorRow:
    """One registered operator architecture."""

    key: str
    class_name: str
    module: str
    categories: tuple[str, ...]
    uncertainty_strategy: str


def render_operator_line(registry: Mapping[str, type]) -> str:
    """The README bullet the registry implies."""
    keys = ", ".join(registry)
    return f"{BULLET_PREFIX}{keys} ({len(registry)} registered architectures; `{REGISTRY_PATH}`)"


def readme_drift(readme_text: str, registry: Mapping[str, type]) -> list[str]:
    """Why the README's Neural Operators bullet does not match the registry; empty when it does."""
    expected = render_operator_line(registry)
    bullets = [line for line in readme_text.splitlines() if line.startswith(BULLET_PREFIX)]
    if not bullets:
        return [f"README has no line starting with {BULLET_PREFIX!r}; expected: {expected}"]
    if bullets != [expected]:
        return [
            f"README Neural Operators bullet drifted.\n  found:    {bullets[0]}\n  expected: {expected}"
        ]
    return []


def operator_rows() -> list[OperatorRow]:
    """The registry as rows: class, module, ``list_operators`` categories and UQ strategy."""
    from opifex.neural.operators import (
        get_operator_capability,
        list_operators,
        OPERATOR_REGISTRY,
    )

    categories = list_operators()
    return [
        OperatorRow(
            key=key,
            class_name=cls.__name__,
            module=cls.__module__,
            categories=tuple(name for name, keys in categories.items() if key in keys),
            uncertainty_strategy=get_operator_capability(key).default_strategy.value,
        )
        for key, cls in OPERATOR_REGISTRY.items()
    ]


def _print_catalogue(rows: list[OperatorRow]) -> None:
    """Print the catalogue as a table."""
    width = max(len(row.key) for row in rows)
    print(f"{len(rows)} registered operator architectures ({REGISTRY_PATH})")
    for row in rows:
        categories = ", ".join(row.categories) or "-"
        print(
            f"{row.key:<{width}}  {row.class_name:<36} {row.uncertainty_strategy:<24} {categories}"
        )


def main() -> int:
    """Print the catalogue, or check the README against it with ``--check``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 when the README drifts")
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from opifex.neural.operators import OPERATOR_REGISTRY

    if not args.check:
        _print_catalogue(operator_rows())
        return 0
    drift = readme_drift(args.readme.read_text(), OPERATOR_REGISTRY)
    for message in drift:
        logger.error(message)
    if not drift:
        logger.info("README names the %d registered operator architectures", len(OPERATOR_REGISTRY))
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())

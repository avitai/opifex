"""Shared utilities for example smoke tests.

Examples live under ``examples/`` in directories whose names may contain
hyphens (``neural-operators``), so they cannot be imported via the normal
dotted-module machinery. This loader executes an example file in isolation by
path. Loading is cheap and side-effect-free **only** for examples that guard
their execution under ``if __name__ == "__main__":`` and expose a callable
``main()`` — which is the contract every example must follow (see
``docs/development/example-documentation-design.md``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples"


def discover_examples() -> list[Path]:
    """Return every runnable example script under ``examples/``."""
    return sorted(
        p
        for p in EXAMPLES_ROOT.rglob("*.py")
        if p.name != "__init__.py" and "_common" not in p.parts
    )


def load_example(path: Path) -> ModuleType:
    """Import an example module by file path without running its ``main()``.

    Safe only when the example guards execution under ``__main__``; importing an
    unguarded flat script would run its full training body.
    """
    module_name = f"_example_{path.relative_to(EXAMPLES_ROOT).as_posix().replace('/', '_').removesuffix('.py')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

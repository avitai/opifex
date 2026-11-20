"""Import every module of the installed ``opifex`` package and fail on any import error.

Run it with the interpreter of the environment to check::

    python scripts/check_every_module_imports.py

The walk measures the ``opifex`` that interpreter resolves, which it prints first: the
repository's ``src`` tree under ``uv run``, the installed wheel in a clean environment.
Run from a path, the script puts only ``scripts/`` on ``sys.path``, never the repository
root, so a clean environment is measured as a user installs it.

A module is skipped only when its ``ModuleNotFoundError`` names a root in
:data:`OPTIONAL_THIRD_PARTY_ROOTS`; every other failure, a ``ModuleNotFoundError`` naming
an ``opifex`` module included, propagates and the script exits nonzero.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass

import opifex


#: Third-party roots a module may import at module level while only some extras install
#: them. Today that is ``ase`` alone, for ``opifex.deployment.ase_calculator``, which
#: subclasses ``ase.calculators.calculator.Calculator`` and so cannot defer the import.
#: ``pyscf`` is not here because ``qh9_eval`` imports it inside a function, which no
#: import-time walk reaches.
#:
#: **The list is deliberately this short.** A module that newly imports an optional
#: dependency at module level fails the walk rather than being skipped: the skip is keyed
#: to the name that was missing, never to the module that failed.
OPTIONAL_THIRD_PARTY_ROOTS = ("ase",)

#: A walk returning fewer modules than this is not walking the tree, and would pass by
#: importing almost nothing. A chosen round number well under the 601 modules measured,
#: not a snapshot of that count: it catches a broken walk, not the package's size.
MINIMUM_MODULES_WALKED = 100


@dataclass(frozen=True, slots=True, kw_only=True)
class WalkReport:
    """What one walk over ``opifex`` reached."""

    walked: int
    imported: int
    skipped: tuple[str, ...]


def walk_every_module() -> WalkReport:
    """Import every module under ``opifex``, skipping only missing optional roots.

    Returns:
        The count of modules walked and imported, and each skipped module with the
        optional root it was missing.

    Raises:
        RuntimeError: If the walk reaches fewer than :data:`MINIMUM_MODULES_WALKED`
            modules, or its counts do not account for each other.
    """
    walked, imported, skipped = 0, 0, []
    for info in pkgutil.walk_packages(opifex.__path__, "opifex."):
        walked += 1
        try:
            importlib.import_module(info.name)
        except ModuleNotFoundError as missing:
            root = (missing.name or "").split(".")[0]
            if root not in OPTIONAL_THIRD_PARTY_ROOTS:
                raise
            skipped.append(f"{info.name} <- {root}")
            continue
        imported += 1
    if walked < MINIMUM_MODULES_WALKED:
        msg = (
            f"the walk reached only {walked} modules, so it is not measuring the package; "
            "a walk that imports almost nothing passes for the wrong reason"
        )
        raise RuntimeError(msg)
    if imported + len(skipped) != walked:
        msg = f"{walked} walked but {imported} imported and {len(skipped)} skipped"
        raise RuntimeError(msg)
    return WalkReport(walked=walked, imported=imported, skipped=tuple(skipped))


def main() -> int:
    """Walk the resolved ``opifex`` and print what the walk reached.

    Returns:
        The process exit status, 0 when every module imported or was skipped.
    """
    print(f"opifex {opifex.__version__} from {opifex.__file__}")
    report = walk_every_module()
    print(
        f"walked {report.walked} imported {report.imported} "
        f"skipped {','.join(report.skipped) or '-'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

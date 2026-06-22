"""Static-scan tests enforcing call-time RNG ownership in production UQ paths.

Hidden fixed PRNG keys (``jax.random.PRNGKey(0)``, ``PRNGKey(42)``, etc.)
in stochastic / prediction / sampling code paths collapse Monte-Carlo
samples to identical predictions and silently destroy the variance that
makes a Bayesian model useful. Tests and examples may use fixed keys; the
production paths under ``src/opifex/uncertainty``,
``src/opifex/neural/bayesian``, and
``src/opifex/neural/operators/specialized/uqno.py`` must not.

The scan is AST-based: it walks the module syntax tree and reports every
``Call`` node whose ``func.attr == "PRNGKey"``. Plain-text scans
false-positive on docstrings and comments — module docstrings legitimately
describe the rule in prose.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest  # noqa: TC002  # pyproject dep kept eager; used in test annotations


# Dotted module names whose on-disk source must not contain
# ``jax.random.PRNGKey(...)`` call sites. A package resolves to its directory
# (recursively scanned); a module resolves to its single source file.
#
# Roots are resolved from the *installed* package via ``importlib.util.find_spec``
# rather than from CWD-relative ``Path("src/opifex/...")`` literals. The literals
# silently resolve to nothing whenever pytest is invoked from any directory other
# than the repository root, which made the central PRNGKey guard pass vacuously
# (it asserted ``not []`` against an empty scan). ``find_spec`` reads each
# module's ``origin`` without executing the module body, so resolution is both
# CWD-independent and free of import-time JAX side effects.
_SCANNED_MODULES: tuple[str, ...] = (
    "opifex.uncertainty",
    "opifex.neural.bayesian",
    "opifex.neural.operators.specialized.uqno",
    "opifex.optimization.l2o",
)


def _resolve_scan_roots() -> tuple[Path, ...]:
    """Resolve each scanned module name to its on-disk path from the package.

    A package's ``origin`` is its ``__init__.py``; the directory containing it is
    scanned recursively. A plain module's ``origin`` is its single ``.py`` file.
    """
    roots: list[Path] = []
    for module_name in _SCANNED_MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(
                f"RNG-safety scan could not resolve {module_name!r} from the "
                "installed package; the scan roots are misconfigured."
            )
        origin = Path(spec.origin).resolve()
        roots.append(origin.parent if origin.name == "__init__.py" else origin)
    return tuple(roots)


# Allowlist for currently-known violations that will be eliminated by an
# upcoming migration task. Each entry is ``(file_path, line_number, reason)``.
# When the migration lands, the corresponding entry MUST be removed in the
# same commit; the test will then enforce that no new violations appear.
#
# The allowlist is intentionally narrow: only known migration-blocked sites
# qualify, and each is pinned by exact line number.
_KNOWN_MIGRATION_BLOCKED: frozenset[tuple[str, int]] = frozenset()
# No production UQ paths currently require an allowlist entry. The
# previous UQNO entries were removed when Phase 3 migrated UQNO to the
# shared Bayesian layers (caller-owned RNG at every stochastic boundary).


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return ``[(line_no, source_excerpt), ...]`` for every PRNGKey call in ``path``."""
    if path.suffix != ".py":
        return []
    tree = ast.parse(path.read_text())
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "PRNGKey"
        ):
            hits.append((node.lineno, ast.unparse(node)))
    return hits


def _collect_python_files() -> list[Path]:
    """Resolve every production source file under the scanned roots.

    Fails loudly if resolution yields nothing: an empty scan must surface as a
    hard error here, never as a silent vacuous pass in the PRNGKey guard.
    """
    files: list[Path] = []
    for target in _resolve_scan_roots():
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
        elif target.is_file():
            files.append(target)
    assert files, (
        "RNG-safety scan found no files — path resolution broke. Expected the "
        f"installed package to expose {_SCANNED_MODULES!r}."
    )
    return files


def test_rng_safety_scan_finds_files_regardless_of_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The scan must resolve real files from the package, not the process CWD.

    Regression guard for a vacuous pass: when the scan roots were
    ``Path("src/opifex/...")`` literals (CWD-relative), invoking pytest from
    any directory other than the repository root made ``_collect_python_files``
    resolve to nothing. The central PRNGKey guard then asserted ``not []`` and
    passed without inspecting a single line of production code. Resolving the
    roots from the installed package makes the scan independent of where the
    test runner happens to be launched.
    """
    monkeypatch.chdir(tmp_path)
    files = _collect_python_files()
    assert files, (
        "RNG-safety scan resolved zero files from a non-repo CWD "
        f"({tmp_path}); scan roots must derive from the installed package, "
        "not the process working directory."
    )
    assert all(path.is_file() for path in files), (
        "scan resolved a path that is not a real file: "
        + ", ".join(str(path) for path in files if not path.is_file())
    )
    assert all(path.suffix == ".py" for path in files)


def test_no_unallowlisted_prngkey_in_production_uq_paths() -> None:
    """Every ``jax.random.PRNGKey(...)`` call must either be in the allowlist or removed."""
    offending: list[str] = []
    for path in _collect_python_files():
        for line_no, source in _scan_file(path):
            key = (str(path), line_no)
            if key in _KNOWN_MIGRATION_BLOCKED:
                continue
            offending.append(f"{path}:{line_no}: {source}")
    assert not offending, (
        "Production UQ paths contain unallowlisted jax.random.PRNGKey(...) "
        "call sites. Either thread caller-owned RNG through these sites or "
        "add the (path, line) tuple to _KNOWN_MIGRATION_BLOCKED with a "
        "comment naming the migration task that will close it.\n" + "\n".join(offending)
    )


def test_allowlist_is_still_needed() -> None:
    """Each allowlist entry must still correspond to an actual PRNGKey call.

    Prevents stale allowlist drift when a migration lands but the allowlist
    entry is forgotten — without this test the safety scan would silently
    permit re-introduction of a fixed key on the same line later.
    """
    stale: list[str] = []
    by_path: dict[str, set[int]] = {}
    for path_str, line_no in _KNOWN_MIGRATION_BLOCKED:
        by_path.setdefault(path_str, set()).add(line_no)
    for path_str, expected_lines in by_path.items():
        path = Path(path_str)
        actual_lines = {line_no for line_no, _ in _scan_file(path)}
        for expected in expected_lines:
            if expected not in actual_lines:
                stale.append(
                    f"{path_str}:{expected} — allowlisted but no PRNGKey "
                    "call at that line anymore. Remove the entry."
                )
    assert not stale, "\n".join(stale)

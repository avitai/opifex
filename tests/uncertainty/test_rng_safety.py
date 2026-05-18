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
from pathlib import Path


# Paths whose Python modules must not contain ``jax.random.PRNGKey(...)``
# call sites. Each entry is either a directory (recursively scanned) or a
# single file.
_SCANNED_PATHS: tuple[Path, ...] = (
    Path("src/opifex/uncertainty"),
    Path("src/opifex/neural/bayesian"),
    Path("src/opifex/neural/operators/specialized/uqno.py"),
)


# Allowlist for currently-known violations that will be eliminated by an
# upcoming migration task. Each entry is ``(file_path, line_number, reason)``.
# When the migration lands, the corresponding entry MUST be removed in the
# same commit; the test will then enforce that no new violations appear.
#
# The allowlist is intentionally narrow: only known migration-blocked sites
# qualify, and each is pinned by exact line number.
_KNOWN_MIGRATION_BLOCKED: frozenset[tuple[str, int]] = frozenset(
    {
        # UQNO ships its own copies of BayesianLinear / BayesianSpectralConvolution
        # and a predict_with_uncertainty fallback that defaults to PRNGKey(0)
        # when the caller omits the key. The UQNO migration to the shared
        # Bayesian layers eliminates all three sites.
        ("src/opifex/neural/operators/specialized/uqno.py", 105),
        ("src/opifex/neural/operators/specialized/uqno.py", 218),
        ("src/opifex/neural/operators/specialized/uqno.py", 630),
    }
)


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
    files: list[Path] = []
    for target in _SCANNED_PATHS:
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
        elif target.is_file():
            files.append(target)
    return files


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

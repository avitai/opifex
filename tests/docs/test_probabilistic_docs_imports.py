"""Verify that public Bayesian / probabilistic docs reference real symbols.

This test enforces two contracts for the three Bayesian-facing public docs:

1. Every ``from opifex.…`` or ``import opifex.…`` line inside a Python fenced
   code block must resolve via :func:`importlib.import_module`. Failure means
   the docs advertise an API surface that no longer exists.
2. The boilerplate / AI / phase / percentage / "total lines" patterns called
   out in the engineering rules must not appear in any of the three files.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

DOC_FILES = (
    REPO_ROOT / "src" / "opifex" / "neural" / "bayesian" / "README.md",
    REPO_ROOT / "src" / "opifex" / "neural" / "quantum" / "README.md",
    REPO_ROOT / "docs" / "methods" / "probabilistic.md",
)

FENCED_BLOCK_PATTERN = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL,
)

IMPORT_LINE_PATTERN = re.compile(
    r"^\s*(?:from\s+(opifex(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+import\s+([^\n#]+)"
    r"|import\s+(opifex(?:\.[A-Za-z_][A-Za-z0-9_]*)*))",
)

# Patterns that must NOT appear anywhere in the three docs. Each is a
# (label, compiled-regex) pair so the failure message can pinpoint the
# offender.
BOILERPLATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("memory-bank reference", re.compile(r"memory-bank")),
    ("Claude reference", re.compile(r"Claude")),
    ("Anthropic reference", re.compile(r"Anthropic")),
    ("internal audit reference", re.compile(r"internal audit")),
    ("Phase N reference", re.compile(r"\bPhase\s+\d+\b")),
    ("FULLY IMPLEMENTED marker", re.compile(r"FULLY IMPLEMENTED")),
    ("percentage claim", re.compile(r"\d+\s*%")),
    ("total lines claim", re.compile(r"\d+\s+total\s+lines")),
    ("rocket emoji", re.compile("\U0001f680")),
    ("white check mark emoji", re.compile("✅")),
    ("chart emoji", re.compile("\U0001f4ca")),
    ("test tube emoji", re.compile("\U0001f9ea")),
    ("wrench emoji", re.compile("\U0001f527")),
)


def _multi_line_imports(block: str) -> list[str]:
    """Return logical lines that begin with a relevant import statement.

    Handles multi-line parenthesised forms like ``from opifex.x import (\n
    A,\n    B,\n)`` by collapsing whitespace and removing parentheses /
    line continuations after detection.
    """
    raw_lines = block.splitlines()
    logical: list[str] = []
    index = 0
    while index < len(raw_lines):
        stripped = raw_lines[index].strip()
        if not stripped.startswith(("from opifex", "import opifex")):
            index += 1
            continue
        if "(" in stripped and ")" not in stripped:
            buffer = [stripped]
            index += 1
            while index < len(raw_lines):
                buffer.append(raw_lines[index].strip())
                if ")" in raw_lines[index]:
                    index += 1
                    break
                index += 1
            combined = " ".join(buffer).replace("(", " ").replace(")", " ")
            logical.append(" ".join(combined.split()))
        elif stripped.endswith("\\"):
            buffer = [stripped.rstrip("\\").strip()]
            index += 1
            while index < len(raw_lines):
                next_line = raw_lines[index].strip()
                if next_line.endswith("\\"):
                    buffer.append(next_line.rstrip("\\").strip())
                    index += 1
                else:
                    buffer.append(next_line)
                    index += 1
                    break
            logical.append(" ".join(buffer))
        else:
            logical.append(stripped.replace("(", " ").replace(")", " ").strip())
            index += 1
    return logical


def _parse_import_targets(line: str) -> tuple[str, tuple[str, ...]]:
    """Return (module_to_import, names_to_check) for a single import line.

    For ``from opifex.x.y import A, B`` returns ``("opifex.x.y", ("A","B"))``.
    For ``import opifex.x.y`` returns ``("opifex.x.y", ())``.
    """
    match = IMPORT_LINE_PATTERN.match(line)
    if match is None:
        raise ValueError(f"unrecognised import line: {line!r}")
    if match.group(1) is not None:
        module = match.group(1)
        raw_names = match.group(2)
        names = tuple(
            name.split(" as ")[0].strip() for name in raw_names.split(",") if name.strip()
        )
        return module, names
    return match.group(3), ()


def _iter_imports():
    """Yield (doc_path, line) tuples for every opifex import in fenced code."""
    for doc_path in DOC_FILES:
        text = doc_path.read_text(encoding="utf-8")
        for block_match in FENCED_BLOCK_PATTERN.finditer(text):
            block = block_match.group(1)
            for line in _multi_line_imports(block):
                yield doc_path, line


def test_doc_files_exist() -> None:
    """Each doc the test inspects must exist on disk."""
    missing = [str(path) for path in DOC_FILES if not path.is_file()]
    assert not missing, f"missing documentation files: {missing}"


@pytest.mark.parametrize(
    ("doc_path", "import_line"),
    [pytest.param(p, l, id=f"{p.name}:{l}") for p, l in _iter_imports()],
)
def test_doc_imports_resolve(doc_path: Path, import_line: str) -> None:
    """Each opifex import in a Python fence must resolve to a real symbol."""
    module_name, names = _parse_import_targets(import_line)
    module = importlib.import_module(module_name)
    for name in names:
        assert hasattr(module, name), (
            f"{doc_path.name}: '{import_line}' references "
            f"{module_name}.{name}, which does not exist"
        )


@pytest.mark.parametrize("doc_path", DOC_FILES, ids=lambda p: p.name)
def test_no_boilerplate_patterns(doc_path: Path) -> None:
    """No emoji / phase / percentage / AI-trace markers in the public docs."""
    text = doc_path.read_text(encoding="utf-8")
    offenders: list[str] = []
    for label, pattern in BOILERPLATE_PATTERNS:
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{label} at line {line_no}: {match.group(0)!r}")
    assert not offenders, f"{doc_path.name} contains forbidden boilerplate:\n" + "\n".join(
        offenders
    )

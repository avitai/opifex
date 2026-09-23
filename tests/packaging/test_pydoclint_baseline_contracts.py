"""Contracts for the pydoclint baseline ratchet.

The pre-commit hook regenerates the baseline from the files it scans, so an entry for a
file that has since been moved or deleted is never dropped: it lingers, describes code
that no longer exists, and hides nothing but still reads as tolerated debt.
"""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "quality" / "pydoclint_baseline.txt"


def _baselined_files() -> list[str]:
    return re.findall(r"^(src/\S+\.py)$", BASELINE.read_text(encoding="utf-8"), re.MULTILINE)


def test_the_baseline_lists_files() -> None:
    """Control: the parser finds entries, so an empty result below is not a broken parse."""
    assert len(_baselined_files()) > 100


def test_every_baselined_file_exists() -> None:
    missing = [path for path in _baselined_files() if not (REPO_ROOT / path).is_file()]
    assert not missing, f"baseline entries for files that no longer exist: {missing}"

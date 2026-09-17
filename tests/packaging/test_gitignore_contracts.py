"""No directory that holds tracked tests is gitignored.

``pre-commit run --all-files`` and ``git add`` see only tracked and trackable files, so an
ignore rule that covers a directory under ``tests/`` makes a new test file there invisible
to both: the hooks report a green run they never saw, and the file never reaches a commit.
The corpus is every directory that already holds a tracked file under ``tests/``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _tracked_test_directories() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "tests"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return sorted({str(Path(path).parent.as_posix()) for path in listed})


TEST_DIRECTORIES = _tracked_test_directories()


def test_the_corpus_holds_the_test_tree() -> None:
    assert "tests" in TEST_DIRECTORIES
    assert "tests/packaging" in TEST_DIRECTORIES


@pytest.mark.parametrize("directory", TEST_DIRECTORIES)
def test_a_new_file_in_every_tracked_test_directory_is_not_ignored(directory: str) -> None:
    probe = f"{directory}/test_probe_for_the_ignore_rules.py"
    completed = subprocess.run(
        ["git", "check-ignore", "-v", "--no-index", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # Exit 1 means "not ignored"; exit 0 prints the rule that ignores the path.
    assert completed.returncode == 1, f"{probe} is ignored by: {completed.stdout.strip()}"

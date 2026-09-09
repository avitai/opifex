"""Contracts for the ruff per-file baseline ratchet."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "check_ruff_baseline.py"
BASELINE = REPO_ROOT / "quality" / "ruff_baseline.json"
ADOPTED_FAMILIES = ("ANN", "D", "ARG", "B", "C901", "PL", "PTH", "RET", "TRY")

FIXTURE_PYPROJECT = """[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["ANN", "RET"]

[tool.ruff.lint.extend-per-file-ignores]
"tests/**" = ["ANN"]

[tool.ruff.lint.per-file-ignores]
"""
ONE_UNANNOTATED_ARGUMENT = "def f(x) -> int:\n    return x\n"
TWO_UNANNOTATED_ARGUMENTS = "def f(x, y) -> int:\n    return x + y\n"
CLEAN = "def f(x: int) -> int:\n    return x\n"


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _pyproject_table(root: Path) -> dict[str, list[str]]:
    with (root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["tool"]["ruff"]["lint"]["per-file-ignores"]


def _fixture_repo(root: Path) -> tuple[Path, Path]:
    _write(root / "pyproject.toml", FIXTURE_PYPROJECT)
    _write(root / "src" / "a.py", ONE_UNANNOTATED_ARGUMENT)
    _write(root / "src" / "b.py", CLEAN)
    _write(root / "tests" / "test_a.py", ONE_UNANNOTATED_ARGUMENT)
    baseline = root / "quality" / "ruff_baseline.json"
    common = ("--repo-root", str(root), "--baseline", str(baseline))
    write_result = _run_checker(*common, "--write-baseline")
    assert write_result.returncode == 0, write_result.stdout + write_result.stderr
    return baseline, root


def test_ruff_selects_the_adopted_families_with_the_shared_exemptions() -> None:
    """pyproject selects the shared rule set and carries the documented exemptions."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        lint = tomllib.load(handle)["tool"]["ruff"]["lint"]

    assert set(ADOPTED_FAMILIES) <= set(lint["select"])
    assert {
        "ANN401",
        "D100",
        "D104",
        "D105",
        "D107",
        "D202",
        "D212",
        "D401",
        "D402",
        "D415",
    } <= set(lint["ignore"])
    assert lint["pydocstyle"]["convention"] == "google"


def test_ruff_baseline_is_confined_to_src_and_rendered_into_pyproject() -> None:
    """The committed baseline is the source of the per-file-ignores table, src only."""
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    table = _pyproject_table(REPO_ROOT)

    assert payload["version"] == 1
    assert payload["scan_paths"] == ["src"]
    assert payload["entries"]
    assert all(path.startswith("src/") for path in payload["entries"])
    assert all(count > 0 for rules in payload["entries"].values() for count in rules.values())
    assert table == {path: sorted(rules) for path, rules in payload["entries"].items()}


def test_ruff_checker_blocks_growth_and_stale_entries(tmp_path: Path) -> None:
    """A baselined file may not gain findings, and a fixed pair must leave the baseline."""
    baseline, root = _fixture_repo(tmp_path)
    common = ("--repo-root", str(root), "--baseline", str(baseline))

    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["entries"] == {"src/a.py": {"ANN001": 1}}
    assert _pyproject_table(root) == {"src/a.py": ["ANN001"]}
    assert _run_checker(*common).returncode == 0

    _write(root / "src" / "a.py", TWO_UNANNOTATED_ARGUMENTS)
    grown = _run_checker(*common)
    assert grown.returncode == 1
    assert "src/a.py ANN001 1 -> 2" in grown.stdout

    _write(root / "src" / "a.py", CLEAN)
    stale = _run_checker(*common)
    assert stale.returncode == 1
    assert "stale" in grown.stdout + stale.stdout
    assert "src/a.py ANN001" in stale.stdout

    assert _run_checker(*common, "--write-baseline").returncode == 0
    assert json.loads(baseline.read_text(encoding="utf-8"))["entries"] == {}
    assert _pyproject_table(root) == {}
    assert _run_checker(*common).returncode == 0


def test_ruff_checker_leaves_unbaselined_files_to_ruff(tmp_path: Path) -> None:
    """Findings in a file without a baseline entry are ruff's failure, not the checker's."""
    baseline, root = _fixture_repo(tmp_path)
    common = ("--repo-root", str(root), "--baseline", str(baseline))

    _write(root / "src" / "c.py", ONE_UNANNOTATED_ARGUMENT)

    assert _run_checker(*common).returncode == 0
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src", "--output-format", "concise"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ruff.returncode == 1
    assert "src/c.py" in ruff.stdout
    assert "src/a.py" not in ruff.stdout


def test_ruff_checker_rejects_a_table_that_drifted_from_the_baseline(tmp_path: Path) -> None:
    """A hand edit to the rendered table fails until the baseline is regenerated."""
    baseline, root = _fixture_repo(tmp_path)
    common = ("--repo-root", str(root), "--baseline", str(baseline))

    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace('["ANN001"]', '["ANN001", "RET504"]'),
        encoding="utf-8",
    )

    drifted = _run_checker(*common)
    assert drifted.returncode == 1
    assert "per-file-ignores" in drifted.stdout


def test_pre_commit_runs_the_ruff_baseline_gate() -> None:
    """Pre-commit should run the checked-in ruff ratchet."""
    contents = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "id: ruff-baseline" in contents
    assert "uv run --no-sync python scripts/check_ruff_baseline.py" in contents

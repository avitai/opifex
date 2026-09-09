"""Check ruff findings in baselined files against a checked-in, shrinking baseline.

The baseline records, per file and rule, how many findings ruff reported when the
rule set was adopted. Ruff itself ignores those pairs through the per-file-ignores
table this script renders into ``pyproject.toml``; this script measures the tree
with that table cleared and fails when a pair grew, when a pair has no findings
left and still sits in the baseline, or when the rendered table drifted from the
baseline. Files without an entry are ruff's own responsibility.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_DIRS = ("src",)
BASELINE_VERSION = 1
TABLE_HEADER = "[tool.ruff.lint.per-file-ignores]"
TABLE_COMMENT = (
    "# Rendered by scripts/check_ruff_baseline.py from quality/ruff_baseline.json;"
    " regenerate with --write-baseline, never edit by hand."
)
CLEAR_TABLE = "lint.per-file-ignores = {}"

Entries = dict[str, dict[str, int]]


def _write_stdout(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _run_ruff(root: Path, scan_dirs: list[str]) -> list[dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        *scan_dirs,
        "--output-format",
        "json",
        "--exit-zero",
        "--config",
        CLEAR_TABLE,
    ]
    result = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)
    return json.loads(result.stdout)


def _collect_entries(root: Path, scan_dirs: list[str]) -> Entries:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for finding in _run_ruff(root, scan_dirs):
        path = Path(finding["filename"]).resolve().relative_to(root).as_posix()
        counts[path][finding["code"]] += 1
    return {path: dict(sorted(counts[path].items())) for path in sorted(counts)}


def _baseline_payload(entries: Entries, scan_dirs: list[str]) -> dict[str, Any]:
    return {"version": BASELINE_VERSION, "scan_paths": scan_dirs, "entries": entries}


def _load_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"ruff baseline does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != BASELINE_VERSION:
        raise SystemExit(f"unsupported ruff baseline version in {path}")
    return payload


def _render_table(entries: Entries) -> str:
    lines = [TABLE_HEADER, TABLE_COMMENT]
    for path, rules in entries.items():
        rendered_rules = ", ".join(f'"{rule}"' for rule in sorted(rules))
        lines.append(f'"{path}" = [{rendered_rules}]')
    return "\n".join(lines) + "\n"


def _replace_table(pyproject: Path, entries: Entries) -> None:
    text = pyproject.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{re.escape(TABLE_HEADER)}\n(?:(?!\[).*\n?)*", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        raise SystemExit(f"{pyproject} has no {TABLE_HEADER} table to render into")
    end = match.end()
    trailer = "" if end >= len(text) else "\n"
    pyproject.write_text(text[: match.start()] + _render_table(entries) + trailer + text[end:])


def _pyproject_table(pyproject: Path) -> dict[str, list[str]]:
    with pyproject.open("rb") as handle:
        return tomllib.load(handle)["tool"]["ruff"]["lint"].get("per-file-ignores", {})


def _write_baseline(
    root: Path, baseline_path: Path, entries: Entries, scan_dirs: list[str]
) -> None:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _baseline_payload(entries, scan_dirs)
    baseline_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _replace_table(root / "pyproject.toml", entries)
    pairs = sum(len(rules) for rules in entries.values())
    _write_stdout(f"wrote {pairs} ruff baseline pairs over {len(entries)} files to {baseline_path}")


def _compare_to_baseline(root: Path, current: Entries, baseline: Entries) -> int:
    grown: list[str] = []
    stale: list[str] = []
    for path, rules in baseline.items():
        for rule, allowed in rules.items():
            found = current.get(path, {}).get(rule, 0)
            if found > allowed:
                grown.append(f"- {path} {rule} {allowed} -> {found}")
            elif found == 0:
                stale.append(f"- {path} {rule}")

    expected_table = {path: sorted(rules) for path, rules in baseline.items()}
    table_drifted = _pyproject_table(root / "pyproject.toml") != expected_table

    if grown or stale or table_drifted:
        _write_stdout("ruff baseline check failed")
        if grown:
            _write_stdout("\nBaselined pairs that gained findings:")
            _write_stdout("\n".join(grown))
        if stale:
            _write_stdout("\nBaseline pairs now stale, no findings left (run --write-baseline):")
            _write_stdout("\n".join(stale))
        if table_drifted:
            _write_stdout(
                "\nThe per-file-ignores table in pyproject.toml does not match the baseline"
                " (run --write-baseline)."
            )
        return 1

    pairs = sum(len(rules) for rules in baseline.values())
    _write_stdout(f"ruff baseline check passed: {pairs} pairs over {len(baseline)} files")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline JSON file; defaults to quality/ruff_baseline.json under the root.",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        default=None,
        help="Directory to scan, relative to the root. Defaults to src/.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Record the current findings as the baseline and render the table.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the ruff baseline checker."""
    args = _parse_args()
    root = args.repo_root.resolve()
    baseline_path = (args.baseline or root / "quality" / "ruff_baseline.json").resolve()
    scan_dirs = list(args.source_dir or DEFAULT_SCAN_DIRS)
    current = _collect_entries(root, scan_dirs)

    if args.write_baseline:
        _write_baseline(root, baseline_path, current, scan_dirs)
        return 0

    baseline = _load_baseline(baseline_path)
    return _compare_to_baseline(root, current, baseline["entries"])


if __name__ == "__main__":
    raise SystemExit(main())

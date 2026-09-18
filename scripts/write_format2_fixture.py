#!/usr/bin/env python3
"""Write the format-2 module fixture in an environment that neither floats nor drifts.

The fixture must be written by the release that produced it (substrax 0.1.9), which the
project environment no longer holds, so the generator runs in an isolated
``uv run --no-project`` environment. Left to itself that environment resolves jax, jaxlib and
flax afresh on every run, and a release of one of them between two CI jobs changes what the
fixture step imports: jax 0.11.2 broke flax 0.12.9's import that way. This script reads those
three versions from ``uv.lock`` and pins them, so the isolated environment agrees with the
project's on the numerical stack and moves only when the lock does.

Usage::

    python3 scripts/write_format2_fixture.py tests/core/training/fixtures/format2
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


PRODUCERS = ("substrax==0.1.9",)
LOCKED = ("jax", "jaxlib", "flax")
GENERATOR = Path("scripts") / "make_format2_module_fixture.py"
REPO_ROOT = Path(__file__).resolve().parents[1]


def locked_versions(lock: Path) -> dict[str, str]:
    """The versions ``lock`` holds for the packages the fixture environment must share.

    Args:
        lock: The project's ``uv.lock``.

    Returns:
        Package name to version, in :data:`LOCKED` order.

    Raises:
        SystemExit: If the lock holds no version for one of them.
    """
    packages = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
    versions = {package["name"]: package["version"] for package in packages}
    missing = [name for name in LOCKED if name not in versions]
    if missing:
        raise SystemExit(f"{lock} holds no version for {', '.join(missing)}")
    return {name: versions[name] for name in LOCKED}


def command(destination: str) -> list[str]:
    """The isolated generator invocation for ``destination``."""
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is not on PATH")
    pins = [
        *PRODUCERS,
        *(f"{name}=={v}" for name, v in locked_versions(REPO_ROOT / "uv.lock").items()),
    ]
    with_arguments = [argument for pin in pins for argument in ("--with", pin)]
    return [uv, "run", "--no-project", *with_arguments, "python", str(GENERATOR), destination]


def main(argv: list[str]) -> int:
    """Run the generator in the pinned environment; the exit status is the generator's."""
    if len(argv) != 2:
        sys.stderr.write("usage: write_format2_fixture.py DESTINATION\n")
        return 2
    invocation = command(argv[1])
    sys.stdout.write(" ".join(invocation) + "\n")
    return subprocess.run(invocation, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))

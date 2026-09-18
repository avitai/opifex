#!/usr/bin/env python
"""Launch an opifex training run on a rented vast.ai GPU via SkyPilot.

A thin, opifex-flavoured wrapper around the SkyPilot lifecycle (modelled on
datarax's ``benchmarks/automation`` orchestrator, but minimal): it provisions a
vast.ai instance from :mod:`deploy/vast/qh9_train.sky.yaml`, rsyncs the working
tree (respecting ``.skyignore``) and the QH9 database, runs the training driver,
streams its logs, then fetches the run directory back and (optionally) tears the
instance down.

SkyPilot does the heavy lifting (vast provisioning, SSH, rsync); this wrapper just
wires the cluster name, hyper-parameter env overrides, result download and
teardown into one command following opifex's ``argparse`` -> frozen-dataclass
convention.

Prerequisites (one-time):
    uv tool install "skypilot[vast]"      # or: pipx install "skypilot[vast]"
    sky check vast                         # wires ~/.config/vastai/vast_api_key

Example:
    python deploy/vast/launch.py --cluster opifex-qh9 \
        --env QH9_BATCH=16 --env QH9_EPOCHS=30 --download-dir runs/qh9_mul128

The local results land under ``--download-dir`` (default ``runs/<cluster>``) with
the trainer's ``train.log`` / ``metrics.json`` / ``checkpoints/`` laid out as the
local driver writes them.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


logger = logging.getLogger("launch_vast")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASK_YAML = _REPO_ROOT / "deploy" / "vast" / "qh9_train.sky.yaml"
_REMOTE_OUT = "/root/results/qh9_mul128"


@dataclass(frozen=True, slots=True)
class LaunchArgs:
    """Parsed command-line arguments for the vast.ai launch wrapper."""

    cluster: str
    sky: str
    env_overrides: tuple[str, ...]
    download_dir: Path
    remote_out: str
    teardown: bool
    detach: bool
    extra_sky_args: tuple[str, ...] = field(default_factory=tuple)


def _parse_args(argv: list[str] | None) -> LaunchArgs:
    """Parse command-line arguments into a :class:`LaunchArgs`."""
    parser = argparse.ArgumentParser(description="Launch opifex QH9 training on vast.ai.")
    parser.add_argument("--cluster", default="opifex-qh9", help="SkyPilot cluster name.")
    parser.add_argument(
        "--sky",
        default=shutil.which("sky") or "sky",
        help="Path to the sky executable (default: first on PATH).",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        dest="env_overrides",
        metavar="KEY=VALUE",
        help="Override a task env (repeatable), e.g. --env QH9_BATCH=24.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Local dir to fetch the remote run into (default: runs/<cluster>).",
    )
    parser.add_argument("--remote-out", default=_REMOTE_OUT, help="Remote run directory to fetch.")
    parser.add_argument(
        "--teardown",
        action="store_true",
        help="Run `sky down <cluster>` after fetching results (otherwise leave it up).",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Launch detached (`sky launch -d`) and exit without fetching results.",
    )
    namespace = parser.parse_args(argv)
    download_dir = namespace.download_dir or (_REPO_ROOT / "runs" / namespace.cluster)
    return LaunchArgs(
        cluster=namespace.cluster,
        sky=namespace.sky,
        env_overrides=tuple(namespace.env_overrides),
        download_dir=download_dir,
        remote_out=namespace.remote_out,
        teardown=namespace.teardown,
        detach=namespace.detach,
    )


def _run(command: list[str]) -> None:
    """Run a subprocess, streaming output, raising on a non-zero exit."""
    logger.info("$ %s", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"command failed ({completed.returncode}): {' '.join(command)}")


def _launch_command(args: LaunchArgs) -> list[str]:
    """Build the `sky launch` command for the QH9 task."""
    command = [args.sky, "launch", "--infra", "vast", "-c", args.cluster, "-y", str(_TASK_YAML)]
    if args.detach:
        command.insert(2, "-d")
    for override in args.env_overrides:
        command.extend(["--env", override])
    return command


def launch(args: LaunchArgs) -> None:
    """Provision, train, fetch results and optionally tear down."""
    _run(_launch_command(args))
    if args.detach:
        logger.info(
            "launched detached on cluster %r; stream with `%s logs %s` and fetch later.",
            args.cluster,
            args.sky,
            args.cluster,
        )
        return
    args.download_dir.mkdir(parents=True, exist_ok=True)
    logger.info("fetching %s -> %s", args.remote_out, args.download_dir)
    _run([args.sky, "rsync", f"{args.cluster}:{args.remote_out}/", f"{args.download_dir}/"])
    if args.teardown:
        _run([args.sky, "down", "-y", args.cluster])
    logger.info("done; results in %s", args.download_dir)


def main(argv: list[str] | None = None) -> None:
    """Entry point: configure logging and run the launch lifecycle."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    launch(_parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])

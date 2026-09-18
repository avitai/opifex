#!/usr/bin/env python
"""Launch an opifex training run on a vast.ai GPU via the vast CLI directly.

SkyPilot's vast integration is unreliable here (its ``vastai-sdk`` 1.0+ breaks
offer filtering, and its catalog lacks the RTX PRO 6000 server variant), so this
launcher drives the ``vastai`` CLI directly for **deterministic** hardware: it
picks a specific single-GPU offer, provisions it (injecting the SSH key in the
onstart so connection works on a plain CUDA image), rsyncs the working tree and
the QH9 database, builds opifex's canonical isolated env with the repo's own
``setup.sh`` (whose ``gpu`` extra ships the >=12.8 CUDA wheels Blackwell sm_120
needs), and starts the training driver from the activated venv in a ``tmux``
session that survives disconnects.

Prerequisites:
    - ``~/.config/vastai/vast_api_key`` (the vast CLI key).
    - A local SSH key whose public half is passed via ``--ssh-key`` (the launcher
      registers it with vast and injects it into the instance).
    - ``uvx`` available (used to run the ``vastai`` CLI without a global install).

Example:
    python deploy/vast/launch_direct.py \
        --gpu RTX_PRO_6000_S --disk 256 \
        --db /mnt/ssd2/Data/qh9/raw/QH9Stable.db \
        --ssh-key ~/.ssh/hetzner_avitai \
        -- --hidden "128x0e + 128x1o + 128x2e + 128x3o + 128x4e" \
           --sh-lmax 4 --num-interactions 5 --start-refinement-layer 2 \
           --bottleneck-mul 32 --batch-size 16 --epochs 30

Everything after ``--`` is forwarded verbatim to ``scripts/train_qh9_blocks.py``
on the instance (the ``--db`` and ``--out`` are supplied automatically). The run's
``train.log`` / ``metrics.json`` / ``checkpoints/`` live under ``--remote-out`` on
the instance; fetch them with ``--fetch`` (or the printed rsync command).
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger("launch_vast_direct")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REMOTE_REPO = "/root/opifex"
_REMOTE_DB = "/root/qh9/QH9Stable.db"
_REMOTE_OUT = "/root/results/qh9_run"
# Synced code: source, scripts and packaging only (mirrors .skyignore intent).
# Cache / build / data dirs are excluded so a checkout's local ``.cache`` /
# ``.uv-cache`` (often many GB) is never uploaded -- that ballooned a sync to
# >5 GB and stalled provisioning before this exclusion was added.
_RSYNC_EXCLUDES = (
    ".venv",
    ".git",
    ".cache",
    ".uv-cache",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "cache",
    "checkpoints",
    "results",
    "mlruns",
    "example_data",
    "examples",
    "tests",
    "docs",
    "documents",
    "site",
    "htmlcov",
    "memory-bank",
    "internal_docs",
    "plans",
    "temp",
    "wandb",
    "examples_output",
    "benchmark_results",
    "__pycache__",
    "*.db",
)


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    """Parsed launch configuration."""

    gpu: str
    offer_id: int | None
    disk: int
    image: str
    ssh_key: Path
    db: Path | None
    db_url: str | None
    remote_out: str
    batch_size: int
    epochs: int
    train_args: tuple[str, ...]
    fetch: Path | None
    dry_run: bool
    entrypoint: str | None = None
    setup_extras: tuple[str, ...] = ("neural-dft",)
    env_vars: tuple[str, ...] = ()
    ssh_timeout_minutes: float = 25.0
    keep_on_failure: bool = False
    onstart_extra: tuple[str, ...] = field(default_factory=tuple)

    @property
    def syncs_examples(self) -> bool:
        """Whether the run needs the (normally excluded) ``examples/`` tree synced.

        A custom ``--entrypoint`` (e.g. running an example script) requires the
        ``examples/`` and ``tests/`` trees the QH9 trainer path deliberately omits.
        """
        return self.entrypoint is not None


def _vastai(args: list[str], *, capture: bool = False) -> str:
    """Run a ``vastai`` CLI command (via uvx) and return stdout if captured."""
    command = ["uvx", "vastai", *args]
    logger.info("$ %s", " ".join(command))
    if capture:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout
    subprocess.run(command, check=True)
    return ""


def _select_offer(config: LaunchConfig) -> int:
    """Return an explicit offer id, or the cheapest matching single-GPU offer."""
    if config.offer_id is not None:
        return config.offer_id
    query = f"gpu_name={config.gpu} num_gpus=1 rentable=true disk_space>={config.disk}"
    raw = _vastai(["search", "offers", query, "-o", "dph+", "--raw"], capture=True)
    offers = json.loads(raw)
    if not offers:
        raise SystemExit(f"no rentable single-GPU offers for {config.gpu!r}")
    offer = offers[0]
    logger.info(
        "selected offer %s: %sx %s $%.2f/hr disk=%dGB",
        offer["id"],
        offer["num_gpus"],
        offer["gpu_name"],
        offer["dph_total"],
        round(offer["disk_space"]),
    )
    return int(offer["id"])


def _onstart_script(config: LaunchConfig) -> str:
    """Build the onstart that installs sshd and injects the SSH public key.

    A plain CUDA image has no sshd and (because a custom onstart overrides vast's
    default key injection) no authorized key, so both are set up explicitly here.
    """
    public_key = config.ssh_key.with_suffix(".pub").read_text().strip()
    lines = [
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update && apt-get install -y openssh-server git rsync curl tmux ca-certificates",
        "mkdir -p /run/sshd /root/.ssh",
        f"echo {shlex.quote(public_key)} >> /root/.ssh/authorized_keys",
        "chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys",
        "(service ssh start || /usr/sbin/sshd)",
        *config.onstart_extra,
    ]
    return "; ".join(lines)


def _create_instance(config: LaunchConfig, offer_id: int) -> int:
    """Register the SSH key, create the instance and return its id."""
    public_key = config.ssh_key.with_suffix(".pub").read_text().strip()
    # Registering an already-present key is a harmless no-op error; ignore it.
    try:
        _vastai(["create", "ssh-key", public_key])
    except subprocess.CalledProcessError:
        logger.info("ssh key already registered with vast (continuing)")
    raw = _vastai(
        [
            "create",
            "instance",
            str(offer_id),
            "--image",
            config.image,
            "--disk",
            str(config.disk),
            "--ssh",
            "--direct",
            "--label",
            "opifex-train",
            "--onstart-cmd",
            _onstart_script(config),
            "--raw",
        ],
        capture=True,
    )
    contract = json.loads(raw)
    instance_id = int(contract["new_contract"])
    logger.info("created instance %d", instance_id)
    return instance_id


def _instance_state(instance_id: int) -> dict[str, Any]:
    """Return the raw vast instance record (``actual_status``, ``ssh_host`` ...)."""
    raw = _vastai(["show", "instance", str(instance_id), "--raw"], capture=True)
    return json.loads(raw)


def _instance_ssh(info: dict[str, Any]) -> tuple[str, int] | None:
    """Return ``(host, port)`` once the instance exposes an SSH endpoint.

    The endpoint is offered whenever ``ssh_host`` is populated -- which happens
    while ``actual_status`` is still ``"loading"`` (the image is pulling but the
    proxy sshd is already reachable). Gating on ``actual_status == "running"``
    would skip that window entirely and is why a slow-loading image used to time
    the launcher out; instead the caller probes SSH on every poll where a host is
    present and only the probe decides readiness.
    """
    host = info.get("ssh_host")
    port = info.get("ssh_port")
    if not host or not port:
        return None
    return str(host), int(port)


def _destroy_instance(instance_id: int) -> None:
    """Best-effort destroy so a failed provision never bills idle."""
    try:
        _vastai(["destroy", "instance", str(instance_id)])
        logger.info("destroyed instance %d", instance_id)
    except subprocess.CalledProcessError:
        logger.exception(
            "FAILED to auto-destroy instance %d -- it may still be billing; "
            "run: uvx vastai destroy instance %d",
            instance_id,
            instance_id,
        )


def _ssh_base(config: LaunchConfig, host: str, port: int) -> list[str]:
    """Return the ssh command prefix for the instance."""
    return [
        "ssh",
        "-i",
        str(config.ssh_key),
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=12",
        f"root@{host}",
    ]


# vast ``actual_status`` values that mean the instance can never become usable;
# seeing one aborts the wait immediately instead of burning the whole timeout.
_TERMINAL_STATUSES = frozenset({"exited", "error", "offline"})


def _wait_for_ssh(
    config: LaunchConfig, instance_id: int, *, timeout_minutes: float = 25.0
) -> tuple[str, int]:
    """Poll until the instance accepts SSH, returning ``(host, port)``.

    Robust against slow-loading images: the instance's ``actual_status`` is logged
    every poll (so a long ``"loading"`` is visible, not silent), SSH is probed as
    soon as an endpoint appears (even while still loading), the wait runs to a
    wall-clock ``timeout_minutes`` rather than a fixed attempt count, and a
    terminal status (``error`` / ``exited`` / ``offline``) aborts immediately. On
    timeout it raises so the caller can tear the instance down -- it must never be
    left billing in ``loading``.

    Args:
        config: The launch configuration (for the SSH probe).
        instance_id: The vast instance id to wait on.
        timeout_minutes: Wall-clock budget before giving up.

    Returns:
        The reachable ``(host, port)``.

    Raises:
        SystemExit: If the instance reaches a terminal status or the timeout
            elapses without SSH becoming reachable.
    """
    deadline = time.monotonic() + timeout_minutes * 60.0
    poll = 0
    last_status = ""
    while time.monotonic() < deadline:
        poll += 1
        info = _instance_state(instance_id)
        status = str(info.get("actual_status") or "unknown")
        if status != last_status:
            logger.info("instance %d status: %s", instance_id, status)
            last_status = status
        if status in _TERMINAL_STATUSES:
            raise SystemExit(f"instance {instance_id} reached terminal status {status!r}")
        endpoint = _instance_ssh(info)
        if endpoint is not None:
            host, port = endpoint
            probe = [*_ssh_base(config, host, port), "-o", "BatchMode=yes", "echo ok"]
            result = subprocess.run(probe, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.info("ssh ready at %s:%d (status=%s, poll %d)", host, port, status, poll)
                return host, port
        remaining = int(deadline - time.monotonic())
        logger.info("waiting for ssh (status=%s, ~%ds left)...", status, max(remaining, 0))
        time.sleep(20)
    raise SystemExit(
        f"instance {instance_id} did not become SSH-reachable within "
        f"{timeout_minutes:.0f} min (last status: {last_status!r})"
    )


def _repo_excludes(config: LaunchConfig) -> tuple[str, ...]:
    """Return the rsync excludes for the repo upload.

    Drops ``examples``/``tests`` from the default excludes when a custom
    entrypoint needs them (e.g. running an example), keeping the big cache/data
    dirs excluded so the upload stays small.
    """
    if not config.syncs_examples:
        return _RSYNC_EXCLUDES
    return tuple(pattern for pattern in _RSYNC_EXCLUDES if pattern not in {"examples", "tests"})


def _rsync(
    config: LaunchConfig,
    host: str,
    port: int,
    source: str,
    dest: str,
    *,
    exclude_patterns: tuple[str, ...] = (),
) -> None:
    """Rsync ``source`` to ``root@host:dest`` over the instance's SSH.

    Compression (``-z``) is always on: the upload is bound by the local upstream
    link (the instance downlink is ~9 Gbit/s), so trading CPU for bytes-on-the-wire
    is a net win even for the weakly-compressible (~1.18x) QH9 float DB. For the
    one-off, much faster path that avoids re-uploading the DB at all, stage it once
    and pass ``--db-url`` (see :func:`_download_remote_db`).
    """
    ssh = f"ssh -i {config.ssh_key} -p {port} -o StrictHostKeyChecking=no"
    command = ["rsync", "-az", "--info=stats1", "-e", ssh]
    for pattern in exclude_patterns:
        command.extend(["--exclude", pattern])
    command.extend([source, f"root@{host}:{dest}"])
    logger.info("$ %s", " ".join(command))
    subprocess.run(command, check=True)


def _remote(config: LaunchConfig, host: str, port: int, script: str) -> None:
    """Run a bash script on the instance over SSH."""
    command = [*_ssh_base(config, host, port), "bash -s"]
    logger.info("remote: %s", script.strip().splitlines()[0])
    subprocess.run(command, input=script, text=True, check=True)


def _download_remote_db(config: LaunchConfig, host: str, port: int) -> None:
    """Pull the QH9 DB onto the instance from ``config.db_url`` (the fast path).

    Local upstream is the rsync bottleneck (a single TCP stream already saturates
    it; parallel streams were measured *slower*). Downloading from a URL the
    instance can reach uses its ~9 Gbit/s downlink instead, so a DB staged once on
    fast-upstream storage (e.g. a datacenter box or object store) transfers in
    seconds rather than tens of minutes. ``curl`` retries/resumes for robustness.
    """
    url = config.db_url
    if url is None:
        raise ValueError("_download_remote_db called without db_url")
    download = (
        f"set -e; mkdir -p {Path(_REMOTE_DB).parent}; "
        f"curl -fSL --retry 5 --retry-delay 5 -C - -o {_REMOTE_DB} {shlex.quote(url)}; "
        f"ls -la {_REMOTE_DB}"
    )
    logger.info("downloading DB on instance from %s", url)
    _remote(config, host, port, download)


def _setup_and_launch(config: LaunchConfig, host: str, port: int) -> None:
    """Build opifex's canonical isolated env via setup.sh and start training.

    Uses the repo's own ``setup.sh --backend cuda12`` (which builds the venv with
    the isolated CUDA toolchain -- the gpu extra's >=12.8 wheels give Blackwell
    sm_120 codegen) and the generated ``activate.sh``, so the remote environment
    matches a local checkout exactly rather than an ad-hoc venv.
    """
    extra_flags = " ".join(f"--extra {shlex.quote(extra)}" for extra in config.setup_extras)
    setup = (
        'set -e; export PATH="$HOME/.local/bin:$PATH"; '
        "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; "
        # Extras are configurable: the QH9 trainer needs ``neural-dft`` (pyscf, which
        # qh9_eval imports transitively); an example may need none or different ones.
        f"cd {_REMOTE_REPO}; chmod +x setup.sh; "
        f"./setup.sh --backend cuda12 --python 3.12 {extra_flags}; "
        'source activate.sh; python -c "import jax; print(jax.devices())"'
    )
    _remote(config, host, port, setup)
    forwarded = " ".join(shlex.quote(a) for a in config.train_args)
    # The run command: a custom ``--entrypoint`` (e.g. an example script) verbatim,
    # otherwise the QH9 block trainer with its DB/out/batch/epoch flags.
    if config.entrypoint is not None:
        run_command = f"exec {config.entrypoint}"
    else:
        run_command = (
            "exec uv run python scripts/train_qh9_blocks.py --dataset stable "
            f"--db {_REMOTE_DB} --batch-size {config.batch_size} "
            f"--epochs {config.epochs} --out {config.remote_out} {forwarded}"
        )
    env_export = f"export {' '.join(config.env_vars)}" if config.env_vars else ":"
    # Write the run command to a script FILE and run that in tmux, rather than
    # embedding it (with shlex-quoted args) inside a nested ``tmux '...'`` string --
    # the nested single quotes broke the invocation, and a bare ``python`` did not
    # resolve in the non-interactive tmux shell. The script exports the uv path,
    # activates the env and uses ``uv run python`` (which resolves the project venv
    # regardless of PATH), so the run starts reliably.
    run_script = "\n".join(
        [
            "#!/bin/bash",
            "set -e",
            f"cd {_REMOTE_REPO}",
            'export PATH="$HOME/.local/bin:$PATH"',
            "source activate.sh",
            env_export,
            run_command,
        ]
    )
    write_script = (
        f"mkdir -p {config.remote_out} && cat > {_REMOTE_REPO}/run_train.sh "
        f"<<'OPIFEX_TRAIN_EOF'\n{run_script}\nOPIFEX_TRAIN_EOF"
    )
    _remote(config, host, port, write_script)
    launch_train = (
        f"tmux new-session -d -s train "
        f"'bash {_REMOTE_REPO}/run_train.sh > {config.remote_out}/console.log 2>&1'"
    )
    _remote(config, host, port, launch_train)
    logger.info("training started in tmux session 'train' on %s:%d", host, port)


def launch(config: LaunchConfig) -> None:
    """Run the full provision -> sync -> setup -> train lifecycle.

    Everything after the instance is created runs under a guard that destroys the
    instance on any failure (unless ``--keep-on-failure``): a provision that
    stalls in ``loading`` or a sync/setup error must never leave a GPU billing
    idle. On success the instance is intentionally left running (it is doing the
    training); the teardown command is printed.
    """
    offer_id = _select_offer(config)
    if config.dry_run:
        logger.info("dry-run: would provision offer %d and launch; stopping.", offer_id)
        return
    instance_id = _create_instance(config, offer_id)
    try:
        host, port = _wait_for_ssh(config, instance_id, timeout_minutes=config.ssh_timeout_minutes)
        _rsync(
            config,
            host,
            port,
            f"{_REPO_ROOT}/",
            f"{_REMOTE_REPO}/",
            exclude_patterns=_repo_excludes(config),
        )
        # DB transfer: prefer the fast instance-side download when a URL is staged;
        # otherwise rsync the local copy (upstream-bound -- see _download_remote_db).
        if config.db_url is not None:
            _download_remote_db(config, host, port)
        elif config.db is not None:
            _remote(config, host, port, f"mkdir -p {Path(_REMOTE_DB).parent}")
            _rsync(config, host, port, str(config.db), _REMOTE_DB)
        _setup_and_launch(config, host, port)
    except BaseException as error:
        # Any failure -- including a stalled ``loading`` timeout or Ctrl-C -- must
        # tear the instance down so it never bills idle. The traceback surfaces
        # once at the top level via ``raise``, so log only a concise reason here
        # (TRY400's ``logging.exception`` would duplicate that traceback).
        if config.keep_on_failure:
            logger.error(  # noqa: TRY400 - concise reason; traceback re-raised below
                "launch failed (%s); instance %d KEPT (--keep-on-failure) and is "
                "billing -- teardown: uvx vastai destroy instance %d",
                error,
                instance_id,
                instance_id,
            )
        else:
            logger.error(  # noqa: TRY400 - concise reason; traceback re-raised below
                "launch failed (%s); destroying instance %d to stop billing",
                error,
                instance_id,
            )
            _destroy_instance(instance_id)
        raise
    ssh_cmd = " ".join(_ssh_base(config, host, port))
    logger.info("monitor:  %s 'tail -f %s/train.log'", ssh_cmd, config.remote_out)
    logger.info("teardown: uvx vastai destroy instance %d", instance_id)
    if config.fetch is not None:
        config.fetch.mkdir(parents=True, exist_ok=True)
        logger.info(
            "fetch later: rsync -a -e 'ssh -i %s -p %d' root@%s:%s/ %s/",
            config.ssh_key,
            port,
            host,
            config.remote_out,
            config.fetch,
        )


def _parse_args(argv: list[str] | None) -> LaunchConfig:
    """Parse command-line arguments into a :class:`LaunchConfig`."""
    parser = argparse.ArgumentParser(description="Launch opifex training on vast.ai (direct CLI).")
    parser.add_argument("--gpu", default="RTX_PRO_6000_S", help="vast gpu_name to search for.")
    parser.add_argument("--offer-id", type=int, default=None, help="Explicit offer id.")
    parser.add_argument("--disk", type=int, default=256, help="Instance disk size (GB).")
    parser.add_argument(
        "--image",
        default="nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        help="Docker image (CUDA version is irrelevant; jax[cuda12] is bundled).",
    )
    parser.add_argument("--ssh-key", type=Path, required=True, help="Local private SSH key path.")
    parser.add_argument("--db", type=Path, default=None, help="Local QH9 database to rsync.")
    parser.add_argument(
        "--db-url",
        default=None,
        help="URL to download the QH9 DB onto the instance (fast: uses the instance's "
        "~9 Gbit/s downlink instead of the upstream-bound local rsync). Overrides --db.",
    )
    parser.add_argument("--remote-out", default=_REMOTE_OUT, help="Remote run directory.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--entrypoint",
        default=None,
        help="Custom remote run command (run in the activated venv via tmux), e.g. "
        "'uv run python examples/atomistic/nequip_md17.py'. Overrides the default QH9 "
        "trainer and syncs the examples/ + tests/ trees; --db is then optional.",
    )
    parser.add_argument(
        "--setup-extra",
        action="append",
        default=None,
        help="uv extra passed to setup.sh (repeatable). Default: neural-dft (QH9).",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=None,
        dest="env_vars",
        help="KEY=VALUE env var exported before the run (repeatable). Default: "
        "JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false.",
    )
    parser.add_argument("--fetch", type=Path, default=None, help="Local dir to fetch results into.")
    parser.add_argument("--dry-run", action="store_true", help="Select an offer and stop.")
    parser.add_argument(
        "--ssh-timeout-minutes",
        type=float,
        default=25.0,
        help="Wall-clock budget to wait for the instance to become SSH-reachable.",
    )
    parser.add_argument(
        "--keep-on-failure",
        action="store_true",
        help="Do NOT auto-destroy the instance if provisioning/setup fails (default: destroy "
        "so a stalled instance never bills idle).",
    )
    parser.add_argument(
        "train_args", nargs="*", help="Args forwarded to train_qh9_blocks.py (after --)."
    )
    namespace = parser.parse_args(argv)
    return LaunchConfig(
        gpu=namespace.gpu,
        offer_id=namespace.offer_id,
        disk=namespace.disk,
        image=namespace.image,
        ssh_key=namespace.ssh_key.expanduser(),
        db=namespace.db,
        db_url=namespace.db_url,
        remote_out=namespace.remote_out,
        batch_size=namespace.batch_size,
        epochs=namespace.epochs,
        train_args=tuple(namespace.train_args),
        fetch=namespace.fetch,
        dry_run=namespace.dry_run,
        entrypoint=namespace.entrypoint,
        setup_extras=(
            tuple(namespace.setup_extra) if namespace.setup_extra is not None else ("neural-dft",)
        ),
        env_vars=(
            tuple(namespace.env_vars)
            if namespace.env_vars is not None
            else ("JAX_ENABLE_X64=1", "XLA_PYTHON_CLIENT_PREALLOCATE=false")
        ),
        ssh_timeout_minutes=namespace.ssh_timeout_minutes,
        keep_on_failure=namespace.keep_on_failure,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point: configure logging and run the launch."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    launch(_parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python
"""GPU diagnostics and JAX environment helpers for Opifex scripts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any


def get_optimal_jax_env_vars() -> dict[str, str]:
    """Return conservative JAX environment defaults for local GPU development."""
    return {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
        "TF_CPP_MIN_LOG_LEVEL": "1",
        "JAX_ENABLE_X64": "0",
        "JAX_CUDA_PLUGIN_VERIFY": "false",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
    }


def configure_jax_env_vars(*, override: bool = False) -> dict[str, str]:
    """Apply default JAX environment variables when they are not already set."""
    configured: dict[str, str] = {}
    for key, value in get_optimal_jax_env_vars().items():
        if override or key not in os.environ:
            os.environ[key] = value
        configured[key] = os.environ[key]
    return configured


def _run_nvidia_smi(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run nvidia-smi and return the result, or None when unavailable."""
    try:
        return subprocess.run(
            ["nvidia-smi", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def get_gpu_memory_info() -> dict[str, Any]:
    """Return first-GPU memory information in MB when nvidia-smi is available."""
    info: dict[str, Any] = {
        "total": None,
        "used": None,
        "free": None,
        "unit": "MB",
        "source": "unavailable",
    }

    result = _run_nvidia_smi(
        [
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None:
        info["error"] = "nvidia-smi not found"
        return info
    if result.returncode != 0:
        info["error"] = result.stderr.strip() or "nvidia-smi failed"
        return info

    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    try:
        total, used, free = (int(part.strip()) for part in first_line.split(",")[:3])
    except (TypeError, ValueError):
        info["error"] = f"could not parse nvidia-smi output: {first_line!r}"
        return info

    info.update({"total": total, "used": used, "free": free, "source": "nvidia-smi"})
    return info


def _print_jax_info() -> None:
    """Print JAX backend/device details if JAX can be imported."""
    try:
        import jax
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"JAX import failed: {exc}")
        return

    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print("JAX devices:")
    for device in jax.devices():
        print(f"  - {device}")


def print_comprehensive_gpu_info() -> None:
    """Print GPU, JAX, and relevant environment diagnostics."""
    print("\n=== GPU Diagnostics ===")

    list_result = _run_nvidia_smi(["-L"])
    if list_result is None:
        print("nvidia-smi: not found")
    elif list_result.returncode == 0 and list_result.stdout.strip():
        print("nvidia-smi devices:")
        for line in list_result.stdout.strip().splitlines():
            print(f"  {line}")
    else:
        message = list_result.stderr.strip() or "no devices reported"
        print(f"nvidia-smi: {message}")

    print(f"GPU memory: {get_gpu_memory_info()}")

    print("\n=== JAX Diagnostics ===")
    _print_jax_info()

    print("\n=== Relevant Environment ===")
    for key in get_optimal_jax_env_vars():
        print(f"{key}: {os.environ.get(key, 'Not set')}")
    print(f"JAX_PLATFORMS: {os.environ.get('JAX_PLATFORMS', 'Not set')}")
    print(f"OPIFEX_BACKEND: {os.environ.get('OPIFEX_BACKEND', 'Not set')}")
    print(f"OPIFEX_ENV_ROOT: {os.environ.get('OPIFEX_ENV_ROOT', 'Not set')}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print comprehensive diagnostics instead of only environment defaults",
    )
    parser.add_argument(
        "--configure",
        action="store_true",
        help="Apply default JAX environment variables before printing diagnostics",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the GPU utility CLI."""
    args = parse_args(argv)
    if args.configure:
        configure_jax_env_vars()

    if args.full:
        print_comprehensive_gpu_info()
    else:
        for key, value in get_optimal_jax_env_vars().items():
            print(f"{key}={os.environ.get(key, value)}")
        print(f"GPU memory: {get_gpu_memory_info()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

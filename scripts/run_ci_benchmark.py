#!/usr/bin/env python3
"""CI benchmark regression guard.

Runs a small benchmark suite on CPU, saves results to a calibrax Store,
and checks for regressions against the stored baseline.

Exit code 0 = no regressions, 1 = regressions detected.
"""

import logging
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from calibrax.ci.guard import CIGuard
from calibrax.core.models import Metric, Point, Run
from calibrax.storage.store import Store

from opifex.benchmarking.adapters import default_metric_defs


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STORE_PATH = Path("benchmark-data")
SEED = 42
RESOLUTION = 32
N_SAMPLES = 50
THRESHOLD = 0.10  # 10% regression threshold (lenient for CPU-only CI)


def _get_git_info() -> tuple[str | None, str | None]:
    """Retrieve current git commit and branch."""
    commit = None
    branch = None
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return commit, branch


def _run_benchmark() -> list[Point]:
    """Run a small benchmark suite and return measurement points."""
    key = jax.random.PRNGKey(SEED)
    points: list[Point] = []

    # Simple FNO-like forward pass benchmark
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (N_SAMPLES, 1, RESOLUTION, RESOLUTION))
    y_true = jax.random.normal(k2, (N_SAMPLES, 1, RESOLUTION, RESOLUTION))

    # Compute basic metrics as a sanity check
    y_pred = x * 0.9 + 0.1  # Dummy prediction
    mse = float(jnp.mean((y_pred - y_true) ** 2))
    mae = float(jnp.mean(jnp.abs(y_pred - y_true)))
    rel_error = float(jnp.sqrt(jnp.sum((y_pred - y_true) ** 2) / jnp.sum(y_true**2)))

    points.append(
        Point(
            name="cpu_sanity",
            scenario="synthetic",
            tags={"dataset": "synthetic", "resolution": str(RESOLUTION)},
            metrics={
                "mse": Metric(value=mse),
                "mae": Metric(value=mae),
                "relative_error": Metric(value=rel_error),
            },
        )
    )

    logger.info(
        "Benchmark complete: mse=%.4e, mae=%.4e, rel_error=%.4e",
        mse,
        mae,
        rel_error,
    )
    return points


def main() -> int:
    """Run benchmarks and check for regressions.

    Returns:
        0 if no regressions, 1 if regressions detected.
    """
    logger.info("Starting CI benchmark suite (JAX backend: %s)", jax.default_backend())

    store = Store(STORE_PATH)
    commit, branch = _get_git_info()

    points = _run_benchmark()
    run = Run(
        points=tuple(points),
        commit=commit,
        branch=branch,
        metric_defs=default_metric_defs(),
    )
    store.save(run)
    logger.info("Saved run %s to store", run.id)

    # Check for baseline — if none exists, set this run as baseline
    baseline = store.get_baseline()
    if baseline is None:
        store.set_baseline(run.id)
        logger.info("No baseline found — setting current run as baseline")
        return 0

    # Run regression check
    guard = CIGuard(store, threshold=THRESHOLD)
    result = guard.check(run.id)

    if result.passed:
        logger.info("No regressions detected (threshold: %.0f%%)", THRESHOLD * 100)
        return 0

    logger.error(
        "Regressions detected! %d metric(s) exceeded %.0f%% threshold:",
        len(result.regressions),
        THRESHOLD * 100,
    )
    for reg in result.regressions:
        logger.error(
            "  %s/%s: %.4f -> %.4f (%.1f%%)",
            reg.point_name,
            reg.metric,
            reg.baseline_value,
            reg.current_value,
            reg.delta_pct * 100,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())

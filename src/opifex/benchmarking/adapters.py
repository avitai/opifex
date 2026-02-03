"""Adapter for converting BenchmarkResult lists to calibrax Run objects.

Bridges the opifex benchmarking pipeline (which produces BenchmarkResult lists)
with calibrax's Run-based analysis and storage APIs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from calibrax.core.models import (
    MetricDef,
    MetricDirection,
    MetricPriority,
    Point,
    Run,
)


if TYPE_CHECKING:
    from calibrax.core.result import BenchmarkResult


def results_to_run(
    results: list[BenchmarkResult],
    *,
    commit: str | None = None,
    branch: str | None = None,
    metric_defs: dict[str, MetricDef] | None = None,
) -> Run:
    """Convert a list of BenchmarkResult objects to a calibrax Run.

    Maps each BenchmarkResult to a Point:
    - ``BenchmarkResult.name`` -> ``Point.name``
    - ``BenchmarkResult.tags["dataset"]`` -> ``Point.scenario`` (default: "unknown")
    - ``BenchmarkResult.tags`` -> ``Point.tags``
    - ``BenchmarkResult.metrics`` -> ``Point.metrics`` (same Metric type)

    Args:
        results: List of benchmark results to convert.
        commit: Git commit hash to attach to the Run.
        branch: Git branch name to attach to the Run.
        metric_defs: Metric definitions for semantic interpretation.

    Returns:
        A calibrax Run containing one Point per BenchmarkResult.
    """
    points = tuple(
        Point(
            name=r.name,
            scenario=r.tags.get("dataset", "unknown"),
            tags=dict(r.tags),
            metrics=dict(r.metrics),
        )
        for r in results
    )

    return Run(
        points=points,
        commit=commit,
        branch=branch,
        metric_defs=metric_defs or {},
    )


def default_metric_defs() -> dict[str, MetricDef]:
    """Create standard metric definitions for scientific ML benchmarks.

    Returns:
        Dictionary mapping metric names to MetricDef objects with proper
        direction, units, and priority annotations.
    """
    return {
        "mse": MetricDef(
            name="mse",
            unit="",
            direction=MetricDirection.LOWER,
            group="accuracy",
            priority=MetricPriority.PRIMARY,
            description="Mean squared error",
        ),
        "mae": MetricDef(
            name="mae",
            unit="",
            direction=MetricDirection.LOWER,
            group="accuracy",
            priority=MetricPriority.SECONDARY,
            description="Mean absolute error",
        ),
        "relative_error": MetricDef(
            name="relative_error",
            unit="",
            direction=MetricDirection.LOWER,
            group="accuracy",
            priority=MetricPriority.PRIMARY,
            description="Relative L2 error",
        ),
        "r2_score": MetricDef(
            name="r2_score",
            unit="",
            direction=MetricDirection.HIGHER,
            group="accuracy",
            priority=MetricPriority.SECONDARY,
            description="Coefficient of determination",
        ),
        "rmse": MetricDef(
            name="rmse",
            unit="",
            direction=MetricDirection.LOWER,
            group="accuracy",
            priority=MetricPriority.SECONDARY,
            description="Root mean squared error",
        ),
        "throughput": MetricDef(
            name="throughput",
            unit="samples/sec",
            direction=MetricDirection.HIGHER,
            group="performance",
            priority=MetricPriority.SECONDARY,
            description="Inference throughput",
        ),
        "peak_memory_mb": MetricDef(
            name="peak_memory_mb",
            unit="MB",
            direction=MetricDirection.LOWER,
            group="performance",
            priority=MetricPriority.SECONDARY,
            description="Peak memory usage",
        ),
        "execution_time": MetricDef(
            name="execution_time",
            unit="sec",
            direction=MetricDirection.LOWER,
            group="performance",
            priority=MetricPriority.SECONDARY,
            description="Forward pass execution time",
        ),
    }

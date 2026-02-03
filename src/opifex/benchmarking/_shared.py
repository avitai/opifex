"""Shared constants and utilities for the benchmarking module.

Centralises domain inference, metric classification, and chemical accuracy
thresholds to eliminate duplication across sub-modules.
"""

from calibrax.core import BenchmarkResult


# ── Metric direction classification ──────────────────────────────────────

LOWER_IS_BETTER: frozenset[str] = frozenset(
    {"mse", "mae", "rmse", "relative_error", "mape", "execution_time"}
)
"""Metrics where a lower value indicates better performance."""

ACCURACY_METRIC_KEYS: tuple[str, ...] = (
    "mse",
    "mae",
    "rmse",
    "r2_score",
    "relative_error",
)
"""Standard accuracy metric keys used across reporting and analysis."""


# ── Chemical accuracy thresholds ─────────────────────────────────────────

CHEMICAL_ACCURACY_THRESHOLDS: dict[str, float] = {
    "quantum_computing": 1e-3,  # Hartree
    "materials_science": 5e-2,  # eV/atom
    "molecular_dynamics": 1e-2,  # eV
}
"""Domain-specific accuracy thresholds for chemical/physical accuracy checks."""


# ── Domain inference ─────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "quantum_computing": ("quantum", "dft", "molecular"),
    "fluid_dynamics": ("fluid", "burgers", "navier", "darcy"),
    "materials_science": ("material", "crystal", "solid"),
    "climate_modeling": ("climate", "weather", "atmospheric"),
}


def infer_domain(dataset_name: str) -> str:
    """Infer scientific domain from dataset name.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        Inferred domain string, or ``"general"`` if no match.
    """
    dataset_lower = dataset_name.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(term in dataset_lower for term in keywords):
            return domain
    return "general"


# ── Metric extraction helpers ────────────────────────────────────────────


def extract_metric_value(
    result: BenchmarkResult,
    metric_name: str,
    default: float = float("inf"),
) -> float:
    """Extract a scalar metric value from a BenchmarkResult.

    Args:
        result: Benchmark result to extract from.
        metric_name: Name of the metric.
        default: Value to return if metric is absent.

    Returns:
        The metric value as a float.
    """
    metric = result.metrics.get(metric_name)
    return metric.value if metric is not None else default

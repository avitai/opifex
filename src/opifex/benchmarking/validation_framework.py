"""Validation Framework for Opifex Advanced Benchmarking System.

Scientific accuracy validation against reference computational methods.
Provides convergence analysis, chemical accuracy assessment, and error analysis
for rigorous scientific computing validation.

Generic dataclasses (ConvergenceAnalysis, AccuracyAssessment) are replaced
by calibrax.validation equivalents (ConvergenceResult, AccuracyResult).
"""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from calibrax.core import BenchmarkResult
from calibrax.metrics import calculate_all as calculate_all_metrics
from calibrax.validation import (
    AccuracyResult,
    check_accuracy,
    check_convergence,
    ConvergenceResult,
)

from opifex.benchmarking._shared import CHEMICAL_ACCURACY_THRESHOLDS, infer_domain


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationReport:
    """Report of validation results against reference methods."""

    benchmark_name: str
    reference_method: str
    accuracy_metrics: dict[str, float]
    convergence_metrics: dict[str, float]
    chemical_accuracy_status: bool | None = None
    tolerance_violations: list[str] = field(default_factory=list)
    validation_passed: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        """Determine overall validation status."""
        passed = len(self.tolerance_violations) == 0 and (
            self.chemical_accuracy_status is None or self.chemical_accuracy_status
        )
        object.__setattr__(self, "validation_passed", passed)


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorAnalysis:
    """Error analysis between predictions and ground truth.

    Physics-specific: includes spatial and temporal pattern detection
    not available in calibrax generic validation.
    """

    global_errors: dict[str, float]
    local_errors: dict[str, jax.Array]
    error_distribution: dict[str, Any]
    outlier_analysis: dict[str, Any]
    spatial_error_patterns: dict[str, Any] | None = None
    temporal_error_patterns: dict[str, Any] | None = None


class ValidationFramework:
    """Scientific accuracy validation against reference computational methods.

    Provides:
    - Comparison against established computational methods (FEM, FDM, spectral)
    - Convergence rate analysis across multiple tolerance levels
    - Chemical accuracy assessment for quantum computing applications
    - Statistical error analysis with spatial and temporal pattern detection
    """

    def __init__(
        self,
        default_tolerances: list[float] | None = None,
        reference_methods: dict[str, Callable] | None = None,
    ) -> None:
        """Initialize validation framework.

        Args:
            default_tolerances: Default tolerance levels for convergence testing.
            reference_methods: Dictionary of reference computational methods.
        """
        self.default_tolerances = default_tolerances or [
            1e-2,
            1e-3,
            1e-4,
            1e-5,
            1e-6,
        ]
        self.reference_methods = reference_methods or {}

        self.chemical_accuracy_thresholds = dict(CHEMICAL_ACCURACY_THRESHOLDS)

    def validate_against_reference(
        self,
        result: BenchmarkResult,
        reference_method: str,
        reference_data: jax.Array | None = None,
        predictions: jax.Array | None = None,
    ) -> ValidationReport:
        """Validate benchmark results against reference computational method.

        Args:
            result: Benchmark result to validate.
            reference_method: Name of reference method.
            reference_data: Reference solution data (if available).
            predictions: Raw model predictions (if available). Required for
                meaningful accuracy metrics when reference_data is provided.

        Returns:
            Validation report with accuracy metrics and tolerance violations.
        """
        dataset = result.tags.get("dataset", result.name)
        fallback_metrics = {k: v.value for k, v in result.metrics.items()}

        # Collect data before constructing frozen report
        accuracy_metrics: dict[str, float] = {}
        notes = ""

        if reference_data is not None and predictions is not None:
            accuracy_metrics = _compute_accuracy_metrics(predictions, reference_data)
        elif reference_data is not None:
            logger.warning(
                "reference_data provided without predictions; "
                "using result metrics instead of computing accuracy"
            )
            accuracy_metrics = fallback_metrics
        elif reference_method in self.reference_methods:
            try:
                ref_method = self.reference_methods[reference_method]
                generated_reference_data = ref_method(result)
                if generated_reference_data is not None and predictions is not None:
                    accuracy_metrics = _compute_accuracy_metrics(
                        predictions, generated_reference_data
                    )
                elif generated_reference_data is not None:
                    logger.warning(
                        "Reference method produced data but no predictions "
                        "available; using result metrics"
                    )
                    accuracy_metrics = fallback_metrics
                else:
                    notes = "Reference method returned None"
                    accuracy_metrics = fallback_metrics
            except (ValueError, TypeError, RuntimeError) as e:
                notes = f"Reference method failed: {e}"
                warnings.warn(
                    f"Reference method '{reference_method}' failed: {e}",
                    stacklevel=2,
                )
        else:
            accuracy_metrics = fallback_metrics

        tolerance_violations = _check_tolerance_violations(
            accuracy_metrics, dataset, self._infer_domain
        )

        return ValidationReport(
            benchmark_name=dataset,
            reference_method=reference_method,
            accuracy_metrics=accuracy_metrics,
            convergence_metrics={},
            tolerance_violations=tolerance_violations,
            notes=notes,
        )

    def check_convergence_rates(
        self,
        results_sequence: list[BenchmarkResult],
        tolerances: list[float] | None = None,
    ) -> ConvergenceResult:
        """Analyze convergence rates across multiple tolerance levels.

        Delegates to calibrax.validation.check_convergence after extracting
        metric series from BenchmarkResult sequence.

        Args:
            results_sequence: Sequence of results at different tolerance levels.
            tolerances: Tolerance levels tested.

        Returns:
            ConvergenceResult from calibrax with rates and achievement flags.
        """
        if tolerances is None:
            tolerances = self.default_tolerances[: len(results_sequence)]

        # Extract metric series from calibrax BenchmarkResult objects
        metric_series: dict[str, list[float]] = {}
        for metric_name in ("mse", "mae", "relative_error"):
            values = []
            for result in results_sequence:
                metric = result.metrics.get(metric_name)
                if metric is not None:
                    values.append(metric.value)
            if values:
                metric_series[metric_name] = values

        return check_convergence(metric_series, tolerances)

    def assess_chemical_accuracy(
        self,
        result: BenchmarkResult,
        target_accuracy: float | None = None,
        accuracy_type: str = "chemical_accuracy",
    ) -> AccuracyResult:
        """Assess chemical accuracy for quantum computing applications.

        Delegates to calibrax.validation.check_accuracy after extracting
        the appropriate metric from the BenchmarkResult.

        Args:
            result: Benchmark result to assess.
            target_accuracy: Target accuracy threshold (defaults to domain standard).
            accuracy_type: Type of accuracy being assessed.

        Returns:
            AccuracyResult from calibrax with pass/fail and margin.
        """
        dataset = result.tags.get("dataset", result.name)
        domain = self._infer_domain(dataset)

        if target_accuracy is None:
            target_accuracy = self.chemical_accuracy_thresholds.get(domain, 1e-3)

        # Select metric based on accuracy type
        if accuracy_type == "chemical_accuracy":
            metric = result.metrics.get("mse")
            achieved = metric.value if metric is not None else float("inf")
            units = "Hartree" if domain == "quantum_computing" else "energy_units"
        elif accuracy_type == "force_accuracy":
            metric = result.metrics.get("mae")
            achieved = metric.value if metric is not None else float("inf")
            units = "eV/Ang"
        else:
            metric = result.metrics.get("relative_error")
            achieved = metric.value if metric is not None else float("inf")
            units = "relative"

        return check_accuracy(
            achieved=achieved,
            target=target_accuracy,
            metric_type=accuracy_type,
            units=units,
        )

    def generate_error_analysis(
        self,
        predictions: jax.Array,
        ground_truth: jax.Array,
        spatial_coords: jax.Array | None = None,
        temporal_coords: jax.Array | None = None,
    ) -> ErrorAnalysis:
        """Generate error analysis for predictions vs ground truth.

        Args:
            predictions: Model predictions.
            ground_truth: Ground truth data.
            spatial_coords: Spatial coordinates (if available).
            temporal_coords: Temporal coordinates (if available).

        Returns:
            ErrorAnalysis with global, local, distribution, and pattern data.
        """
        errors = predictions - ground_truth
        abs_errors = jnp.abs(errors)

        global_errors = {
            "mse": float(jnp.mean(errors**2)),
            "mae": float(jnp.mean(abs_errors)),
            "max_error": float(jnp.max(abs_errors)),
            "std_error": float(jnp.std(errors)),
            "rmse": float(jnp.sqrt(jnp.mean(errors**2))),
        }

        local_errors = {
            "absolute_errors": abs_errors,
            "relative_errors": abs_errors / (jnp.abs(ground_truth) + 1e-8),
            "squared_errors": errors**2,
        }

        error_distribution = {
            "percentiles": {
                "p50": float(jnp.percentile(abs_errors, 50)),
                "p75": float(jnp.percentile(abs_errors, 75)),
                "p90": float(jnp.percentile(abs_errors, 90)),
                "p95": float(jnp.percentile(abs_errors, 95)),
                "p99": float(jnp.percentile(abs_errors, 99)),
            },
            "moments": {
                "mean": float(jnp.mean(errors)),
                "variance": float(jnp.var(errors)),
                "skewness": float(_compute_skewness(errors)),
                "kurtosis": float(_compute_kurtosis(errors)),
            },
        }

        error_threshold = global_errors["mae"] + 3 * global_errors["std_error"]
        outlier_mask = abs_errors > error_threshold

        outlier_analysis = {
            "outlier_count": int(jnp.sum(outlier_mask)),
            "outlier_percentage": float(jnp.mean(outlier_mask) * 100),
            "outlier_threshold": float(error_threshold),
            "max_outlier_error": float(jnp.max(abs_errors * outlier_mask)),
        }

        spatial_patterns = None
        if spatial_coords is not None:
            spatial_patterns = _analyze_spatial_patterns(errors, spatial_coords)

        temporal_patterns = None
        if temporal_coords is not None:
            temporal_patterns = _analyze_temporal_patterns(errors, temporal_coords)

        return ErrorAnalysis(
            global_errors=global_errors,
            local_errors=local_errors,
            error_distribution=error_distribution,
            outlier_analysis=outlier_analysis,
            spatial_error_patterns=spatial_patterns,
            temporal_error_patterns=temporal_patterns,
        )

    @staticmethod
    def _infer_domain(dataset_name: str) -> str:
        """Infer scientific domain from dataset name."""
        return infer_domain(dataset_name)


def _compute_accuracy_metrics(
    predictions: jax.Array,
    reference_data: jax.Array,
) -> dict[str, float]:
    """Compute accuracy metrics of predictions against reference data.

    Delegates to calibrax.metrics.calculate_all for standard metric computation.

    Args:
        predictions: Model prediction array.
        reference_data: Reference solution data.

    Returns:
        Dictionary of accuracy metric name to value.
    """
    return calculate_all_metrics(predictions, reference_data)


def _check_tolerance_violations(
    metrics: dict[str, float],
    dataset_name: str,
    infer_domain: Callable[[str], str],
) -> list[str]:
    """Check for tolerance violations based on dataset domain.

    Args:
        metrics: Accuracy metrics dictionary.
        dataset_name: Dataset name for domain inference.
        infer_domain: Callable to infer domain from dataset name.

    Returns:
        List of violation description strings.
    """
    violations: list[str] = []
    domain = infer_domain(dataset_name)

    if domain == "quantum_computing":
        if metrics.get("mse", float("inf")) > 1e-4:
            violations.append("MSE exceeds quantum computing tolerance (1e-4)")
        if metrics.get("relative_error", float("inf")) > 1e-3:
            violations.append("Relative error exceeds quantum tolerance (1e-3)")
    elif domain == "fluid_dynamics":
        if metrics.get("mse", float("inf")) > 1e-2:
            violations.append("MSE exceeds fluid dynamics tolerance (1e-2)")
        if metrics.get("relative_error", float("inf")) > 1e-1:
            violations.append("Relative error exceeds fluid dynamics tolerance (1e-1)")
    elif domain == "materials_science":
        if metrics.get("mse", float("inf")) > 1e-3:
            violations.append("MSE exceeds materials science tolerance (1e-3)")
        if metrics.get("relative_error", float("inf")) > 5e-2:
            violations.append("Relative error exceeds materials tolerance (5e-2)")

    return violations


def _compute_skewness(data: jax.Array) -> jax.Array:
    """Compute skewness of data."""
    mean = jnp.mean(data)
    std = jnp.std(data)
    normalized = (data - mean) / (std + 1e-8)
    return jnp.mean(normalized**3)


def _compute_kurtosis(data: jax.Array) -> jax.Array:
    """Compute excess kurtosis of data."""
    mean = jnp.mean(data)
    std = jnp.std(data)
    normalized = (data - mean) / (std + 1e-8)
    return jnp.mean(normalized**4) - 3


def _analyze_spatial_patterns(
    errors: jax.Array, spatial_coords: jax.Array
) -> dict[str, Any]:
    """Analyze spatial patterns in errors."""
    return {
        "spatial_correlation": float(
            jnp.corrcoef(errors.flatten(), spatial_coords[:, 0].flatten())[0, 1]
        ),
        "spatial_variance": float(jnp.var(errors, axis=tuple(range(1, errors.ndim)))),
    }


def _analyze_temporal_patterns(
    errors: jax.Array, temporal_coords: jax.Array
) -> dict[str, Any]:
    """Analyze temporal patterns in errors."""
    return {
        "temporal_correlation": float(
            jnp.corrcoef(errors.flatten(), temporal_coords.flatten())[0, 1]
        ),
        "temporal_trend": float(
            jnp.polyfit(temporal_coords.flatten(), errors.flatten(), 1)[0]
        ),
    }

"""Chemical accuracy validation for scientific ML benchmarks.

Assesses whether a benchmark result meets domain-specific accuracy thresholds
by delegating to ``calibrax.validation.check_accuracy()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from calibrax.validation.accuracy import AccuracyResult, check_accuracy

from opifex.benchmarking._shared import CHEMICAL_ACCURACY_THRESHOLDS


if TYPE_CHECKING:
    from calibrax.core.result import BenchmarkResult


# Default chemical accuracy thresholds per domain
_DEFAULT_THRESHOLDS: dict[str, float] = dict(CHEMICAL_ACCURACY_THRESHOLDS)

_DOMAIN_UNITS: dict[str, str] = {
    "quantum_computing": "Hartree",
    "materials_science": "eV/atom",
    "molecular_dynamics": "eV",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ChemicalAccuracyAssessment:
    """Result of a chemical accuracy assessment.

    Wraps a ``calibrax.validation.AccuracyResult`` with domain context
    and actionable recommendations.

    Attributes:
        passed: Whether the result meets the chemical accuracy threshold.
        domain: Scientific domain used for assessment.
        threshold: Accuracy threshold applied.
        achieved: Achieved error value.
        margin: Headroom (positive) or deficit (negative) relative to threshold.
        accuracy_result: Underlying calibrax AccuracyResult.
        recommendations: Suggested actions if assessment fails.
    """

    passed: bool
    domain: str
    threshold: float
    achieved: float
    margin: float
    accuracy_result: AccuracyResult
    recommendations: tuple[str, ...] = field(default_factory=tuple)


class ChemicalAccuracyValidator:
    """Validates benchmark results against domain-specific chemical accuracy thresholds.

    Delegates accuracy computation to ``calibrax.validation.check_accuracy()``.

    Note: Registry registration intentionally omitted -- validators are
    instantiated directly, not discovered dynamically.

    Args:
        thresholds: Custom domain-to-threshold mapping. Merged with defaults.
        error_metric: Metric name to extract from BenchmarkResult.
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        error_metric: str = "relative_error",
    ) -> None:
        """Initialize the validator.

        Args:
            thresholds: Custom domain-to-threshold mapping. Merged with defaults.
            error_metric: Metric name to extract from BenchmarkResult.
        """
        self._thresholds = dict(_DEFAULT_THRESHOLDS)
        if thresholds:
            self._thresholds.update(thresholds)
        self._error_metric = error_metric

    def assess(
        self,
        result: BenchmarkResult,
        domain: str | None = None,
    ) -> ChemicalAccuracyAssessment:
        """Assess whether a benchmark result meets chemical accuracy for a domain.

        Args:
            result: Benchmark result containing error metrics.
            domain: Scientific domain. Auto-detected from result tags/domain if None.

        Returns:
            Assessment with pass/fail, margin, and recommendations.

        Raises:
            ValueError: If domain is unknown and cannot be auto-detected.
            KeyError: If the error metric is not present in the result.
        """
        resolved_domain = self._resolve_domain(result, domain)
        threshold = self._thresholds[resolved_domain]
        units = _DOMAIN_UNITS.get(resolved_domain, "relative")

        achieved = result.metrics[self._error_metric].value
        accuracy_result = check_accuracy(
            achieved=achieved,
            target=threshold,
            metric_type="chemical_accuracy",
            units=units,
        )

        recommendations = self._generate_recommendations(
            accuracy_result, resolved_domain, threshold, achieved
        )

        return ChemicalAccuracyAssessment(
            passed=accuracy_result.passed,
            domain=resolved_domain,
            threshold=threshold,
            achieved=achieved,
            margin=accuracy_result.margin,
            accuracy_result=accuracy_result,
            recommendations=tuple(recommendations),
        )

    def _resolve_domain(
        self,
        result: BenchmarkResult,
        domain: str | None,
    ) -> str:
        """Resolve the domain from explicit argument or result metadata.

        Args:
            result: Benchmark result to inspect.
            domain: Explicit domain override.

        Returns:
            Resolved domain name.

        Raises:
            ValueError: If domain cannot be resolved.
        """
        if domain is not None:
            if domain not in self._thresholds:
                raise ValueError(
                    f"Unknown domain '{domain}'. "
                    f"Known domains: {list(self._thresholds.keys())}"
                )
            return domain

        # Auto-detect from result
        for candidate in (result.domain, result.tags.get("domain", "")):
            if candidate in self._thresholds:
                return candidate

        raise ValueError(
            f"Cannot auto-detect domain from result. "
            f"Specify domain explicitly or add a 'domain' tag. "
            f"Known domains: {list(self._thresholds.keys())}"
        )

    def _generate_recommendations(
        self,
        accuracy_result: AccuracyResult,
        domain: str,
        threshold: float,
        achieved: float,
    ) -> list[str]:
        """Generate actionable recommendations for failed assessments.

        Args:
            accuracy_result: The calibrax accuracy result.
            domain: Scientific domain.
            threshold: Target threshold.
            achieved: Achieved error.

        Returns:
            List of recommendation strings.
        """
        if accuracy_result.passed:
            return []

        ratio = achieved / threshold if threshold > 0 else float("inf")
        recommendations = [
            f"Error ({achieved:.2e}) exceeds {domain} threshold ({threshold:.2e})"
        ]

        if ratio > 10:
            recommendations.append("Consider a fundamentally different architecture")
        elif ratio > 2:
            recommendations.append("Increase model capacity or training iterations")
        else:
            recommendations.append(
                "Fine-tune hyperparameters — result is close to threshold"
            )

        return recommendations

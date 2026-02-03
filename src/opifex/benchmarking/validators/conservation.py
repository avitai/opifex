"""Conservation law validation for scientific ML benchmarks.

Orchestrates conservation law checks from ``opifex.core.physics.conservation``
and optionally delegates convergence analysis to calibrax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from calibrax.validation.convergence import check_convergence, ConvergenceResult
from calibrax.validation.framework import ValidationReport

from opifex.core.physics.conservation import (
    energy_violation,
    mass_violation,
    momentum_violation,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True, kw_only=True)
class ConservationReport:
    """Report from conservation law validation.

    Uses a local dataclass instead of ``calibrax.validation.ValidationReport``
    because conservation checking requires violation *magnitudes*
    (``dict[str, float]``) rather than textual violation descriptions
    (``tuple[str, ...]``), plus domain-specific fields (``worst_violation``,
    ``all_conserved``) that ``ValidationReport`` does not provide.
    :meth:`to_validation_report` bridges the two when calibrax interop is needed.

    Attributes:
        violations: Conservation law name to violation magnitude.
        all_conserved: True if all violations are zero (within tolerance).
        worst_violation: Maximum violation across all checked laws.
        convergence: Optional convergence result from multi-resolution analysis.
    """

    violations: dict[str, float]
    all_conserved: bool
    worst_violation: float
    convergence: ConvergenceResult | None = None

    def to_validation_report(self) -> ValidationReport:
        """Convert to a calibrax ``ValidationReport`` for cross-tool interop.

        Returns:
            A ``ValidationReport`` with violation magnitudes as accuracy_metrics
            and textual summaries in the violations tuple.
        """
        return ValidationReport(
            name="conservation_check",
            reference="physics_laws",
            accuracy_metrics=dict(self.violations),
            violations=tuple(
                f"{law}: {mag:.2e}" for law, mag in self.violations.items() if mag > 0
            ),
            passed=self.all_conserved,
            notes=f"worst_violation={self.worst_violation:.2e}",
        )


class ConservationValidator:
    """Validates physics conservation laws on model predictions.

    Orchestrates existing pure-JAX functions from
    ``opifex.core.physics.conservation`` and provides a unified interface.

    Args:
        laws: Conservation laws to check. Defaults to energy and momentum.
        energy_tolerance: Tolerance for energy conservation check.
        momentum_tolerance: Tolerance for momentum conservation check.
        mass_target: Target mass for mass conservation check.
        mass_tolerance: Tolerance for mass conservation check.
    """

    def __init__(
        self,
        laws: Sequence[str] | None = None,
        energy_tolerance: float = 1e-6,
        momentum_tolerance: float = 1e-5,
        mass_target: float = 1.0,
        mass_tolerance: float = 1e-4,
    ) -> None:
        """Initialize the conservation validator.

        Args:
            laws: Conservation laws to check. Defaults to energy and momentum.
            energy_tolerance: Tolerance for energy conservation check.
            momentum_tolerance: Tolerance for momentum conservation check.
            mass_target: Target mass for mass conservation check.
            mass_tolerance: Tolerance for mass conservation check.
        """
        self._laws = list(laws) if laws is not None else ["energy", "momentum"]
        self._energy_tolerance = energy_tolerance
        self._momentum_tolerance = momentum_tolerance
        self._mass_target = mass_target
        self._mass_tolerance = mass_tolerance

    def validate(
        self,
        y_pred: jax.Array,
        y_true: jax.Array,
    ) -> ConservationReport:
        """Validate conservation laws on a single prediction set.

        Args:
            y_pred: Model predictions.
            y_true: Ground truth values.

        Returns:
            ConservationReport with violations and overall status.
        """
        violations: dict[str, float] = {}

        for law in self._laws:
            violations[law] = float(self._compute_violation(law, y_pred, y_true))

        all_conserved = all(v == 0.0 for v in violations.values())
        worst = max(violations.values()) if violations else 0.0

        return ConservationReport(
            violations=violations,
            all_conserved=all_conserved,
            worst_violation=worst,
        )

    def validate_convergence(
        self,
        predictions: Sequence[jax.Array],
        truths: Sequence[jax.Array],
        tolerances: Sequence[float],
    ) -> ConvergenceResult:
        """Validate conservation convergence across multiple resolutions.

        Computes violations at each resolution and delegates convergence
        analysis to ``calibrax.validation.check_convergence()``.

        Args:
            predictions: Predictions at increasing resolutions.
            truths: Ground truths at increasing resolutions.
            tolerances: Tolerance thresholds for convergence check.

        Returns:
            ConvergenceResult with rates and achievement flags.
        """
        # Compute violation series for each law across resolutions
        metric_series: dict[str, list[float]] = {law: [] for law in self._laws}

        for y_pred, y_true in zip(predictions, truths, strict=True):
            for law in self._laws:
                violation = float(self._compute_violation(law, y_pred, y_true))
                # Ensure positive values for log-based convergence computation
                metric_series[law].append(max(violation, 1e-15))

        return check_convergence(metric_series, tolerances)

    def _compute_violation(
        self,
        law: str,
        y_pred: jax.Array,
        y_true: jax.Array,
    ) -> jax.Array:
        """Compute violation for a specific conservation law.

        Args:
            law: Conservation law name.
            y_pred: Model predictions.
            y_true: Ground truth values.

        Returns:
            Scalar violation value.
        """
        if law == "energy":
            return energy_violation(y_pred, y_true, tolerance=self._energy_tolerance)
        if law == "momentum":
            return momentum_violation(
                y_pred, y_true, tolerance=self._momentum_tolerance
            )
        if law == "mass":
            return mass_violation(
                y_pred, self._mass_target, tolerance=self._mass_tolerance
            )
        return jnp.array(0.0)

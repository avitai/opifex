"""Conservation law enforcement for physics-informed learning.

This module provides the **single source of truth** for all conservation law
implementations across the Opifex framework.

All functions are pure, JAX-compatible, and designed for use in
JIT-compiled training loops with full support for automatic differentiation.

Key Features:
- Conservation law violation computation (energy, momentum, mass, etc.)
- Tolerance-based violation filtering
- JAX transformations (JIT, vmap, grad) compatible
- Component-wise momentum conservation
- Parameter constraint application

Author: Opifex Framework Team
Date: October 2025
License: MIT
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from collections.abc import Callable


class ConservationLaw(Enum):
    """Enum defining all supported conservation laws.

    This is the single source of truth for conservation law names
    across the entire Opifex framework.
    """

    ENERGY = "energy"
    MOMENTUM = "momentum"
    ANGULAR_MOMENTUM = "angular_momentum"
    MASS = "mass"
    CHARGE = "charge"
    PARTICLE_NUMBER = "particle_number"
    PROBABILITY = "probability"


def energy_violation(
    y_pred: jax.Array,
    y_true: jax.Array,
    tolerance: float = 1e-6,
    monitoring_enabled: bool = True,
) -> jax.Array:
    """Compute energy conservation violation with tolerance checking.

    Only penalizes violations that exceed the configured tolerance threshold.
    Respects the monitoring flag to enable/disable checking.

    Args:
        y_pred: Predicted values (shape: [batch, ...])
        y_true: True values (shape: [batch, ...])
        tolerance: Tolerance threshold for violations
        monitoring_enabled: Whether monitoring is enabled

    Returns:
        Energy conservation violation (0 if below tolerance or disabled)

    Examples:
        >>> y_pred = jnp.array([[1.0, 2.0]])
        >>> y_true = jnp.array([[1.0, 2.0]])
        >>> violation = energy_violation(y_pred, y_true)
        >>> print(violation)  # Should be ~0.0
    """
    # Compute energy as sum of squares
    energy_pred = jnp.sum(y_pred**2, axis=-1)
    energy_true = jnp.sum(y_true**2, axis=-1)

    # Compute squared difference (MSE)
    violation = jnp.mean((energy_pred - energy_true) ** 2)

    # Apply tolerance and monitoring flags (JAX-compatible)
    return jnp.where(
        monitoring_enabled, jnp.where(violation > tolerance, violation, 0.0), 0.0
    )


def momentum_violation(
    y_pred: jax.Array,
    y_true: jax.Array,
    tolerance: float = 1e-5,
) -> jax.Array:
    """Compute momentum conservation violation (component-wise).

    Momentum is a vector quantity - each component must be conserved separately.
    This ensures physical correctness (momentum is NOT a scalar!).

    Args:
        y_pred: Predicted values (shape: [..., num_components])
        y_true: True values (shape: [..., num_components])
        tolerance: Tolerance threshold for violations

    Returns:
        Component-wise momentum conservation violation

    Examples:
        >>> y_pred = jnp.array([[1.0, 2.0, 3.0]])  # 3D momentum
        >>> y_true = jnp.array([[1.0, 2.0, 3.0]])
        >>> violation = momentum_violation(y_pred, y_true)
        >>> print(violation)  # Should be ~0.0
    """
    # Compute total momentum for each component (sum over batch/spatial dims)
    # Keep last dimension for component-wise comparison
    momentum_pred = jnp.sum(y_pred, axis=tuple(range(y_pred.ndim - 1)))
    momentum_true = jnp.sum(y_true, axis=tuple(range(y_true.ndim - 1)))

    # Component-wise violation
    component_violations = jnp.abs(momentum_pred - momentum_true)

    # Apply tolerance threshold using JAX operations
    violations_above_threshold = jnp.where(
        component_violations > tolerance, component_violations**2, 0.0
    )

    # Return mean violation across all components
    return jnp.mean(violations_above_threshold)


def mass_violation(
    y_pred: jax.Array,
    target_mass: float,
    tolerance: float = 1e-4,
) -> jax.Array:
    """Compute mass conservation violation with tolerance checking.

    Args:
        y_pred: Predicted values
        target_mass: Target total mass
        tolerance: Tolerance threshold for violations

    Returns:
        Mass conservation violation (0 if below tolerance)

    Examples:
        >>> y_pred = jnp.array([[0.5, 0.5]])  # Total mass = 1.0
        >>> violation = mass_violation(y_pred, target_mass=1.0)
        >>> print(violation)  # Should be ~0.0
    """
    # Compute total mass as sum of absolute values
    total_mass = jnp.sum(jnp.abs(y_pred), axis=-1)

    # Compute violation magnitude
    violation_magnitude = jnp.abs(total_mass - target_mass)
    violation = jnp.mean(violation_magnitude)

    # Apply tolerance threshold (square for gradient smoothness)
    return jnp.where(violation > tolerance, violation**2, 0.0)


def particle_number_violation(
    y_pred: jax.Array,
    target_particle_number: float,
    tolerance: float = 1e-4,
) -> jax.Array:
    """Compute particle number conservation violation with tolerance checking.

    Args:
        y_pred: Predicted values
        target_particle_number: Target number of particles
        tolerance: Tolerance threshold for violations

    Returns:
        Particle number conservation violation

    Examples:
        >>> y_pred = jnp.array([[1.0, 1.0, 1.0]])  # 3 particles
        >>> violation = particle_number_violation(y_pred, target_particle_number=3.0)
        >>> print(violation)  # Should be ~0.0
    """
    # Compute particle number as sum of absolute values
    particle_pred = jnp.sum(jnp.abs(y_pred), axis=-1)

    # Compute violation magnitude
    violation_magnitude = jnp.abs(particle_pred - target_particle_number)
    violation = jnp.mean(violation_magnitude)

    # Apply tolerance threshold (square for gradient smoothness)
    return jnp.where(violation > tolerance, violation**2, 0.0)


def symmetry_violation(
    y_pred: jax.Array,
    tolerance: float = 1e-6,
) -> jax.Array:
    """Compute symmetry conservation violation with tolerance checking.

    Checks reflection symmetry as a basic symmetry operation.
    Only penalizes violations that exceed the configured tolerance threshold.

    Args:
        y_pred: Predicted values
        tolerance: Tolerance threshold for violations

    Returns:
        Symmetry preservation violation

    Examples:
        >>> y_pred = jnp.array([[1.0, 2.0, 2.0, 1.0]])  # Symmetric
        >>> violation = symmetry_violation(y_pred)
        >>> print(violation)  # Should be ~0.0
    """
    # Check reflection symmetry (basic symmetry operation)
    reflection_violation = jnp.mean((y_pred - jnp.flip(y_pred, axis=-1)) ** 2)

    # Apply tolerance threshold
    return jnp.where(reflection_violation > tolerance, reflection_violation, 0.0)


def apply_conservation_constraint(
    params: jax.Array,
    law: ConservationLaw,
    weight: float = 1.0,
) -> jax.Array:
    """Apply conservation constraint to model parameters.

    Args:
        params: Model parameters to constrain
        law: Conservation law to enforce
        weight: Constraint weight (0-1), where:
            - 0.0 = no constraint (returns original params)
            - 1.0 = full constraint
            - 0.5 = partial constraint (blend of constrained and original)

    Returns:
        Constrained parameters

    Examples:
        >>> params = jnp.array([[1.0, 2.0, 3.0]])
        >>> constrained = apply_conservation_constraint(
        ...     params, ConservationLaw.ENERGY, weight=1.0
        ... )
        >>> # constrained will have normalized energy
    """
    if law == ConservationLaw.ENERGY:
        # Normalize parameter norm for energy conservation
        param_norm = jnp.linalg.norm(params, axis=-1, keepdims=True)
        target_norm = jnp.sqrt(params.shape[-1])
        constrained = params * (target_norm / (param_norm + 1e-8))
        return weight * constrained + (1 - weight) * params

    if law == ConservationLaw.MASS:
        # Enforce mass conservation through normalization
        total_mass = jnp.sum(jnp.abs(params), axis=-1, keepdims=True)
        constrained = params / (total_mass + 1e-8)
        return weight * constrained + (1 - weight) * params

    # Default: return params unchanged for unsupported laws
    return params


class MultiScalePhysics:
    """Multi-scale physics integration for hierarchical systems.

    Supports different physical scales (molecular, atomic, electronic) with
    scale-specific loss computations and configurable weighting.
    """

    def __init__(self, scales: list[str], scale_weights: dict[str, float]):
        """Initialize multi-scale physics system.

        Args:
            scales: List of physical scales to consider
            scale_weights: Weight for each scale
        """
        self.scales = scales
        self.scale_weights = scale_weights

        # Normalize weights to sum to 1
        total_weight = sum(scale_weights.values())
        if total_weight > 0:
            self.normalized_weights = {
                scale: weight / total_weight for scale, weight in scale_weights.items()
            }
        else:
            # Equal weighting if not specified
            self.normalized_weights = {scale: 1.0 / len(scales) for scale in scales}

    def compute_loss(
        self,
        x: jax.Array,
        y_pred: jax.Array,
        y_true: jax.Array,
        base_loss_fn: Callable,
    ) -> jax.Array:
        """Compute multi-scale physics loss.

        Args:
            x: Input data
            y_pred: Predicted values
            y_true: True values
            base_loss_fn: Base loss function (e.g., MSE)

        Returns:
            Weighted combination of scale-specific losses
        """
        total_loss = jnp.array(0.0)

        for scale in self.scales:
            weight = self.normalized_weights[scale]
            scale_loss = self._compute_scale_loss(
                scale, x, y_pred, y_true, base_loss_fn
            )
            total_loss = total_loss + weight * scale_loss

        return total_loss

    def _compute_scale_loss(
        self,
        scale: str,
        x: jax.Array,
        y_pred: jax.Array,
        y_true: jax.Array,
        base_loss_fn: Callable,
    ) -> jax.Array:
        """Compute loss for a specific physical scale.

        Args:
            scale: Physical scale name
            x: Input data
            y_pred: Predicted values
            y_true: True values
            base_loss_fn: Base loss function

        Returns:
            Scale-specific loss
        """
        if scale == "molecular":
            # Molecular scale: intermolecular interactions (MSE with moderate weight)
            return base_loss_fn(y_pred, y_true) * 0.5

        if scale == "atomic":
            # Atomic scale: intramolecular interactions (MAE with lower weight)
            return jnp.mean(jnp.abs(y_pred - y_true)) * 0.3

        if scale == "electronic":
            # Electronic scale: quantum effects (higher order with lower weight)
            return jnp.mean((y_pred - y_true) ** 4) * 0.2

        # Default: use base loss function
        return base_loss_fn(y_pred, y_true)

    def get_normalized_weights(self) -> dict[str, float]:
        """Get normalized scale weights that sum to 1.0.

        Returns:
            Dictionary of normalized weights
        """
        return self.normalized_weights


class AdaptiveConstraintWeighting:
    """Adaptive weight adjustment for physics constraints.

    Dynamically adjusts constraint weights based on violation severity,
    increasing weights for constraints that are violated more frequently
    or severely.
    """

    def __init__(
        self,
        constraints: list[str],
        initial_weights: dict[str, float],
        adaptation_rate: float = 0.1,
    ):
        """Initialize adaptive weighting system.

        Args:
            constraints: List of constraint names
            initial_weights: Initial weight for each constraint
            adaptation_rate: Rate of weight adaptation (0-1)
        """
        self.constraints = constraints
        self.current_weights = dict(initial_weights)
        self.adaptation_rate = adaptation_rate

    def update_weights(
        self, constraint_violations: dict[str, float]
    ) -> dict[str, float]:
        """Update constraint weights based on violations.

        Constraints with higher violations get increased weights to
        prioritize fixing those constraints.

        Args:
            constraint_violations: Current violation for each constraint

        Returns:
            Updated weights (normalized to sum to 1.0)
        """
        if not constraint_violations:
            return self.current_weights

        max_violation = max(constraint_violations.values())

        if max_violation == 0:
            # No violations, maintain current weights
            return self.current_weights

        # Update weights based on violation ratios
        new_weights = {}
        for constraint in self.constraints:
            violation = constraint_violations.get(constraint, 0.0)
            violation_ratio = violation / max_violation

            # Increase weight for constraints with higher violations
            current_weight = self.current_weights.get(
                constraint, 1.0 / len(self.constraints)
            )
            new_weight = current_weight * (1.0 + self.adaptation_rate * violation_ratio)
            new_weights[constraint] = new_weight

        # Normalize weights to sum to 1.0
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for constraint in new_weights:
                new_weights[constraint] /= total_weight

        self.current_weights = new_weights
        return new_weights

    def get_current_weights(self) -> dict[str, float]:
        """Get current constraint weights.

        Returns:
            Dictionary of current weights
        """
        return self.current_weights


class ConstraintAggregator:
    """Compose multiple physics constraints into a single loss.

    This class orchestrates conservation law enforcement, multi-scale physics,
    and adaptive weighting to create a unified constraint loss function.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize constraint aggregator.

        Args:
            config: Configuration dictionary with constraint settings
        """
        self.config = config
        self.conservation_laws = config.get("conservation_laws", [])

        # Conservation law configuration
        self.energy_tolerance = config.get("energy_conservation_tolerance", 1e-6)
        self.energy_monitoring = config.get("energy_conservation_monitoring", True)
        self.momentum_tolerance = config.get("momentum_conservation_tolerance", 1e-5)
        self.particle_tolerance = config.get("particle_conservation_tolerance", 1e-4)
        self.target_particle_number = config.get("target_particle_number", 10.0)
        self.symmetry_tolerance = config.get("symmetry_tolerance", 1e-6)

        # Adaptive weighting setup
        self.adaptive_weighting = config.get("adaptive_weighting", False)
        if self.adaptive_weighting and self.conservation_laws:
            initial_weights = {
                law: 1.0 / len(self.conservation_laws) for law in self.conservation_laws
            }
            self.weight_manager = AdaptiveConstraintWeighting(
                constraints=self.conservation_laws,
                initial_weights=initial_weights,
                adaptation_rate=config.get("adaptation_rate", 0.1),
            )

    def compute_constraint_loss(
        self, x: jax.Array, y_pred: jax.Array, y_true: jax.Array
    ) -> jax.Array:
        """Compute aggregated constraint loss.

        Args:
            x: Input data
            y_pred: Predicted values
            y_true: True values

        Returns:
            Total constraint loss
        """
        if not self.conservation_laws:
            return jnp.array(0.0)

        total_loss = jnp.array(0.0)

        # Get current weights (adaptive or uniform)
        if self.adaptive_weighting and hasattr(self, "weight_manager"):
            weights = self.weight_manager.get_current_weights()
        else:
            # Uniform weighting
            weights = {
                law: 1.0 / len(self.conservation_laws) for law in self.conservation_laws
            }

        # Compute each conservation law violation
        for law in self.conservation_laws:
            weight = weights[law]
            violation = self._compute_conservation_violation(law, x, y_pred, y_true)
            total_loss = total_loss + weight * violation

        return total_loss

    def compute_constraint_metrics(
        self, x: jax.Array, y_pred: jax.Array, y_true: jax.Array
    ) -> dict[str, float]:
        """Compute constraint violation metrics.

        Args:
            x: Input data
            y_pred: Predicted values
            y_true: True values

        Returns:
            Dictionary of constraint metrics
        """
        metrics = {}

        for law in self.conservation_laws:
            violation = self._compute_conservation_violation(law, x, y_pred, y_true)
            metrics[f"{law}_conservation"] = float(violation)

        # Update adaptive weights if enabled
        if self.adaptive_weighting and hasattr(self, "weight_manager") and metrics:
            self.weight_manager.update_weights(metrics)

        return metrics

    def _compute_conservation_violation(
        self, law: str, x: jax.Array, y_pred: jax.Array, y_true: jax.Array
    ) -> jax.Array:
        """Compute violation for a specific conservation law.

        Args:
            law: Conservation law name
            x: Input data
            y_pred: Predicted values
            y_true: True values

        Returns:
            Conservation violation value
        """
        if law == "energy":
            return energy_violation(
                y_pred,
                y_true,
                tolerance=self.energy_tolerance,
                monitoring_enabled=self.energy_monitoring,
            )

        if law == "momentum":
            return momentum_violation(y_pred, y_true, tolerance=self.momentum_tolerance)

        if law == "particle_number":
            return particle_number_violation(
                y_pred,
                target_particle_number=self.target_particle_number,
                tolerance=self.particle_tolerance,
            )

        if law == "symmetry":
            return symmetry_violation(y_pred, tolerance=self.symmetry_tolerance)

        # Unknown law, return zero
        return jnp.array(0.0)


__all__ = [
    "AdaptiveConstraintWeighting",
    "ConservationLaw",
    "ConstraintAggregator",
    "MultiScalePhysics",
    "apply_conservation_constraint",
    "energy_violation",
    "mass_violation",
    "momentum_violation",
    "particle_number_violation",
    "symmetry_violation",
]

"""Tests for conservation law implementations.

Following TDD principles: These tests are written FIRST to define
the expected behavior of the conservation module.

All tests use JAX arrays and are designed to verify:
- Conservation law enum correctness
- Violation detection accuracy
- Tolerance threshold behavior
- JAX compatibility (JIT, vmap, etc.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.core.physics.conservation import (
    apply_conservation_constraint,
    ConservationLaw,
    energy_violation,
    mass_violation,
    momentum_violation,
    particle_number_violation,
    symmetry_violation,
)


class TestConservationLawEnum:
    """Test ConservationLaw enum definition."""

    def test_enum_members(self):
        """Test that all expected conservation laws are defined."""
        expected_laws = {
            "ENERGY",
            "MOMENTUM",
            "ANGULAR_MOMENTUM",
            "MASS",
            "CHARGE",
            "PARTICLE_NUMBER",
            "PROBABILITY",
        }
        actual_laws = {law.name for law in ConservationLaw}
        assert actual_laws == expected_laws

    def test_enum_values(self):
        """Test enum string values match lowercase names."""
        assert ConservationLaw.ENERGY.value == "energy"
        assert ConservationLaw.MOMENTUM.value == "momentum"
        assert ConservationLaw.MASS.value == "mass"
        assert ConservationLaw.CHARGE.value == "charge"


class TestEnergyViolation:
    """Test energy conservation violation computation."""

    def test_perfect_conservation(self):
        """Test zero violation for perfect energy conservation."""
        y_pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y_true = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        violation = energy_violation(y_pred, y_true)
        assert jnp.isclose(violation, 0.0, atol=1e-6)

    def test_violation_detection(self):
        """Test that energy violations are detected."""
        y_pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y_true = jnp.array([[1.1, 2.1], [3.1, 4.1]])

        violation = energy_violation(y_pred, y_true)
        assert violation > 0.0

    def test_tolerance_threshold(self):
        """Test tolerance-based violation filtering."""
        y_pred = jnp.array([[1.0, 2.0]])
        y_true = jnp.array([[1.00001, 2.00001]])

        # Small violation below tolerance should return 0
        violation = energy_violation(y_pred, y_true, tolerance=1e-3)
        assert jnp.isclose(violation, 0.0)

    def test_tolerance_exceeded(self):
        """Test that violations exceeding tolerance are detected."""
        y_pred = jnp.array([[1.0, 2.0]])
        y_true = jnp.array([[1.1, 2.1]])

        # Large violation above tolerance should return non-zero
        violation = energy_violation(y_pred, y_true, tolerance=1e-6)
        assert violation > 0.0

    def test_monitoring_disabled(self):
        """Test monitoring enable/disable flag."""
        y_pred = jnp.array([[1.0, 2.0]])
        y_true = jnp.array([[2.0, 3.0]])

        # Disabled monitoring should return 0 regardless of violation
        violation = energy_violation(y_pred, y_true, monitoring_enabled=False)
        assert jnp.isclose(violation, 0.0)

    def test_jit_compatibility(self):
        """Test that energy_violation works with JAX JIT."""

        @jax.jit
        def compute_violation(y_pred, y_true):
            return energy_violation(y_pred, y_true)

        y_pred = jnp.array([[1.0, 2.0]])
        y_true = jnp.array([[1.5, 2.5]])

        violation = compute_violation(y_pred, y_true)
        assert jnp.isfinite(violation)


class TestMomentumViolation:
    """Test momentum conservation violation (component-wise)."""

    def test_perfect_component_conservation(self):
        """Test zero violation for perfect momentum conservation."""
        y_pred = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y_true = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        violation = momentum_violation(y_pred, y_true)
        assert jnp.isclose(violation, 0.0, atol=1e-6)

    def test_component_wise_violation(self):
        """Test detection of component-wise momentum violations."""
        # Only y-component (index 1) is violated
        y_pred = jnp.array([[1.0, 2.0, 3.0]])
        y_true = jnp.array([[1.0, 2.5, 3.0]])

        violation = momentum_violation(y_pred, y_true)
        assert violation > 0.0

    def test_all_components_violated(self):
        """Test violation when all components are violated."""
        y_pred = jnp.array([[1.0, 2.0, 3.0]])
        y_true = jnp.array([[1.5, 2.5, 3.5]])

        violation = momentum_violation(y_pred, y_true)
        assert violation > 0.0

    def test_tolerance_threshold(self):
        """Test tolerance filtering for momentum."""
        y_pred = jnp.array([[1.0, 2.0, 3.0]])
        y_true = jnp.array([[1.000001, 2.000001, 3.000001]])

        # Tiny violation below tolerance
        violation = momentum_violation(y_pred, y_true, tolerance=1e-3)
        assert jnp.isclose(violation, 0.0)

    def test_batch_processing(self):
        """Test momentum violation with batched data."""
        batch_size = 10
        y_pred = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 3))
        y_true = y_pred + 0.01  # Small perturbation

        violation = momentum_violation(y_pred, y_true)
        assert jnp.isfinite(violation)


class TestMassViolation:
    """Test mass conservation violation."""

    def test_perfect_mass_conservation(self):
        """Test zero violation for perfect mass conservation."""
        y_pred = jnp.array([[0.5, 0.5]])  # Total mass = 1.0
        target_mass = 1.0

        violation = mass_violation(y_pred, target_mass)
        assert jnp.isclose(violation, 0.0, atol=1e-6)

    def test_mass_violation_detection(self):
        """Test detection of mass violations."""
        y_pred = jnp.array([[0.5, 0.5]])  # Total mass = 1.0
        target_mass = 2.0  # Different from actual

        violation = mass_violation(y_pred, target_mass)
        assert violation > 0.0

    def test_tolerance_threshold(self):
        """Test tolerance-based filtering."""
        y_pred = jnp.array([[0.5, 0.500001]])  # Total ~1.0
        target_mass = 1.0

        violation = mass_violation(y_pred, target_mass, tolerance=1e-3)
        assert jnp.isclose(violation, 0.0)


class TestParticleNumberViolation:
    """Test particle number conservation violation."""

    def test_perfect_particle_conservation(self):
        """Test zero violation for perfect particle number conservation."""
        y_pred = jnp.array([[1.0, 1.0, 1.0]])  # 3 particles
        target_particle_number = 3.0

        violation = particle_number_violation(y_pred, target_particle_number)
        assert jnp.isclose(violation, 0.0, atol=1e-6)

    def test_particle_violation_detection(self):
        """Test detection of particle number violations."""
        y_pred = jnp.array([[1.0, 1.0, 1.0]])  # 3 particles
        target_particle_number = 5.0

        violation = particle_number_violation(y_pred, target_particle_number)
        assert violation > 0.0


class TestSymmetryViolation:
    """Test symmetry preservation violation."""

    def test_perfect_symmetry(self):
        """Test zero violation for perfectly symmetric data."""
        # Reflection symmetric
        y_pred = jnp.array([[1.0, 2.0, 2.0, 1.0]])

        violation = symmetry_violation(y_pred)
        assert jnp.isclose(violation, 0.0, atol=1e-6)

    def test_symmetry_violation_detection(self):
        """Test detection of symmetry violations."""
        # Not symmetric
        y_pred = jnp.array([[1.0, 2.0, 3.0, 4.0]])

        violation = symmetry_violation(y_pred)
        assert violation > 0.0


class TestApplyConservationConstraint:
    """Test conservation constraint application to parameters."""

    def test_energy_constraint(self):
        """Test energy conservation constraint application."""
        params = jnp.array([[1.0, 2.0, 3.0]])

        constrained = apply_conservation_constraint(
            params, ConservationLaw.ENERGY, weight=1.0
        )

        # Should normalize parameter norm
        assert constrained.shape == params.shape
        assert jnp.isfinite(constrained).all()

    def test_mass_constraint(self):
        """Test mass conservation constraint application."""
        params = jnp.array([[1.0, 2.0, 3.0]])

        constrained = apply_conservation_constraint(
            params, ConservationLaw.MASS, weight=1.0
        )

        # Should normalize total mass
        assert constrained.shape == params.shape
        assert jnp.isfinite(constrained).all()

    def test_constraint_weight(self):
        """Test constraint weighting."""
        params = jnp.array([[1.0, 2.0, 3.0]])

        # Full constraint (weight=1.0)
        constrained_full = apply_conservation_constraint(
            params, ConservationLaw.ENERGY, weight=1.0
        )

        # No constraint (weight=0.0) - should return original
        constrained_none = apply_conservation_constraint(
            params, ConservationLaw.ENERGY, weight=0.0
        )

        assert jnp.allclose(constrained_none, params)
        assert not jnp.allclose(constrained_full, params)

    def test_partial_constraint(self):
        """Test partial constraint application (0 < weight < 1)."""
        params = jnp.array([[1.0, 2.0, 3.0]])

        constrained = apply_conservation_constraint(
            params, ConservationLaw.ENERGY, weight=0.5
        )

        # Should be between original and fully constrained
        assert constrained.shape == params.shape
        assert jnp.isfinite(constrained).all()


class TestJAXCompatibility:
    """Test JAX transformation compatibility."""

    def test_vmap_energy_violation(self):
        """Test vectorization with vmap."""
        y_pred_batch = jnp.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        y_true_batch = jnp.array([[[1.1, 2.1]], [[3.1, 4.1]]])

        # Vectorize over batch dimension
        violations = jax.vmap(energy_violation)(y_pred_batch, y_true_batch)

        assert violations.shape == (2,)
        assert jnp.isfinite(violations).all()

    def test_grad_compatibility(self):
        """Test gradient computation through conservation functions."""

        def loss_fn(y_pred, y_true):
            return energy_violation(y_pred, y_true)

        y_pred = jnp.array([[1.0, 2.0]])
        y_true = jnp.array([[1.5, 2.5]])

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(y_pred, y_true)

        assert grads.shape == y_pred.shape
        assert jnp.isfinite(grads).all()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        y_pred = jnp.array([[]]).reshape(0, 2)
        y_true = jnp.array([[]]).reshape(0, 2)

        # Should not raise, may return NaN or 0
        violation = energy_violation(y_pred, y_true)
        # Accept either 0 or NaN for empty input
        assert jnp.isnan(violation) or jnp.isclose(violation, 0.0)

    def test_single_element(self):
        """Test with single element arrays."""
        y_pred = jnp.array([[1.0]])
        y_true = jnp.array([[1.0]])

        violation = energy_violation(y_pred, y_true)
        assert jnp.isclose(violation, 0.0, atol=1e-6)

    def test_negative_tolerance(self):
        """Test behavior with negative tolerance (should still work)."""
        y_pred = jnp.array([[1.0, 2.0]])
        y_true = jnp.array([[1.0, 2.0]])

        # Negative tolerance means all violations are penalized
        violation = energy_violation(y_pred, y_true, tolerance=-1.0)
        assert jnp.isfinite(violation)


class TestMultiScalePhysics:
    """Test multi-scale physics integration."""

    def test_initialization(self):
        """Test multi-scale physics initialization."""
        from opifex.core.physics.conservation import MultiScalePhysics

        scales = ["molecular", "atomic", "electronic"]
        scale_weights = {"molecular": 0.5, "atomic": 0.3, "electronic": 0.2}

        msp = MultiScalePhysics(scales, scale_weights)

        assert msp.scales == scales
        # Weights should be normalized to sum to 1.0
        assert abs(sum(msp.normalized_weights.values()) - 1.0) < 1e-6

    def test_compute_loss(self):
        """Test multi-scale loss computation."""
        from opifex.core.physics.conservation import MultiScalePhysics

        scales = ["molecular", "atomic"]
        scale_weights = {"molecular": 1.0, "atomic": 1.0}

        msp = MultiScalePhysics(scales, scale_weights)

        x = jnp.array([[1.0, 2.0]])
        y_pred = jnp.array([[1.1, 2.1]])
        y_true = jnp.array([[1.0, 2.0]])

        base_loss_fn = lambda y_pred, y_true: jnp.mean((y_pred - y_true) ** 2)
        loss = msp.compute_loss(x, y_pred, y_true, base_loss_fn)

        assert jnp.isfinite(loss)
        assert loss > 0.0


class TestAdaptiveConstraintWeighting:
    """Test adaptive constraint weighting."""

    def test_initialization(self):
        """Test adaptive weighting initialization."""
        from opifex.core.physics.conservation import AdaptiveConstraintWeighting

        constraints = ["energy", "momentum"]
        initial_weights = {"energy": 0.5, "momentum": 0.5}

        acw = AdaptiveConstraintWeighting(constraints, initial_weights)

        assert acw.constraints == constraints
        assert acw.current_weights == initial_weights

    def test_update_weights(self):
        """Test weight update based on violations."""
        from opifex.core.physics.conservation import AdaptiveConstraintWeighting

        constraints = ["energy", "momentum"]
        initial_weights = {"energy": 0.5, "momentum": 0.5}

        acw = AdaptiveConstraintWeighting(
            constraints, initial_weights, adaptation_rate=0.1
        )

        # Simulate energy having higher violation
        violations = {"energy": 1.0, "momentum": 0.1}
        new_weights = acw.update_weights(violations)

        # Energy should get higher weight
        assert new_weights["energy"] > new_weights["momentum"]
        # Weights should still sum to 1.0
        assert abs(sum(new_weights.values()) - 1.0) < 1e-6


class TestConstraintAggregator:
    """Test constraint aggregator."""

    def test_initialization(self):
        """Test constraint aggregator initialization."""
        from opifex.core.physics.conservation import ConstraintAggregator

        config = {
            "conservation_laws": ["energy", "momentum"],
            "energy_conservation_tolerance": 1e-6,
            "momentum_conservation_tolerance": 1e-5,
        }

        aggregator = ConstraintAggregator(config)

        assert aggregator.conservation_laws == ["energy", "momentum"]
        assert aggregator.energy_tolerance == 1e-6
        assert aggregator.momentum_tolerance == 1e-5

    def test_compute_constraint_loss(self):
        """Test constraint loss computation."""
        from opifex.core.physics.conservation import ConstraintAggregator

        config = {
            "conservation_laws": ["energy"],
            "energy_conservation_tolerance": 1e-6,
        }

        aggregator = ConstraintAggregator(config)

        x = jnp.array([[1.0, 2.0]])
        y_pred = jnp.array([[1.1, 2.1]])
        y_true = jnp.array([[1.0, 2.0]])

        loss = aggregator.compute_constraint_loss(x, y_pred, y_true)

        assert jnp.isfinite(loss)

    def test_compute_constraint_metrics(self):
        """Test constraint metrics computation."""
        from opifex.core.physics.conservation import ConstraintAggregator

        config = {
            "conservation_laws": ["energy", "momentum"],
        }

        aggregator = ConstraintAggregator(config)

        x = jnp.array([[1.0, 2.0]])
        y_pred = jnp.array([[1.1, 2.1]])
        y_true = jnp.array([[1.0, 2.0]])

        metrics = aggregator.compute_constraint_metrics(x, y_pred, y_true)

        assert "energy_conservation" in metrics
        assert "momentum_conservation" in metrics
        assert isinstance(metrics["energy_conservation"], float)

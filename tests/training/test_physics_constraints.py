"""Tests for physics constraints module.

This module tests the general physics-informed learning constraints
that can be applied during neural network training.
"""

import jax
import jax.numpy as jnp

from opifex.core.physics.conservation import (
    AdaptiveConstraintWeighting,
    ConstraintAggregator,
    energy_violation,
    momentum_violation,
    MultiScalePhysics,
    particle_number_violation,
    symmetry_violation,
)


class TestConservationViolations:
    """Test conservation violation computations."""

    def test_energy_conservation_below_tolerance(self):
        """Energy violations below tolerance should return 0."""
        y_pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y_true = jnp.array([[1.0 + 1e-7, 2.0], [3.0, 4.0 + 1e-7]])

        violation = energy_violation(
            y_pred, y_true, tolerance=1e-5, monitoring_enabled=True
        )

        assert float(violation) == 0.0, "Small violations should be ignored"

    def test_energy_conservation_above_tolerance(self):
        """Energy violations above tolerance should be penalized."""
        y_pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y_true = jnp.array([[2.0, 3.0], [4.0, 5.0]])  # Large difference

        violation = energy_violation(
            y_pred, y_true, tolerance=1e-5, monitoring_enabled=True
        )

        assert float(violation) > 0.0, "Large violations should be penalized"

    def test_energy_conservation_monitoring_disabled(self):
        """Monitoring disabled should return 0."""
        y_pred = jnp.array([[1.0, 2.0]])
        y_true = jnp.array([[5.0, 6.0]])  # Large difference

        violation = energy_violation(
            y_pred, y_true, tolerance=1e-5, monitoring_enabled=False
        )

        assert float(violation) == 0.0, "Disabled monitoring should return 0"

    def test_momentum_conservation_component_wise(self):
        """Momentum should be conserved component-wise, not as scalar."""
        # Create data where x-component is conserved but y-component is not
        y_pred = jnp.array([[1.0, 5.0], [-1.0, 3.0]])  # x sums to 0, y sums to 8
        y_true = jnp.array([[1.0, 1.0], [-1.0, 1.0]])  # x sums to 0, y sums to 2

        violation = momentum_violation(y_pred, y_true, tolerance=1e-5)

        # Should detect y-component violation (8 vs 2 = difference of 6)
        assert float(violation) > 0.0, "Y-component violation should be detected"

    def test_momentum_conservation_all_components_conserved(self):
        """All momentum components conserved should give near-zero violation."""
        y_pred = jnp.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])  # Sums to [0, 0, 0]
        y_true = jnp.array([[0.5, 1.0, 1.5], [-0.5, -1.0, -1.5]])  # Sums to [0, 0, 0]

        violation = momentum_violation(y_pred, y_true, tolerance=1e-5)

        assert float(violation) == 0.0, "Conserved momentum should have no violation"

    def test_particle_conservation_below_tolerance(self):
        """Particle number close to target should return 0."""
        # Each row sums to 2.0, so mean particle count is 2.0
        y_pred = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        target_particles = 2.0  # Matches mean particle count per row

        violation = particle_number_violation(y_pred, target_particles, tolerance=1e-4)

        assert float(violation) == 0.0, "Exact match should give zero violation"

    def test_particle_conservation_above_tolerance(self):
        """Particle number far from target should be penalized."""
        y_pred = jnp.array([[1.0, 1.0], [1.0, 1.0]])  # Total: 4.0
        target_particles = 10.0  # Very different

        violation = particle_number_violation(y_pred, target_particles, tolerance=1e-4)

        assert float(violation) > 0.0, "Large particle violations should be penalized"

    def test_symmetry_preservation_symmetric_field(self):
        """Symmetric field should have zero violation."""
        # Create perfectly symmetric field: [1, 2, 3, 2, 1]
        y_pred = jnp.array([[1.0, 2.0, 3.0, 2.0, 1.0]])

        violation = symmetry_violation(y_pred, tolerance=1e-6)

        assert float(violation) == 0.0, "Symmetric field should have no violation"

    def test_symmetry_preservation_asymmetric_field(self):
        """Asymmetric field should have positive violation."""
        # Create asymmetric field: [1, 2, 3, 4, 5]
        y_pred = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        violation = symmetry_violation(y_pred, tolerance=1e-6)

        assert float(violation) > 0.0, "Asymmetric field should have violation"

    def test_all_violations_are_jax_compatible(self):
        """All violation functions should be JIT-compatible."""
        y_pred = jnp.array([[1.0, 2.0]])
        y_true = jnp.array([[1.1, 2.1]])

        # These should all compile with JIT
        jitted_energy = jax.jit(energy_violation)
        jitted_momentum = jax.jit(momentum_violation)
        jitted_particle = jax.jit(particle_number_violation)
        jitted_symmetry = jax.jit(symmetry_violation)

        # Should execute without error
        _ = jitted_energy(y_pred, y_true, 1e-5, True)
        _ = jitted_momentum(y_pred, y_true, 1e-5)
        _ = jitted_particle(y_pred, 10.0, 1e-4)
        _ = jitted_symmetry(y_pred, 1e-6)


class TestMultiScalePhysics:
    """Test multi-scale physics integration."""

    def test_single_scale_loss(self):
        """Single scale should apply appropriate transformation."""
        x = jnp.array([[1.0, 2.0]])
        y_pred = jnp.array([[3.0, 4.0]])
        y_true = jnp.array([[3.5, 4.5]])
        base_loss_fn = lambda pred, true: jnp.mean((pred - true) ** 2)

        multi_scale = MultiScalePhysics(
            scales=["molecular"], scale_weights={"molecular": 1.0}
        )

        loss = multi_scale.compute_loss(x, y_pred, y_true, base_loss_fn)

        assert loss.shape == (), "Loss should be scalar"
        assert float(loss) > 0.0, "Loss should be positive"

    def test_multi_scale_composition(self):
        """Multiple scales should be weighted and combined."""
        x = jnp.array([[1.0, 2.0]])
        y_pred = jnp.array([[3.0, 4.0]])
        y_true = jnp.array([[3.5, 4.5]])
        base_loss_fn = lambda pred, true: jnp.mean((pred - true) ** 2)

        multi_scale = MultiScalePhysics(
            scales=["molecular", "atomic", "electronic"],
            scale_weights={"molecular": 0.5, "atomic": 0.3, "electronic": 0.2},
        )

        loss = multi_scale.compute_loss(x, y_pred, y_true, base_loss_fn)

        assert loss.shape == (), "Loss should be scalar"
        assert float(loss) > 0.0, "Combined loss should be positive"

    def test_scale_weights_normalization(self):
        """Scale weights should be normalized to sum to 1."""
        multi_scale = MultiScalePhysics(
            scales=["molecular", "atomic"],
            scale_weights={"molecular": 2.0, "atomic": 2.0},  # Sum to 4.0
        )

        weights = multi_scale.get_normalized_weights()

        assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights should sum to 1"

    def test_molecular_scale_uses_mse(self):
        """Molecular scale should use MSE loss."""
        x = jnp.array([[1.0]])
        y_pred = jnp.array([[2.0]])
        y_true = jnp.array([[1.0]])
        base_loss_fn = lambda pred, true: jnp.mean((pred - true) ** 2)

        multi_scale = MultiScalePhysics(
            scales=["molecular"], scale_weights={"molecular": 1.0}
        )

        loss = multi_scale.compute_loss(x, y_pred, y_true, base_loss_fn)

        # Molecular scale uses MSE * 0.5
        expected_loss = 0.5 * ((2.0 - 1.0) ** 2)
        assert abs(float(loss) - expected_loss) < 1e-6


class TestAdaptiveConstraintWeighting:
    """Test adaptive constraint weight adjustment."""

    def test_weight_adaptation_increases_for_high_violations(self):
        """Constraints with high violations should get higher weights."""
        weighting = AdaptiveConstraintWeighting(
            constraints=["energy_conservation", "momentum_conservation"],
            initial_weights={"energy_conservation": 0.5, "momentum_conservation": 0.5},
            adaptation_rate=0.1,
        )

        # Energy has high violation, momentum has low
        violations = {"energy_conservation": 10.0, "momentum_conservation": 1.0}

        new_weights = weighting.update_weights(violations)

        assert (
            new_weights["energy_conservation"] > new_weights["momentum_conservation"]
        ), "Energy should have higher weight due to higher violation"

    def test_weights_sum_to_one_after_adaptation(self):
        """Weights should be normalized to sum to 1.0."""
        weighting = AdaptiveConstraintWeighting(
            constraints=[
                "energy_conservation",
                "momentum_conservation",
                "particle_conservation",
            ],
            initial_weights={
                "energy_conservation": 0.33,
                "momentum_conservation": 0.33,
                "particle_conservation": 0.34,
            },
            adaptation_rate=0.2,
        )

        violations = {
            "energy_conservation": 5.0,
            "momentum_conservation": 10.0,
            "particle_conservation": 2.0,
        }

        new_weights = weighting.update_weights(violations)

        total_weight = sum(new_weights.values())
        assert abs(total_weight - 1.0) < 1e-6, (
            f"Weights should sum to 1.0, got {total_weight}"
        )

    def test_zero_violations_maintain_weights(self):
        """Zero violations should maintain current weights."""
        weighting = AdaptiveConstraintWeighting(
            constraints=["energy_conservation", "momentum_conservation"],
            initial_weights={"energy_conservation": 0.6, "momentum_conservation": 0.4},
            adaptation_rate=0.1,
        )

        violations = {"energy_conservation": 0.0, "momentum_conservation": 0.0}

        new_weights = weighting.update_weights(violations)

        # Weights should remain roughly the same
        assert abs(new_weights["energy_conservation"] - 0.6) < 0.1
        assert abs(new_weights["momentum_conservation"] - 0.4) < 0.1


class TestConstraintAggregator:
    """Test constraint aggregation and composition."""

    def test_single_conservation_law(self):
        """Single conservation law should be computed correctly."""
        config = {
            "conservation_laws": ["energy"],
            "energy_conservation_tolerance": 1e-5,
            "energy_conservation_monitoring": True,
        }

        aggregator = ConstraintAggregator(config)

        x = jnp.array([[1.0, 2.0]])
        y_pred = jnp.array([[3.0, 4.0]])
        y_true = jnp.array([[3.1, 4.1]])

        loss = aggregator.compute_constraint_loss(x, y_pred, y_true)

        assert loss.shape == (), "Loss should be scalar"
        assert float(loss) >= 0.0, "Loss should be non-negative"

    def test_multiple_conservation_laws(self):
        """Multiple conservation laws should be combined."""
        config = {
            "conservation_laws": ["energy", "momentum", "particle_number"],
            "energy_conservation_tolerance": 1e-5,
            "momentum_conservation_tolerance": 1e-5,
            "particle_conservation_tolerance": 1e-4,
            "target_particle_number": 10.0,
        }

        aggregator = ConstraintAggregator(config)

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = jnp.array([[1.5, 2.5], [3.5, 4.5]])
        y_true = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        loss = aggregator.compute_constraint_loss(x, y_pred, y_true)

        assert loss.shape == (), "Loss should be scalar"
        assert float(loss) >= 0.0, "Combined loss should be non-negative"

    def test_constraint_metrics_collection(self):
        """Metrics should be collected for each constraint."""
        config = {
            "conservation_laws": ["energy", "momentum"],
            "energy_conservation_tolerance": 1e-5,
            "momentum_conservation_tolerance": 1e-5,
        }

        aggregator = ConstraintAggregator(config)

        x = jnp.array([[1.0, 2.0]])
        y_pred = jnp.array([[1.5, 2.5]])
        y_true = jnp.array([[1.0, 2.0]])

        metrics = aggregator.compute_constraint_metrics(x, y_pred, y_true)

        assert "energy_conservation" in metrics
        assert "momentum_conservation" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_adaptive_weighting_integration(self):
        """Adaptive weighting should work with constraint aggregator."""
        config = {
            "conservation_laws": ["energy", "momentum"],
            "adaptive_weighting": True,
            "adaptation_rate": 0.1,
            "energy_conservation_tolerance": 1e-5,
            "momentum_conservation_tolerance": 1e-5,
        }

        aggregator = ConstraintAggregator(config)

        x = jnp.array([[1.0, 2.0]])
        y_pred = jnp.array([[3.0, 4.0]])  # Large violations
        y_true = jnp.array([[1.0, 2.0]])

        # First call to establish baseline
        loss1 = aggregator.compute_constraint_loss(x, y_pred, y_true)

        # Get metrics to trigger weight adaptation
        aggregator.compute_constraint_metrics(x, y_pred, y_true)

        # Second call should use adapted weights
        loss2 = aggregator.compute_constraint_loss(x, y_pred, y_true)

        assert hasattr(aggregator, "weight_manager"), "Should have weight manager"
        # Both losses should be valid
        assert float(loss1) >= 0.0
        assert float(loss2) >= 0.0

    def test_no_conservation_laws_returns_zero(self):
        """No conservation laws configured should return zero loss."""
        config = {"conservation_laws": []}

        aggregator = ConstraintAggregator(config)

        x = jnp.array([[1.0, 2.0]])
        y_pred = jnp.array([[3.0, 4.0]])
        y_true = jnp.array([[1.0, 2.0]])

        loss = aggregator.compute_constraint_loss(x, y_pred, y_true)

        assert float(loss) == 0.0, "No constraints should give zero loss"

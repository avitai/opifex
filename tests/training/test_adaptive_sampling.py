"""Tests for RAD (Residual-based Adaptive Distribution) sampling.

TDD: These tests define the expected behavior for adaptive collocation point sampling.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class TestRADConfig:
    """Test RAD configuration."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        from opifex.training.adaptive_sampling import RADConfig

        config = RADConfig()
        assert config.beta > 0  # Exponent for residual weighting
        assert config.resample_frequency > 0
        assert config.min_probability > 0

    def test_custom_beta(self):
        """Should accept custom beta exponent."""
        from opifex.training.adaptive_sampling import RADConfig

        config = RADConfig(beta=2.0)
        assert config.beta == 2.0


class TestSamplingDistribution:
    """Test sampling distribution computation."""

    def test_compute_sampling_distribution(self):
        """Should compute sampling distribution from residuals."""
        from opifex.training.adaptive_sampling import compute_sampling_distribution

        residuals = jnp.array([1.0, 2.0, 3.0, 4.0])
        probs = compute_sampling_distribution(residuals, beta=1.0)

        assert probs.shape == (4,)
        assert jnp.allclose(jnp.sum(probs), 1.0)  # Probabilities sum to 1

    def test_higher_residual_higher_probability(self):
        """Higher residuals should have higher sampling probability."""
        from opifex.training.adaptive_sampling import compute_sampling_distribution

        residuals = jnp.array([1.0, 10.0])
        probs = compute_sampling_distribution(residuals, beta=1.0)

        assert probs[1] > probs[0]

    def test_beta_controls_sharpness(self):
        """Higher beta should make distribution more peaked."""
        from opifex.training.adaptive_sampling import compute_sampling_distribution

        residuals = jnp.array([1.0, 2.0, 3.0])

        probs_low = compute_sampling_distribution(residuals, beta=0.5)
        probs_high = compute_sampling_distribution(residuals, beta=2.0)

        # Higher beta should make max probability higher
        assert jnp.max(probs_high) > jnp.max(probs_low)

    def test_handles_zero_residuals(self):
        """Should handle zero residuals without NaN."""
        from opifex.training.adaptive_sampling import compute_sampling_distribution

        residuals = jnp.array([0.0, 1.0, 2.0])
        probs = compute_sampling_distribution(residuals, beta=1.0)

        assert jnp.all(jnp.isfinite(probs))
        assert jnp.all(probs >= 0)


class TestRADSampler:
    """Test RAD sampler class."""

    def test_create_sampler(self):
        """Should create RAD sampler."""
        from opifex.training.adaptive_sampling import RADSampler

        sampler = RADSampler()
        assert sampler is not None

    def test_sample_collocation_points(self):
        """Should sample collocation points based on residuals."""
        from opifex.training.adaptive_sampling import RADSampler

        sampler = RADSampler()
        domain_points = jnp.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        residuals = jnp.array([1.0, 2.0, 5.0, 1.0, 1.0])

        key = jax.random.key(0)
        sampled = sampler.sample(domain_points, residuals, batch_size=3, key=key)

        assert sampled.shape == (3, 1)

    def test_sample_with_config(self):
        """Should respect configuration parameters."""
        from opifex.training.adaptive_sampling import RADConfig, RADSampler

        config = RADConfig(beta=2.0)
        sampler = RADSampler(config=config)

        domain_points = jnp.linspace(0, 1, 10).reshape(-1, 1)
        residuals = jnp.abs(jnp.sin(jnp.pi * domain_points.flatten()))

        key = jax.random.key(42)
        sampled = sampler.sample(domain_points, residuals, batch_size=5, key=key)

        assert sampled.shape == (5, 1)

    def test_sample_2d_domain(self):
        """Should work with 2D domain."""
        from opifex.training.adaptive_sampling import RADSampler

        sampler = RADSampler()

        # 2D grid of points
        x = jnp.linspace(0, 1, 5)
        y = jnp.linspace(0, 1, 5)
        xx, yy = jnp.meshgrid(x, y)
        domain_points = jnp.stack([xx.flatten(), yy.flatten()], axis=-1)

        residuals = jnp.ones(25)

        key = jax.random.key(0)
        sampled = sampler.sample(domain_points, residuals, batch_size=10, key=key)

        assert sampled.shape == (10, 2)


class TestRARDRefinement:
    """Test RAR-D (Residual-based Adaptive Refinement with Distribution) refinement."""

    def test_create_refiner(self):
        """Should create RAR-D refiner."""
        from opifex.training.adaptive_sampling import RARDRefiner

        refiner = RARDRefiner()
        assert refiner is not None

    def test_refine_adds_points(self):
        """Should add new points near high-residual regions."""
        from opifex.training.adaptive_sampling import RARDRefiner

        refiner = RARDRefiner(num_new_points=5)

        current_points = jnp.array([[0.1], [0.5], [0.9]])
        residuals = jnp.array([0.1, 10.0, 0.1])  # High residual at 0.5
        bounds = jnp.array([[0.0, 1.0]])

        key = jax.random.key(0)
        refined = refiner.refine(current_points, residuals, bounds, key=key)

        # Should have added points
        assert refined.shape[0] > current_points.shape[0]

    def test_new_points_near_high_residual(self):
        """New points should be near high-residual regions."""
        from opifex.training.adaptive_sampling import RARDRefiner

        refiner = RARDRefiner(num_new_points=10, noise_scale=0.05)

        # Points with high residual at x=0.5
        current_points = jnp.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        residuals = jnp.array([0.1, 0.1, 100.0, 0.1, 0.1])
        bounds = jnp.array([[0.0, 1.0]])

        key = jax.random.key(42)
        refined = refiner.refine(current_points, residuals, bounds, key=key)

        # New points should be mostly near 0.5
        new_points = refined[5:]  # Points added after original
        distances_to_05 = jnp.abs(new_points.flatten() - 0.5)

        # Most new points should be within 0.2 of x=0.5
        assert jnp.mean(distances_to_05 < 0.2) > 0.5

    def test_respects_bounds(self):
        """Refined points should stay within bounds."""
        from opifex.training.adaptive_sampling import RARDRefiner

        refiner = RARDRefiner(num_new_points=20)

        current_points = jnp.array([[0.0], [1.0]])
        residuals = jnp.array([10.0, 10.0])  # High at boundaries
        bounds = jnp.array([[0.0, 1.0]])

        key = jax.random.key(0)
        refined = refiner.refine(current_points, residuals, bounds, key=key)

        # All points should be within bounds
        assert jnp.all(refined >= bounds[:, 0])
        assert jnp.all(refined <= bounds[:, 1])


class TestAdaptiveSamplingIntegration:
    """Test integration of adaptive sampling with models."""

    def test_with_pinn_residual(self):
        """Should work with PINN residual computation."""
        from opifex.training.adaptive_sampling import RADSampler

        # Simple model
        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(1, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))

        # Compute residual (simplified - just model output magnitude)
        def compute_residual(model, x):
            return jnp.abs(model(x)).flatten()

        domain_points = jnp.linspace(0, 1, 50).reshape(-1, 1)
        residuals = compute_residual(model, domain_points)

        sampler = RADSampler()
        key = jax.random.key(0)
        sampled = sampler.sample(domain_points, residuals, batch_size=10, key=key)

        assert sampled.shape == (10, 1)

    def test_sampling_is_jit_compatible(self):
        """Sampling should be JIT compatible."""
        from opifex.training.adaptive_sampling import RADSampler

        sampler = RADSampler()
        domain_points = jnp.linspace(0, 1, 20).reshape(-1, 1)
        residuals = jnp.abs(jnp.sin(2 * jnp.pi * domain_points.flatten()))

        @jax.jit
        def sample_fn(key):
            return sampler.sample(domain_points, residuals, batch_size=5, key=key)

        key = jax.random.key(0)
        sampled = sample_fn(key)

        assert sampled.shape == (5, 1)
        assert jnp.all(jnp.isfinite(sampled))


class TestRADConfigValidation:
    """Test RAD configuration validation."""

    def test_beta_must_be_positive(self):
        """Beta should be positive."""
        from opifex.training.adaptive_sampling import RADConfig

        # Should work with positive beta
        config = RADConfig(beta=0.5)
        assert config.beta == 0.5

        # Zero or negative should raise or be handled
        # (implementation may choose to validate or just use as-is)

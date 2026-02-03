"""Tests for GradNorm multi-task loss balancing.

TDD: These tests define the expected behavior for GradNorm loss weight adaptation.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class TestGradNormConfig:
    """Test GradNorm configuration."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        from opifex.core.physics.gradnorm import GradNormConfig

        config = GradNormConfig()
        assert config.alpha >= 0  # Asymmetry parameter
        assert config.learning_rate > 0
        assert config.min_weight > 0

    def test_custom_alpha(self):
        """Should accept custom asymmetry parameter."""
        from opifex.core.physics.gradnorm import GradNormConfig

        config = GradNormConfig(alpha=1.5)
        assert config.alpha == 1.5


class TestGradNormBalancer:
    """Test GradNorm balancer class."""

    def test_create_balancer(self):
        """Should create GradNorm balancer."""
        from opifex.core.physics.gradnorm import GradNormBalancer

        balancer = GradNormBalancer(num_losses=3, rngs=nnx.Rngs(0))
        assert balancer is not None
        assert len(balancer.weights) == 3

    def test_initial_weights_equal(self):
        """Initial weights should be equal."""
        from opifex.core.physics.gradnorm import GradNormBalancer

        balancer = GradNormBalancer(num_losses=4, rngs=nnx.Rngs(0))
        weights = balancer.weights

        # All weights should be equal initially
        assert jnp.allclose(weights, weights[0])

    def test_weights_positive(self):
        """Weights should always be positive."""
        from opifex.core.physics.gradnorm import GradNormBalancer

        balancer = GradNormBalancer(num_losses=3, rngs=nnx.Rngs(0))
        weights = balancer.weights

        assert jnp.all(weights > 0)


class TestGradNormLossComputation:
    """Test GradNorm loss computation."""

    def test_compute_weighted_loss(self):
        """Should compute weighted sum of losses."""
        from opifex.core.physics.gradnorm import GradNormBalancer

        balancer = GradNormBalancer(num_losses=2, rngs=nnx.Rngs(0))

        losses = jnp.array([1.0, 2.0])
        weighted_loss = balancer.compute_weighted_loss(losses)

        assert jnp.isfinite(weighted_loss)
        assert weighted_loss > 0

    def test_compute_gradnorm_loss(self):
        """Should compute GradNorm balancing loss."""
        from opifex.core.physics.gradnorm import GradNormBalancer

        balancer = GradNormBalancer(num_losses=2, rngs=nnx.Rngs(0))

        # Mock gradient norms and losses
        grad_norms = jnp.array([1.0, 2.0])
        losses = jnp.array([0.5, 0.25])
        initial_losses = jnp.array([1.0, 1.0])

        gradnorm_loss = balancer.compute_gradnorm_loss(
            grad_norms, losses, initial_losses
        )

        assert jnp.isfinite(gradnorm_loss)
        assert gradnorm_loss >= 0


class TestGradNormUpdate:
    """Test GradNorm weight updates."""

    def test_update_weights(self):
        """Should update weights based on gradient magnitudes."""
        from opifex.core.physics.gradnorm import GradNormBalancer, GradNormConfig

        config = GradNormConfig(learning_rate=0.1)
        balancer = GradNormBalancer(num_losses=2, config=config, rngs=nnx.Rngs(0))

        initial_weights = balancer.weights.copy()

        # Simulate gradient info
        grad_norms = jnp.array([1.0, 10.0])  # Second loss has larger gradients
        losses = jnp.array([0.5, 0.5])
        initial_losses = jnp.array([1.0, 1.0])

        balancer.update_weights(grad_norms, losses, initial_losses)

        # Weights should have changed
        new_weights = balancer.weights
        assert not jnp.allclose(new_weights, initial_weights)

    def test_weight_constraints(self):
        """Weights should stay within bounds after update."""
        from opifex.core.physics.gradnorm import GradNormBalancer, GradNormConfig

        config = GradNormConfig(min_weight=0.1, max_weight=10.0)
        balancer = GradNormBalancer(num_losses=2, config=config, rngs=nnx.Rngs(0))

        # Extreme gradient imbalance
        grad_norms = jnp.array([0.001, 1000.0])
        losses = jnp.array([0.1, 0.1])
        initial_losses = jnp.array([1.0, 1.0])

        for _ in range(100):
            balancer.update_weights(grad_norms, losses, initial_losses)

        weights = balancer.weights
        assert jnp.all(weights >= config.min_weight)
        assert jnp.all(weights <= config.max_weight)


class TestGradNormIntegration:
    """Test GradNorm integration with models."""

    def test_with_simple_model(self):
        """Should integrate with NNX model training."""
        from opifex.core.physics.gradnorm import GradNormBalancer

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        balancer = GradNormBalancer(num_losses=2, rngs=nnx.Rngs(1))

        x = jnp.array([[1.0, 2.0]])
        y_target = jnp.array([[1.0]])

        def loss_fn(model):
            y = model(x)
            data_loss = jnp.mean((y - y_target) ** 2)
            reg_loss = jnp.sum(
                jnp.array([jnp.sum(p**2) for p in jax.tree.leaves(model)])
            )
            return jnp.array([data_loss, reg_loss])

        losses = loss_fn(model)
        weighted = balancer.compute_weighted_loss(losses)

        assert jnp.isfinite(weighted)

    def test_gradient_norm_computation(self):
        """Should compute gradient norms correctly."""
        from opifex.core.physics.gradnorm import compute_gradient_norms

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        x = jnp.array([[1.0, 2.0]])

        def loss1(model):
            return jnp.mean(model(x) ** 2)

        def loss2(model):
            return jnp.mean(model(x) ** 4)

        grad_norms = compute_gradient_norms(model, [loss1, loss2])

        assert len(grad_norms) == 2
        assert jnp.all(jnp.isfinite(grad_norms))
        assert jnp.all(grad_norms >= 0)


class TestGradNormTrainingLoop:
    """Test GradNorm in training loop context."""

    def test_training_step(self):
        """Should work in training step."""
        from opifex.core.physics.gradnorm import GradNormBalancer, GradNormConfig

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(1, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        config = GradNormConfig(learning_rate=0.01)
        balancer = GradNormBalancer(num_losses=2, config=config, rngs=nnx.Rngs(1))

        x = jnp.array([[1.0], [2.0]])
        y = jnp.array([[2.0], [4.0]])

        # Track losses over iterations
        initial_losses = None
        for _ in range(5):
            pred = model(x)
            data_loss = jnp.mean((pred - y) ** 2)
            reg_loss = jnp.mean(pred**2)  # Simple regularization
            losses = jnp.array([data_loss, reg_loss])

            if initial_losses is None:
                initial_losses = losses

            weighted_loss = balancer.compute_weighted_loss(losses)
            assert jnp.isfinite(weighted_loss)


class TestInverseTrainingRate:
    """Test inverse training rate computation."""

    def test_compute_inverse_rates(self):
        """Should compute relative inverse training rates."""
        from opifex.core.physics.gradnorm import compute_inverse_training_rates

        current_losses = jnp.array([0.5, 0.25, 0.1])
        initial_losses = jnp.array([1.0, 1.0, 1.0])

        rates = compute_inverse_training_rates(current_losses, initial_losses)

        assert len(rates) == 3
        assert jnp.all(jnp.isfinite(rates))

    def test_rates_normalized(self):
        """Relative rates should have mean 1."""
        from opifex.core.physics.gradnorm import compute_inverse_training_rates

        current_losses = jnp.array([0.5, 0.25])
        initial_losses = jnp.array([1.0, 1.0])

        rates = compute_inverse_training_rates(current_losses, initial_losses)

        # The mean of relative rates should be approximately 1
        assert jnp.isclose(jnp.mean(rates), 1.0, atol=1e-5)

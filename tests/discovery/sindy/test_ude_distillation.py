"""Tests for UDE → SINDy distillation pipeline.

TDD: Distill a trained UDE's neural residual into symbolic equations.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.discovery.sindy.config import SINDyConfig
from opifex.discovery.sindy.ude_distillation import distill_ude_residual


class SimpleResidualNet(nnx.Module):
    """Mimics a trained UDE neural residual that learned: f(x,y) = [0, -xy]."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        # We'll override __call__ to return a known function,
        # simulating what a trained network would output.
        self._dummy = nnx.Linear(2, 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Return the 'learned' residual: [0, -x0*x1]."""
        return jnp.column_stack(
            [
                jnp.zeros(x.shape[0]),
                -x[:, 0] * x[:, 1],
            ]
        )


class TestDistillUDEResidual:
    """Tests for distill_ude_residual."""

    def test_recovers_missing_terms(self):
        """Distillation recovers the symbolic form of the neural residual."""
        # Create a mock trained neural residual that computes [0, -xy]
        net = SimpleResidualNet(rngs=nnx.Rngs(0))

        # Generate evaluation data
        key = jax.random.PRNGKey(42)
        x_eval = jax.random.normal(key, (500, 2)) * 3.0

        config = SINDyConfig(polynomial_degree=2, threshold=0.1)
        result = distill_ude_residual(net, x_eval, config=config)

        # Should recover the xy term in the second equation
        assert result.coefficients is not None
        assert result.coefficients.shape[0] == 2  # 2 output dims

        # Check R² is high (clean data from known function)
        r2 = result.score(x_eval, net(x_eval))
        assert float(r2) > 0.99

    def test_returns_sindy_model(self):
        """Distillation returns a fitted SINDy model."""
        from opifex.discovery.sindy.sindy import SINDy

        net = SimpleResidualNet(rngs=nnx.Rngs(0))
        x_eval = jax.random.normal(jax.random.PRNGKey(0), (100, 2))

        result = distill_ude_residual(net, x_eval)
        assert isinstance(result, SINDy)
        assert result.coefficients is not None

    def test_equations_are_readable(self):
        """Distilled model produces readable equations."""
        net = SimpleResidualNet(rngs=nnx.Rngs(0))
        x_eval = jax.random.normal(jax.random.PRNGKey(0), (200, 2))

        result = distill_ude_residual(net, x_eval)
        eqs = result.equations(["x", "y"])
        assert len(eqs) == 2
        assert all(isinstance(eq, str) for eq in eqs)

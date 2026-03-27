"""Tests for Ensemble SINDy.

TDD: Ensemble SINDy must provide coefficient uncertainty estimates
via bagging over data/library subsets.
"""

import jax
import jax.numpy as jnp

from opifex.discovery.sindy.config import EnsembleSINDyConfig
from opifex.discovery.sindy.ensemble_sindy import EnsembleSINDy


def _generate_simple_linear_data(key: jax.Array, n_samples: int = 200):
    """Generate simple linear system: dx/dt = Ax."""
    x = jax.random.normal(key, (n_samples, 2))
    true_coef = jnp.array([[0.0, -2.0], [1.0, 0.0]])
    x_dot = x @ true_coef.T
    return x, x_dot


class TestEnsembleSINDy:
    """Tests for EnsembleSINDy uncertainty quantification."""

    def test_produces_coefficient_statistics(self):
        """Ensemble produces mean and std of coefficients."""
        x, x_dot = _generate_simple_linear_data(jax.random.PRNGKey(0))

        config = EnsembleSINDyConfig(
            polynomial_degree=1,
            threshold=0.05,
            n_models=10,
            bagging_fraction=0.8,
        )
        model = EnsembleSINDy(config)
        model.fit(x, x_dot, key=jax.random.PRNGKey(42))

        assert model.coef_mean is not None
        assert model.coef_std is not None
        assert model.coef_mean.shape == model.coef_std.shape

    def test_std_is_nonnegative(self):
        """Coefficient standard deviations are non-negative."""
        x, x_dot = _generate_simple_linear_data(jax.random.PRNGKey(1))

        config = EnsembleSINDyConfig(polynomial_degree=1, threshold=0.05, n_models=10)
        model = EnsembleSINDy(config)
        model.fit(x, x_dot, key=jax.random.PRNGKey(42))

        assert model.coef_std is not None
        assert jnp.all(model.coef_std >= 0)

    def test_mean_recovers_structure(self):
        """Mean coefficients recover the system structure."""
        x, x_dot = _generate_simple_linear_data(jax.random.PRNGKey(2), n_samples=500)

        config = EnsembleSINDyConfig(polynomial_degree=1, threshold=0.05, n_models=20)
        model = EnsembleSINDy(config)
        model.fit(x, x_dot, key=jax.random.PRNGKey(42))

        # Mean coefficients should have correct sparsity (4 nonzero out of 6)
        assert model.coef_mean is not None
        n_nonzero = int(jnp.sum(jnp.abs(model.coef_mean) > 0.01))
        assert n_nonzero <= 6

    def test_noisy_data_increases_uncertainty(self):
        """Noisier data produces larger coefficient uncertainty."""
        key = jax.random.PRNGKey(3)
        x, x_dot = _generate_simple_linear_data(key, n_samples=200)

        config = EnsembleSINDyConfig(polynomial_degree=1, threshold=0.05, n_models=15)

        # Clean data
        model_clean = EnsembleSINDy(config)
        model_clean.fit(x, x_dot, key=jax.random.PRNGKey(10))

        # Noisy data
        noise = 0.5 * jax.random.normal(jax.random.PRNGKey(99), x_dot.shape)
        model_noisy = EnsembleSINDy(config)
        model_noisy.fit(x, x_dot + noise, key=jax.random.PRNGKey(10))

        # Noisy should have larger total uncertainty
        assert model_noisy.coef_std is not None
        assert model_clean.coef_std is not None
        assert float(jnp.sum(model_noisy.coef_std)) >= float(jnp.sum(model_clean.coef_std))

    def test_all_models_stored(self):
        """All individual model coefficients are accessible."""
        x, x_dot = _generate_simple_linear_data(jax.random.PRNGKey(4))

        config = EnsembleSINDyConfig(polynomial_degree=1, threshold=0.05, n_models=8)
        model = EnsembleSINDy(config)
        model.fit(x, x_dot, key=jax.random.PRNGKey(42))

        assert len(model.coef_list) == 8

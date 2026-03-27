"""Tests for SINDy sparse regression optimizers.

TDD: These tests define expected behavior for STLSQ and SR3.
"""

import jax.numpy as jnp

from opifex.discovery.sindy.optimizers import SR3, STLSQ


class TestSTLSQ:
    """Tests for Sequential Thresholded Least Squares."""

    def test_recovers_sparse_system(self):
        """STLSQ recovers known sparse coefficients from clean data."""
        # Ground truth: y = 2*x0 + 0*x1 + 3*x2 (sparse: x1 coef is 0)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (100, 3))
        true_coef = jnp.array([2.0, 0.0, 3.0])
        y = x @ true_coef

        opt = STLSQ(threshold=0.1)
        coef = opt.fit(x, y[:, None])

        assert coef.shape == (1, 3)
        assert jnp.allclose(coef[0], true_coef, atol=0.1)
        # The zero coefficient should be exactly zero after thresholding
        assert float(jnp.abs(coef[0, 1])) < 0.1

    def test_multi_target(self):
        """STLSQ handles multiple target variables."""
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (100, 3))
        true_coef = jnp.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        y = x @ true_coef.T

        opt = STLSQ(threshold=0.1)
        coef = opt.fit(x, y)

        assert coef.shape == (2, 3)

    def test_threshold_controls_sparsity(self):
        """Higher threshold produces sparser coefficients."""
        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (100, 5))
        true_coef = jnp.array([3.0, 0.5, 0.0, 0.0, 2.0])
        y = x @ true_coef

        coef_low = STLSQ(threshold=0.1).fit(x, y[:, None])
        coef_high = STLSQ(threshold=1.0).fit(x, y[:, None])

        nnz_low = int(jnp.sum(jnp.abs(coef_low) > 0))
        nnz_high = int(jnp.sum(jnp.abs(coef_high) > 0))
        assert nnz_high <= nnz_low

    def test_returns_coefficients_array(self):
        """Fit returns a JAX array of coefficients."""
        x = jnp.ones((10, 2))
        y = jnp.ones((10, 1))
        coef = STLSQ().fit(x, y)
        assert isinstance(coef, jnp.ndarray)


class TestSR3:
    """Tests for Sparse Relaxed Regularized Regression."""

    def test_recovers_sparse_system(self):
        """SR3 recovers known sparse coefficients."""
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (100, 3))
        true_coef = jnp.array([2.0, 0.0, 3.0])
        y = x @ true_coef

        opt = SR3(threshold=0.1)
        coef = opt.fit(x, y[:, None])

        assert coef.shape == (1, 3)
        # Should recover approximately correct coefficients
        assert jnp.allclose(coef[0, 0], 2.0, atol=0.3)
        assert jnp.allclose(coef[0, 2], 3.0, atol=0.3)

    def test_l1_regularization(self):
        """SR3 with L1 penalty produces sparse solutions."""
        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (100, 5))
        true_coef = jnp.array([3.0, 0.0, 0.0, 0.0, 2.0])
        y = x @ true_coef

        opt = SR3(threshold=0.5, regularization="l1")
        coef = opt.fit(x, y[:, None])

        # Should have mostly zeros
        n_nonzero = int(jnp.sum(jnp.abs(coef) > 0.01))
        assert n_nonzero <= 3


import jax

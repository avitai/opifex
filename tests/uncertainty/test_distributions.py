"""Tests for correlated predictive differences and linear contrasts.

Verifies :func:`opifex.uncertainty.distributions.correlated_difference_predictive`,
:func:`opifex.uncertainty.distributions.linear_contrast_predictive`, and zero-safe
standard deviation gradients on :meth:`PredictiveDistribution.std`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from substrax.testing import TraceCounter

from opifex.uncertainty import (
    correlated_difference_predictive,
    linear_contrast_predictive,
    PredictiveDistribution,
)


_EPS32 = float(np.finfo(np.float32).eps)


class TestCorrelatedDifferencePredictive:
    """Pairwise predictive difference ``Y_a - Y_b`` with cross-covariance."""

    def test_computes_exact_mean_and_variance_with_cross_covariance(self) -> None:
        mean_a = jnp.array([7.5, 6.0, 5.0])
        mean_b = jnp.array([6.0, 6.0, 8.0])
        var_a = jnp.array([0.5, 1.0, 0.25])
        var_b = jnp.array([0.5, 1.0, 0.25])
        cov_ab = jnp.array([0.4, 1.0, -0.2])

        diff = correlated_difference_predictive(
            mean_a, mean_b, var_a=var_a, var_b=var_b, cov_ab=cov_ab
        )
        assert isinstance(diff, PredictiveDistribution)
        assert jnp.allclose(diff.mean, jnp.array([1.5, 0.0, -3.0]), atol=1e-7)
        assert diff.variance is not None
        assert jnp.allclose(diff.variance, jnp.array([0.2, 0.0, 0.9]), atol=1e-7)
        assert diff.total_uncertainty is not None
        assert jnp.allclose(diff.total_uncertainty, diff.variance, atol=1e-7)
        diff.validate()

    def test_zero_variance_difference_has_finite_zero_std_gradient(self) -> None:
        """Identical correlated targets give ``std == 0`` and finite zero gradient."""
        mean_a = jnp.array([5.0])
        mean_b = jnp.array([5.0])
        var_a = jnp.array([0.8])
        var_b = jnp.array([0.8])
        cov_ab = jnp.array([0.8])

        std_val, (g_va, g_vb, g_cab) = jax.value_and_grad(
            lambda va, vb, cab: correlated_difference_predictive(
                mean_a, mean_b, var_a=va, var_b=vb, cov_ab=cab
            ).std()[0],
            argnums=(0, 1, 2),
        )(var_a, var_b, cov_ab)
        assert float(std_val) == 0.0
        assert float(g_va[0]) == 0.0
        assert float(g_vb[0]) == 0.0
        assert float(g_cab[0]) == 0.0

    def test_rejects_negative_variances_and_cauchy_schwarz_violation_eagerly(self) -> None:
        mean_a = jnp.array([1.0])
        mean_b = jnp.array([0.0])
        with pytest.raises(ValueError, match=r"var_a"):
            correlated_difference_predictive(
                mean_a, mean_b, var_a=jnp.array([-0.1]), var_b=jnp.array([1.0])
            )
        with pytest.raises(ValueError, match=r"var_b"):
            correlated_difference_predictive(
                mean_a, mean_b, var_a=jnp.array([1.0]), var_b=jnp.array([-0.1])
            )
        with pytest.raises(ValueError, match=r"Cauchy-Schwarz"):
            correlated_difference_predictive(
                mean_a,
                mean_b,
                var_a=jnp.array([0.25]),
                var_b=jnp.array([0.25]),
                cov_ab=jnp.array([0.30]),
            )
        with pytest.raises(ValueError, match=r"Cauchy-Schwarz"):
            correlated_difference_predictive(
                mean_a,
                mean_b,
                var_a=jnp.array([0.0]),
                var_b=jnp.array([0.0]),
                cov_ab=jnp.array([1e-4]),
            )

    def test_tolerates_tiny_float32_rounding_at_perfect_correlation_boundary(self) -> None:
        """Sample/GP covariances exceeding ``sqrt(va*vb)`` by a few ULPs clamp cleanly to 0."""
        va = jnp.array([0.3], dtype=jnp.float32)
        vb = jnp.array([0.3], dtype=jnp.float32)
        cab = va * (1.0 + 2.0 * _EPS32)
        diff = correlated_difference_predictive(
            jnp.ones(1, dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.float32),
            var_a=va,
            var_b=vb,
            cov_ab=cab,
        )
        assert diff.variance is not None
        assert float(diff.variance[0]) == 0.0
        assert float(diff.std()[0]) == 0.0

    def test_jit_and_vmap_compile_once(self) -> None:
        mean_a = jnp.linspace(5.0, 9.0, 8)
        mean_b = jnp.linspace(4.0, 7.0, 8)
        var_a = jnp.full(8, 0.4)
        var_b = jnp.full(8, 0.6)
        cov_ab = jnp.full(8, 0.2)

        counter = TraceCounter()
        compiled = jax.jit(
            counter.wrap(
                jax.vmap(
                    lambda ma, mb, va, vb, cab: correlated_difference_predictive(
                        ma, mb, var_a=va, var_b=vb, cov_ab=cab
                    )
                )
            )
        )
        with counter.expect(new_traces=1):
            out1 = compiled(mean_a, mean_b, var_a, var_b, cov_ab)
        with counter.expect(new_traces=0):
            out2 = compiled(mean_a + 1.0, mean_b, var_a, var_b, cov_ab)

        eager = correlated_difference_predictive(
            mean_a, mean_b, var_a=var_a, var_b=var_b, cov_ab=cov_ab
        )
        assert jnp.allclose(out1.mean, eager.mean, atol=4.0 * _EPS32)
        assert out1.variance is not None and eager.variance is not None
        assert jnp.allclose(out1.variance, eager.variance, atol=4.0 * _EPS32)
        assert jnp.all(jnp.isfinite(out2.std()))


class TestLinearContrastPredictive:
    """General linear contrast ``c^T Y`` over joint multi-output Gaussian moments."""

    def test_matches_pairwise_difference_for_two_target_contrast(self) -> None:
        mean = jnp.array([[8.0, 5.5], [6.2, 6.8]])
        cov = jnp.array(
            [
                [[0.36, 0.18], [0.18, 0.49]],
                [[0.25, -0.10], [-0.10, 0.64]],
            ]
        )
        contrast = jnp.array([1.0, -1.0])

        lc = linear_contrast_predictive(mean, cov, contrast)
        pw = correlated_difference_predictive(
            mean[:, 0],
            mean[:, 1],
            var_a=cov[:, 0, 0],
            var_b=cov[:, 1, 1],
            cov_ab=cov[:, 0, 1],
        )
        assert jnp.allclose(lc.mean, pw.mean, atol=4.0 * _EPS32)
        assert lc.variance is not None and pw.variance is not None
        assert jnp.allclose(lc.variance, pw.variance, atol=4.0 * _EPS32)

    def test_rejects_asymmetric_or_indefinite_covariance_eagerly(self) -> None:
        mean = jnp.array([1.0, 2.0])
        contrast = jnp.array([1.0, -1.0])
        asymmetric = jnp.array([[1.0, 0.5], [0.1, 1.0]])
        indefinite = jnp.array([[1.0, 1.5], [1.5, 1.0]])
        with pytest.raises(ValueError, match=r"symmetric"):
            linear_contrast_predictive(mean, asymmetric, contrast)
        with pytest.raises(ValueError, match=r"positive semi-definite"):
            linear_contrast_predictive(mean, indefinite, contrast)
        with pytest.raises(ValueError, match=r"shape"):
            linear_contrast_predictive(mean, jnp.eye(3), contrast)

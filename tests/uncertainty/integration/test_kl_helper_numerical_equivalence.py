"""KL helper numerical equivalence tests.

Pins the contract: Opifex ``diagonal_gaussian_kl`` MUST produce the same
scalar value (within float tolerance) as the existing variational-framework
KL formula in ``opifex.neural.bayesian.variational_framework.MeanFieldGaussian``.

If this test ever fails, it means either:

* the underlying formula changed (bug — fix the change), or
* the legacy distrax-backed formula has been removed (expected — delete this
  test in the same commit that removes ``MeanFieldGaussian.kl_divergence``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.bayesian.variational_framework import MeanFieldGaussian
from opifex.uncertainty.kernels.bayesian import diagonal_gaussian_kl


def _make_mean_field_gaussian(num_params: int, *, seed: int = 0) -> MeanFieldGaussian:
    return MeanFieldGaussian(num_params=num_params, rngs=nnx.Rngs(seed))


def test_diagonal_gaussian_kl_matches_mean_field_gaussian_standard_normal_prior() -> None:
    """For the canonical N(0,1) prior, both helpers must agree to float-tolerance.

    Opifex ``diagonal_gaussian_kl`` delegates to Artifex
    ``gaussian_kl_divergence`` (reduction='sum'). The legacy
    ``MeanFieldGaussian.kl_divergence`` uses distrax. Both must produce the
    same scalar.
    """
    layer = _make_mean_field_gaussian(num_params=16, seed=1)
    mean = layer.mean.value
    logvar = 2.0 * layer.log_std.value

    legacy_kl = float(layer.kl_divergence(prior_mean=0.0, prior_std=1.0))
    opifex_kl = float(diagonal_gaussian_kl(mean, logvar, prior_mean=0.0, prior_std=1.0))

    assert opifex_kl == pytest.approx(legacy_kl, rel=1e-5, abs=1e-6)


def test_diagonal_gaussian_kl_matches_mean_field_gaussian_parametric_prior() -> None:
    """For a parametric ``(prior_mean=2, prior_std=3)`` prior, helpers must agree."""
    layer = _make_mean_field_gaussian(num_params=16, seed=2)
    mean = layer.mean.value
    logvar = 2.0 * layer.log_std.value

    legacy_kl = float(layer.kl_divergence(prior_mean=2.0, prior_std=3.0))
    opifex_kl = float(diagonal_gaussian_kl(mean, logvar, prior_mean=2.0, prior_std=3.0))

    assert opifex_kl == pytest.approx(legacy_kl, rel=1e-5, abs=1e-6)


def test_diagonal_gaussian_kl_is_jit_grad_safe_on_real_layer_parameters() -> None:
    """Wrap real ``MeanFieldGaussian`` parameters through ``jit(grad(...))``."""
    layer = _make_mean_field_gaussian(num_params=8, seed=3)
    mean = layer.mean.value
    logvar = 2.0 * layer.log_std.value

    grad_fn = jax.jit(
        jax.grad(lambda m, lv: diagonal_gaussian_kl(m, lv, prior_mean=0.0, prior_std=1.0))
    )
    grads_mean = grad_fn(mean, logvar)
    assert grads_mean.shape == mean.shape
    assert jnp.all(jnp.isfinite(grads_mean))

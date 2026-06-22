"""Pin ``BayesianSpectralConvolution`` integration against UQNO's existing surface.

UQNO currently ships its own local ``BayesianSpectralConvolution`` copy at
``opifex.neural.operators.specialized.uqno`` (deferred to a later phase's
UQNO migration). These integration tests pin three properties that the
shared ``opifex.uncertainty.layers.bayesian.BayesianSpectralConvolution``
holds today, so the eventual UQNO migration can swap in the shared layer
without changing the contract:

1. The shared layer produces the same input/output spatial shape that UQNO
   expects (1D and 2D rfft modes).
2. KL divergence is finite and matches the closed-form
   ``diagonal_gaussian_kl`` formula on each weight band.
3. Sampling routes through caller-owned RNG (the UQNO migration will need
   to thread ``rngs`` through its forward pass, exactly as Task 2.3 did for
   ``MultiFidelityPINN`` / ``ProbabilisticPINN``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.uncertainty import BayesianSpectralConvolution
from opifex.uncertainty.kernels.bayesian import diagonal_gaussian_kl


def test_bayesian_spectral_convolution_2d_preserves_uqno_spatial_shape() -> None:
    """2D output spatial shape matches input — the UQNO contract."""
    layer = BayesianSpectralConvolution(
        in_channels=2, out_channels=3, modes=(4, 4), prior_std=1.0, rngs=nnx.Rngs(0)
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 16, 16))
    deterministic = layer(x, deterministic=True)
    assert deterministic.shape == (1, 3, 16, 16)


def test_bayesian_spectral_convolution_1d_preserves_uqno_spatial_shape() -> None:
    layer = BayesianSpectralConvolution(
        in_channels=2, out_channels=3, modes=(8,), prior_std=1.0, rngs=nnx.Rngs(0)
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 32))
    deterministic = layer(x, deterministic=True)
    assert deterministic.shape == (1, 3, 32)


def test_bayesian_spectral_convolution_kl_matches_kernel_helper_2d() -> None:
    """Each of the four 2D weight bands contributes ``diagonal_gaussian_kl``."""
    layer = BayesianSpectralConvolution(
        in_channels=2, out_channels=3, modes=(4, 4), prior_std=1.0, rngs=nnx.Rngs(0)
    )
    expected = float(
        diagonal_gaussian_kl(
            layer.weight_mean[...], layer.weight_logvar[...], prior_mean=0.0, prior_std=1.0
        )
        + diagonal_gaussian_kl(
            layer.weight_imag_mean[...],
            layer.weight_imag_logvar[...],
            prior_mean=0.0,
            prior_std=1.0,
        )
        + diagonal_gaussian_kl(
            layer.weight_neg_h_mean[...],
            layer.weight_neg_h_logvar[...],
            prior_mean=0.0,
            prior_std=1.0,
        )
        + diagonal_gaussian_kl(
            layer.weight_neg_h_imag_mean[...],
            layer.weight_neg_h_imag_logvar[...],
            prior_mean=0.0,
            prior_std=1.0,
        )
    )
    layer_kl = float(layer.kl_divergence())
    assert layer_kl == expected


def test_bayesian_spectral_convolution_samples_without_caller_owned_rngs() -> None:
    """Without a per-call ``rngs`` the layer samples via its own stored stream
    (mirrors :class:`nnx.Dropout`); a caller may still thread rngs to override."""
    layer = BayesianSpectralConvolution(
        in_channels=2, out_channels=3, modes=(4, 4), rngs=nnx.Rngs(0)
    )
    x = jnp.ones((1, 2, 8, 8))
    out_a = layer(x)
    out_b = layer(x)
    assert not jnp.array_equal(out_a, out_b)


def test_bayesian_spectral_convolution_sampling_with_explicit_key_is_deterministic_given_key() -> (
    None
):
    layer = BayesianSpectralConvolution(
        in_channels=2, out_channels=3, modes=(4, 4), rngs=nnx.Rngs(0)
    )
    x = jnp.ones((1, 2, 8, 8))
    key = jax.random.PRNGKey(7)
    a = layer(x, rngs=key)
    b = layer(x, rngs=key)
    assert jnp.array_equal(a, b)

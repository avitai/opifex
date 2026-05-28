"""Tests for ``opifex.neural.operators.fno.bayesian`` (CASpec + BSC re-export).

The module exposes:

* ``BayesianSpectralConvolution`` — re-export of the canonical
  variational diagonal-Gaussian spectral conv living at
  :mod:`opifex.uncertainty.layers.bayesian`.
* ``ComputationAwareSpectralConvolution`` — sibling of BSC whose
  uncertainty over the spectral weights is maintained as a CAKF
  low-rank posterior (Pförtner+ 2024, arXiv:2405.08971;
  Wenger+ 2023 CAGP precursor, arXiv:2306.07879) instead of a
  diagonal-Gaussian one. The deterministic forward uses the
  posterior-mean spectral weights; ``cakf_refine`` runs a CAKF
  update step against caller-supplied observations of the flattened
  spectral-weight vector.

The tests cover:

1. ``BayesianSpectralConvolution`` regression — the canonical
   ``deterministic`` flag / per-call ``rngs`` pattern is preserved (no
   regression to the old ``training`` / ``sample`` kwargs that
   Phase 3 removed).
2. CASpec forward shape parity with BSC under deterministic mode.
3. ``cakf_refine`` returns a posterior with a non-empty low-rank
   factor when ``max_iter > 0``; the factor shrinks variance towards
   the observation under matching ``observation_matrix == I``.
4. JAX-transform compatibility (``jax.jit`` over a forward pass).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.operators.fno.bayesian import (
    BayesianSpectralConvolution,
    ComputationAwareSpectralConvolution,
)


def test_bayesian_spectral_convolution_preserves_deterministic_kwarg() -> None:
    """Phase-3 regression: ``deterministic`` is on ``__init__`` and ``__call__``.

    No legacy ``training`` / ``sample`` kwargs are present.
    """
    rngs = nnx.Rngs(0)
    bsc = BayesianSpectralConvolution(
        in_channels=2, out_channels=2, modes=(2,), deterministic=True, rngs=rngs
    )
    assert hasattr(bsc, "deterministic")
    assert bsc.deterministic is True
    # legacy attribute names introduced by Phase-3-removed pattern must NOT exist.
    assert not hasattr(bsc, "training")
    assert not hasattr(bsc, "sample")


def test_caspec_forward_shape_matches_bsc_on_1d_input() -> None:
    """CASpec must accept the same input shape as BSC and emit matching shape."""
    rngs = nnx.Rngs(1)
    caspec = ComputationAwareSpectralConvolution(
        in_channels=2,
        out_channels=3,
        modes=(2,),
        deterministic=True,
        rngs=rngs,
    )
    # BSC's 1D forward convention: (batch, channels, length).
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 2, 8))
    y = caspec(x)
    assert y.shape == (4, 3, 8)


def test_caspec_cakf_refine_returns_factor_with_max_iter_columns() -> None:
    """After ``cakf_refine``, the low-rank factor has ``max_iter`` columns."""
    rngs = nnx.Rngs(2)
    caspec = ComputationAwareSpectralConvolution(
        in_channels=1, out_channels=1, modes=(2,), deterministic=True, rngs=rngs
    )
    # Flatten the BSC mean parameters into a vector for the CAKF dimension.
    flat_dim = caspec.flat_parameter_dim
    observation = jnp.zeros(flat_dim)
    observation_matrix = jnp.eye(flat_dim)
    observation_cov = jnp.eye(flat_dim)
    max_iter = 3

    refined = caspec.cakf_refine(
        observation=observation,
        observation_matrix=observation_matrix,
        observation_cov=observation_cov,
        max_iter=max_iter,
    )
    # The refined wrapper carries a non-empty factor with max_iter columns.
    assert refined.cakf_factor.shape == (flat_dim, max_iter)
    assert refined.cakf_mean.shape == (flat_dim,)


def test_caspec_cakf_refine_shrinks_posterior_mean_toward_observation() -> None:
    """With ``H = I`` and small noise, the posterior mean moves toward the observation."""
    rngs = nnx.Rngs(3)
    caspec = ComputationAwareSpectralConvolution(
        in_channels=1, out_channels=1, modes=(2,), deterministic=True, rngs=rngs
    )
    flat_dim = caspec.flat_parameter_dim
    target = jnp.full((flat_dim,), 2.5)
    refined = caspec.cakf_refine(
        observation=target,
        observation_matrix=jnp.eye(flat_dim),
        observation_cov=1e-3 * jnp.eye(flat_dim),
        max_iter=flat_dim,
    )
    initial_mean = caspec.cakf_mean
    # Posterior mean is closer to the target than the prior mean (zero by
    # CAKF convention; BSC's mean is small-norm).
    assert jnp.linalg.norm(refined.cakf_mean - target) < jnp.linalg.norm(initial_mean - target)


def test_caspec_forward_is_jit_compatible() -> None:
    """CASpec's deterministic forward compiles under ``nnx.jit``."""
    rngs = nnx.Rngs(4)
    caspec = ComputationAwareSpectralConvolution(
        in_channels=1, out_channels=1, modes=(2,), deterministic=True, rngs=rngs
    )
    x = jax.random.normal(jax.random.PRNGKey(5), (2, 1, 8))

    @nnx.jit
    def forward(model: ComputationAwareSpectralConvolution, inp: jax.Array) -> jax.Array:
        return model(inp)

    y = forward(caspec, x)
    assert y.shape == (2, 1, 8)
    assert jnp.all(jnp.isfinite(y))

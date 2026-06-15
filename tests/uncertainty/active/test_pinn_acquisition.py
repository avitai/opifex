"""PINN-residual acquisition tests for Task 8.3.

Reference: ``../al4pde`` residual-based pool selection. We require:

* High-residual candidates outrank low-residual candidates.
* The returned :class:`AcquiredBatch` exposes residual + uncertainty
  metadata so downstream callers can inspect why each point was chosen.
* Both ``nnx.Module`` and plain callables are accepted as the surrogate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.active.pinn_acquisition import pinn_residual_acquisition


def _residual_fn(predictions: jax.Array, candidates: jax.Array) -> jax.Array:
    """Synthetic PDE residual ``r(x) = u_pred(x) - target(x)``.

    With ``target(x) = sin(pi x)`` and ``u_pred(x) = 0`` the residual
    magnitude tracks ``|sin(pi x)|``: peaks near ``x = 0.5`` and zeros at
    the boundary.
    """
    target = jnp.sin(jnp.pi * candidates.squeeze(-1))
    return predictions - target


class _ZeroModule(nnx.Module):
    """nnx.Module surrogate that always predicts zero."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.zeros(x.shape[:-1])


def _zero_callable(x: jax.Array) -> jax.Array:
    return jnp.zeros(x.shape[:-1])


class TestPINNResidualAcquisition:
    def test_high_residual_outranks_low_residual(self) -> None:
        candidates = jnp.linspace(0.0, 1.0, 11).reshape(-1, 1)
        rngs = nnx.Rngs(active_acquire=0)

        result = pinn_residual_acquisition(
            model=_zero_callable,
            candidates=candidates,
            residual_fn=_residual_fn,
            batch_size=3,
            rngs=rngs,
        )

        # Peak |sin(pi x)| occurs near x = 0.5 — i.e. index 5 of 11 evenly
        # spaced points. Top-3 picks must include index 5.
        chosen = {int(i) for i in result.indices}
        assert 5 in chosen
        # Boundary indices 0 and 10 have zero residual — must NOT be picked.
        assert 0 not in chosen and 10 not in chosen

    def test_exposes_residual_and_uncertainty_metadata(self) -> None:
        candidates = jnp.linspace(0.0, 1.0, 8).reshape(-1, 1)
        rngs = nnx.Rngs(active_acquire=0)
        result = pinn_residual_acquisition(
            model=_zero_callable,
            candidates=candidates,
            residual_fn=_residual_fn,
            batch_size=2,
            rngs=rngs,
        )
        metadata = result.metadata_dict()
        assert "residual_norm" in metadata
        assert "predictive_uncertainty" in metadata
        # The recorded residual norm must match what _residual_fn produced.
        expected = jnp.abs(jnp.sin(jnp.pi * candidates.squeeze(-1)))
        assert jnp.allclose(
            jnp.asarray(metadata["residual_norm"]),
            expected,
            atol=1e-6,
        )

    def test_accepts_nnx_module(self) -> None:
        candidates = jnp.linspace(0.0, 1.0, 6).reshape(-1, 1)
        rngs = nnx.Rngs(active_acquire=0)
        result = pinn_residual_acquisition(
            model=_ZeroModule(),
            candidates=candidates,
            residual_fn=_residual_fn,
            batch_size=2,
            rngs=rngs,
        )
        assert result.indices.shape == (2,)

    def test_batch_size_must_be_positive(self) -> None:
        candidates = jnp.zeros((5, 1))
        rngs = nnx.Rngs(active_acquire=0)
        with pytest.raises(ValueError, match="batch_size"):
            pinn_residual_acquisition(
                model=_zero_callable,
                candidates=candidates,
                residual_fn=_residual_fn,
                batch_size=0,
                rngs=rngs,
            )

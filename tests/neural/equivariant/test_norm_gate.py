r"""Tests for the norm-gated equivariant nonlinearity ``NormGate``.

Behaviour is specified against QHNet's ``NormGate``
(``../AIRS/OpenDFT/QHBench/QH9/models/QHNet.py``): the gating signal is an MLP of
the input scalars concatenated with the per-irrep **norms** of the non-scalar
channels; the MLP output replaces the scalar channels and scales each non-scalar
multiplicity. Unlike the rightmost-scalar :func:`~opifex.neural.equivariant.gate`,
every multiplicity is gated and no scalars are consumed as dedicated gates.

The load-bearing test is equivariance (invariant on scalars, equivariant on
vectors because each vector is scaled by an invariant gate). ``jit``/``grad``/
``vmap`` cleanliness is also checked.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant import Irreps, IrrepsArray, NormGate


_RNG = SO3Group()


def _rotate(x: IrrepsArray, rotation: jax.Array) -> IrrepsArray:
    rotated_chunks = []
    for (_, irrep), chunk in zip(x.irreps.blocks, x.chunks, strict=True):
        matrix = wigner_d(irrep.l, rotation).astype(chunk.dtype)
        rotated_chunks.append(jnp.einsum("ij,...uj->...ui", matrix, chunk))
    flat = [c.reshape(*c.shape[:-2], -1) for c in rotated_chunks]
    return IrrepsArray(x.irreps, jnp.concatenate(flat, axis=-1))


def _random_input(irreps: Irreps, seed: int, dtype: jnp.dtype = jnp.float64) -> IrrepsArray:
    array = jax.random.normal(jax.random.PRNGKey(seed), (irreps.dim,), dtype=dtype)
    return IrrepsArray(irreps, array)


class TestNormGate:
    def test_preserves_irreps(self) -> None:
        """The gate keeps the input layout (no scalars consumed)."""
        irreps = Irreps("4x0e + 3x1e + 2x2e")
        layer = NormGate(irreps, rngs=nnx.Rngs(0))
        out = layer(_random_input(irreps, 1))
        assert out.irreps == irreps

    def test_requires_scalars(self) -> None:
        """A gate needs at least one scalar channel to drive it."""
        try:
            NormGate(Irreps("2x1e"), rngs=nnx.Rngs(0))
        except ValueError:
            return
        raise AssertionError("expected ValueError when no scalar channel is present")

    def test_equivariance(self) -> None:
        """``NormGate(D x) = D NormGate(x)``: invariant gates scale equivariant vectors."""
        irreps = Irreps("4x0e + 2x1e + 1x2e")
        layer = NormGate(irreps, rngs=nnx.Rngs(2))
        x = _random_input(irreps, 3)
        rotation = _RNG.random_element(jax.random.PRNGKey(4)).astype(jnp.float64)
        lhs = _rotate(layer(x), rotation).array
        rhs = layer(_rotate(x, rotation)).array
        assert jnp.allclose(lhs, rhs, atol=1e-5)

    def test_scalar_only_is_pure_mlp_gate(self) -> None:
        """With no vectors the gate reduces to an MLP on the scalars."""
        irreps = Irreps("5x0e")
        layer = NormGate(irreps, rngs=nnx.Rngs(5))
        out = layer(_random_input(irreps, 6))
        assert out.irreps == irreps

    def test_jit_grad_vmap(self) -> None:
        """The module is jit/grad/vmap clean over a batch axis."""
        irreps = Irreps("4x0e + 2x1e")
        layer = NormGate(irreps, rngs=nnx.Rngs(7))
        batch = IrrepsArray(irreps, jax.random.normal(jax.random.PRNGKey(8), (6, irreps.dim)))

        @nnx.jit
        def run(module: NormGate) -> jax.Array:
            return jnp.sum(jax.vmap(lambda a: module(a).array)(batch) ** 2)

        grad = nnx.grad(run)(layer)
        assert grad is not None

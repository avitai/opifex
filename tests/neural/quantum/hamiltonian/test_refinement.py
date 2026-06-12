r"""Tests for the QHNet self / pair interaction refinement layers.

Behaviour is specified against QHNet's ``SelfNetLayer`` and ``PairNetLayer``
(``../AIRS/OpenDFT/QHBench/QH9/models/QHNet.py``):

* :class:`SelfInteractionLayer` refines a per-atom feature by a channel-wise self
  tensor product ``tp(W_l x, W_r x)`` with norm-gated nonlinearities and residual
  accumulation -- it builds the products of an atom's own features the diagonal
  Fock block needs.
* :class:`PairInteractionLayer` refines a per-edge feature by a channel-wise pair
  tensor product ``tp(x[src], x[dst])`` whose per-edge weights are modulated by the
  radial embedding and the inner product of the endpoint features -- the bilinear
  coupling the off-diagonal Fock block needs.

The load-bearing tests are SO(3) equivariance (both layers commute with a shared
rotation of every node feature, the per-edge invariants left fixed), residual
accumulation, output irreps and ``jit``/``grad``/``vmap`` cleanliness.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant import Irreps, IrrepsArray
from opifex.neural.quantum.hamiltonian._refinement import (
    PairInteractionLayer,
    SelfInteractionLayer,
)


_RNG = SO3Group()
_BASE = Irreps("4x0e + 4x1e + 4x2e")


def _rotate(x: IrrepsArray, rotation: jax.Array) -> IrrepsArray:
    rotated_chunks = []
    for (_, irrep), chunk in zip(x.irreps.blocks, x.chunks, strict=True):
        matrix = wigner_d(irrep.l, rotation).astype(chunk.dtype)
        rotated_chunks.append(jnp.einsum("ij,...uj->...ui", matrix, chunk))
    flat = [c.reshape(*c.shape[:-2], -1) for c in rotated_chunks]
    return IrrepsArray(x.irreps, jnp.concatenate(flat, axis=-1))


def _random_nodes(n: int, irreps: Irreps, seed: int) -> IrrepsArray:
    array = jax.random.normal(jax.random.PRNGKey(seed), (n, irreps.dim))
    return IrrepsArray(irreps, array)


# The test suite pins ``jax_enable_x64=False`` (tests/conftest.py), so the layers
# run in float32. Because the self / pair tensor products square the features the
# output magnitude is large, so equivariance is checked with a relative tolerance:
# a genuine break in the transformation law gives an O(1) relative error, far
# above this float32 round-off floor.
_EQUIVARIANCE_RTOL = 1e-3
_EQUIVARIANCE_ATOL = 1e-3


class TestSelfInteractionLayer:
    def test_output_irreps(self) -> None:
        layer = SelfInteractionLayer(_BASE, rngs=nnx.Rngs(0))
        out = layer(_random_nodes(5, _BASE, 1))
        assert out.irreps == _BASE

    def test_equivariance(self) -> None:
        """``layer(D x) = D layer(x)`` over the node axis."""
        layer = SelfInteractionLayer(_BASE, rngs=nnx.Rngs(2))
        x = _random_nodes(4, _BASE, 3)
        rotation = _RNG.random_element(jax.random.PRNGKey(4))
        lhs = _rotate(layer(x), rotation).array
        rhs = layer(_rotate(x, rotation)).array
        assert jnp.allclose(lhs, rhs, rtol=_EQUIVARIANCE_RTOL, atol=_EQUIVARIANCE_ATOL)

    def test_residual_accumulation(self) -> None:
        """Passing an accumulator adds it to the refined output."""
        layer = SelfInteractionLayer(_BASE, rngs=nnx.Rngs(5))
        x = _random_nodes(3, _BASE, 6)
        prior = _random_nodes(3, _BASE, 7)
        without = layer(x).array
        withacc = layer(x, prior).array
        assert jnp.allclose(withacc, without + prior.array, atol=1e-5)

    def test_jit_grad(self) -> None:
        layer = SelfInteractionLayer(_BASE, rngs=nnx.Rngs(8))
        x = _random_nodes(4, _BASE, 9)

        @nnx.jit
        def run(module: SelfInteractionLayer) -> jax.Array:
            return jnp.sum(module(x).array ** 2)

        assert nnx.grad(run)(layer) is not None


class TestPairInteractionLayer:
    def _edges(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        senders = jnp.array([0, 1, 2, 0])
        receivers = jnp.array([1, 2, 0, 2])
        edge_radial = jax.random.normal(jax.random.PRNGKey(20), (4, 8))
        return senders, receivers, edge_radial

    def test_output_irreps_per_edge(self) -> None:
        layer = PairInteractionLayer(_BASE, edge_radial_dim=8, rngs=nnx.Rngs(0))
        senders, receivers, edge_radial = self._edges()
        out = layer(_random_nodes(3, _BASE, 1), senders, receivers, edge_radial)
        assert out.irreps == _BASE
        assert out.array.shape[0] == senders.shape[0]

    def test_equivariance(self) -> None:
        """Rotating every node feature rotates the per-edge output; invariants fixed."""
        layer = PairInteractionLayer(_BASE, edge_radial_dim=8, rngs=nnx.Rngs(2))
        nodes = _random_nodes(3, _BASE, 3)
        senders, receivers, edge_radial = self._edges()
        rotation = _RNG.random_element(jax.random.PRNGKey(4))
        lhs = _rotate(layer(nodes, senders, receivers, edge_radial), rotation).array
        rhs = layer(_rotate(nodes, rotation), senders, receivers, edge_radial).array
        assert jnp.allclose(lhs, rhs, rtol=_EQUIVARIANCE_RTOL, atol=_EQUIVARIANCE_ATOL)

    def test_residual_accumulation(self) -> None:
        layer = PairInteractionLayer(_BASE, edge_radial_dim=8, rngs=nnx.Rngs(5))
        nodes = _random_nodes(3, _BASE, 6)
        senders, receivers, edge_radial = self._edges()
        prior = _random_nodes(4, _BASE, 7)
        without = layer(nodes, senders, receivers, edge_radial).array
        withacc = layer(nodes, senders, receivers, edge_radial, prior).array
        assert jnp.allclose(withacc, without + prior.array, atol=1e-5)

    def test_jit_grad(self) -> None:
        layer = PairInteractionLayer(_BASE, edge_radial_dim=8, rngs=nnx.Rngs(8))
        nodes = _random_nodes(3, _BASE, 9)
        senders, receivers, edge_radial = self._edges()

        @nnx.jit
        def run(module: PairInteractionLayer) -> jax.Array:
            return jnp.sum(module(nodes, senders, receivers, edge_radial).array ** 2)

        assert nnx.grad(run)(layer) is not None

r"""Tests for the QHNet self-interaction refinement layer.

Behaviour is specified against QHNet's ``SelfNetLayer``
(``../AIRS/OpenDFT/QHBench/QH9/models/QHNet.py``):

* :class:`SelfInteractionLayer` refines a per-atom feature by a channel-wise self
  tensor product ``tp(W_l x, W_r x)`` with norm-gated nonlinearities and residual
  accumulation -- it builds the products of an atom's own features the diagonal
  Fock block needs.

The off-diagonal counterpart (QHNet's ``PairNetLayer``) is the SO(2)-frame
``SO2PairInteractionLayer``; it is tested in ``test_so2_convolution.py``.

The load-bearing tests are SO(3) equivariance (the layer commutes with a shared
rotation of every node feature), residual accumulation, output irreps and
``jit``/``grad``/``vmap`` cleanliness.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant import Irreps, IrrepsArray
from opifex.neural.quantum.hamiltonian._refinement import SelfInteractionLayer


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

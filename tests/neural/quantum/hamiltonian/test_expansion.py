r"""Tests for the QHNet ``block_from_irreps`` / :class:`PairExpansion` mechanism.

The expansion maps a steerable per-pair feature (an
:class:`~opifex.neural.equivariant.IrrepsArray`) to a dense
``(2 l_i + 1) x (2 l_j + 1)`` Hamiltonian block by contracting the **last** index
of the real Clebsch-Gordan tensor ``clebsch_gordan(l_i, l_j, L)`` with the
``L``-chunk of the feature, summed over ``L``. The defining property tested here
is the block transformation law

.. math::  B(R \cdot x) = D^{l_i}(R)\, B(x)\, D^{l_j}(R)^{\top},

with ``D`` the real Wigner-D matrix -- the block-wise statement of
``H(R x) = D(R) H(x) D(R)^T`` that makes the assembled Hamiltonian equivariant.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.geometry.algebra.wigner import clebsch_gordan, wigner_d
from opifex.neural.equivariant import Irreps, IrrepsArray
from opifex.neural.quantum.hamiltonian._expansion import (
    block_from_irreps,
    pair_feature_irreps,
    PairExpansion,
)


def _random_rotation(seed: int) -> jax.Array:
    """Return a uniformly random proper rotation matrix (det = +1)."""
    key = jax.random.PRNGKey(seed)
    gaussian = jax.random.normal(key, (3, 3))
    orthogonal, _ = jnp.linalg.qr(gaussian)
    return orthogonal * jnp.sign(jnp.linalg.det(orthogonal))


def _rotate_irreps(feature: IrrepsArray, rotation: jax.Array) -> IrrepsArray:
    """Apply the Wigner-D rotation to every block of an :class:`IrrepsArray`."""
    rotated_chunks: list[jax.Array] = []
    for (_, irrep), chunk in zip(feature.irreps.blocks, feature.chunks, strict=True):
        wigner = wigner_d(irrep.l, rotation)
        rotated_chunks.append(jnp.einsum("ij,...j->...i", wigner, chunk))
    array = jnp.concatenate(
        [chunk.reshape(*feature.array.shape[:-1], -1) for chunk in rotated_chunks], axis=-1
    )
    return IrrepsArray(feature.irreps, array)


def test_pair_feature_irreps_covers_triangle_rule() -> None:
    """The pair feature must carry every ``L`` in ``|l_i-l_j|..l_i+l_j``."""
    irreps = pair_feature_irreps(1, 1)
    degrees = sorted(irrep.l for _, irrep in irreps)
    assert degrees == [0, 1, 2]


def test_block_from_irreps_shape() -> None:
    """A ``(l_i, l_j)`` block has shape ``(2 l_i + 1, 2 l_j + 1)``."""
    irreps = pair_feature_irreps(1, 0)
    feature = IrrepsArray(irreps, jnp.ones((irreps.dim,)))
    block = block_from_irreps(feature, 1, 0)
    assert block.shape == (3, 1)


@pytest.mark.parametrize(
    ("l_i", "l_j"),
    [(0, 0), (0, 1), (1, 0), (1, 1)],
)
def test_block_transformation_law(l_i: int, l_j: int) -> None:
    r"""``block_from_irreps`` satisfies ``B(Rx) = D(l_i) B(x) D(l_j)^T``."""
    rotation = _random_rotation(7)
    irreps = pair_feature_irreps(l_i, l_j)
    key = jax.random.PRNGKey(l_i * 10 + l_j)
    feature = IrrepsArray(irreps, jax.random.normal(key, (irreps.dim,)))

    block = block_from_irreps(feature, l_i, l_j)
    block_rotated = block_from_irreps(_rotate_irreps(feature, rotation), l_i, l_j)
    wigner_i = wigner_d(l_i, rotation)
    wigner_j = wigner_d(l_j, rotation)
    expected = wigner_i @ block @ wigner_j.T

    np.testing.assert_allclose(np.asarray(block_rotated), np.asarray(expected), atol=1e-5)


def test_block_from_irreps_contracts_last_cg_index() -> None:
    """Confirm the mechanism contracts the *last* (M) index of the CG tensor."""
    l_i, l_j = 1, 1
    irreps = pair_feature_irreps(l_i, l_j)
    key = jax.random.PRNGKey(3)
    feature = IrrepsArray(irreps, jax.random.normal(key, (irreps.dim,)))

    reference = jnp.zeros((2 * l_i + 1, 2 * l_j + 1))
    for (_, irrep), chunk in zip(feature.irreps.blocks, feature.chunks, strict=True):
        coupling = clebsch_gordan(l_i, l_j, irrep.l)
        reference = reference + jnp.einsum("ijM,M->ij", coupling, chunk[0])

    block = block_from_irreps(feature, l_i, l_j)
    np.testing.assert_allclose(np.asarray(block), np.asarray(reference), atol=1e-6)


def test_pair_expansion_is_equivariant_module() -> None:
    """The learnable :class:`PairExpansion` preserves the block transformation law."""
    from flax import nnx

    rotation = _random_rotation(11)
    node_irreps = Irreps("4x0e + 4x1o + 2x2e")
    expansion = PairExpansion(node_irreps, l_i=1, l_j=1, rngs=nnx.Rngs(0))

    key = jax.random.PRNGKey(5)
    node_feature = IrrepsArray(node_irreps, jax.random.normal(key, (node_irreps.dim,)))

    # Default multiplicities are 1; squeeze the (mul_i, mul_j) = (1, 1) axes.
    block = expansion(node_feature)[0, 0]
    block_rotated = expansion(_rotate_irreps(node_feature, rotation))[0, 0]
    wigner_i = wigner_d(1, rotation)
    wigner_j = wigner_d(1, rotation)
    expected = wigner_i @ block @ wigner_j.T

    np.testing.assert_allclose(np.asarray(block_rotated), np.asarray(expected), atol=1e-5)


def test_pair_expansion_multiplicity_axes() -> None:
    """Multiplicities distinguish same-``l`` shells (the oxygen 1s/2s case)."""
    from flax import nnx

    node_irreps = Irreps("8x0e + 4x1o")
    expansion = PairExpansion(node_irreps, l_i=0, l_j=0, mul_i=2, mul_j=2, rngs=nnx.Rngs(0))
    feature = IrrepsArray(node_irreps, jnp.ones((node_irreps.dim,)))
    block = expansion(feature)
    assert block.shape == (2, 2, 1, 1)
    # The four shell-pair sub-blocks are not all identical (distinct projections).
    flat = np.asarray(block).reshape(4)
    assert len(set(np.round(flat, 6))) > 1


def test_pair_expansion_jit_grad_vmap() -> None:
    """The expansion is ``jit``/``grad``/``vmap`` clean."""
    from flax import nnx

    node_irreps = Irreps("4x0e + 4x1o")
    expansion = PairExpansion(node_irreps, l_i=1, l_j=0, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(expansion)

    def block_norm(state_in: nnx.State, array: jax.Array) -> jax.Array:
        module = nnx.merge(graphdef, state_in)
        feature = IrrepsArray(node_irreps, array)
        return jnp.sum(module(feature) ** 2)

    array = jnp.ones((node_irreps.dim,))
    jitted = jax.jit(block_norm)
    value = jitted(state, array)
    assert jnp.isfinite(value)

    gradient = jax.grad(block_norm, argnums=1)(state, array)
    assert gradient.shape == array.shape

    batch = jnp.broadcast_to(array, (4, node_irreps.dim))
    batched = jax.vmap(block_norm, in_axes=(None, 0))(state, batch)
    assert batched.shape == (4,)

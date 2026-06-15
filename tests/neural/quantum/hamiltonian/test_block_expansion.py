r"""Tests for :class:`HamiltonianBlockExpansion` (QHNet block-form Fock head).

The head maps a steerable bottleneck feature
(:class:`~opifex.neural.equivariant.IrrepsArray`) plus an invariant embedding to a
dense ``(14, 14)`` Fock block by composing the Clebsch-Gordan expansion over the
``(l_out1, l_out2)`` sub-block grid of ``BLOCK_IRREPS = 3x0e + 2x1e + 1x2e``. The
defining property is the QHNet block transformation law

.. math::  H(R \cdot x) = D_{14}(R)\, H(x)\, D_{14}(R)^{\top},

with ``D_{14}`` the real Wigner-D of ``BLOCK_IRREPS`` -- the block-wise statement
of ``H(R x) = D(R) H(x) D(R)^T`` that makes the assembled Hamiltonian equivariant
(reference ``divelab/AIRS`` ``OpenDFT/QHBench/QH9/models/Expanson.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant import Irreps, IrrepsArray
from opifex.neural.quantum.hamiltonian._block_expansion import HamiltonianBlockExpansion
from opifex.neural.quantum.hamiltonian._orbital_layout import BLOCK_IRREPS, FULL_ORBITALS


# A bottleneck feature carrying every degree reachable by the block (l up to 4 =
# 2 + 2 for the d-d shell pair), each with some multiplicity.
FEATURE_IRREPS = Irreps("4x0e + 4x1e + 4x2e + 4x3e + 4x4e")
EMBED_DIM = 16


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


def _wigner_block(rotation: jax.Array) -> jax.Array:
    """Build the ``(14, 14)`` block-diagonal Wigner-D of ``BLOCK_IRREPS``."""
    matrices: list[jax.Array] = []
    for mul, irrep in BLOCK_IRREPS.blocks:
        wigner = wigner_d(irrep.l, rotation)
        matrices.extend([wigner] * mul)
    return jax.scipy.linalg.block_diag(*matrices)


def _make_module(seed: int = 0) -> HamiltonianBlockExpansion:
    """Construct a head with the shared test feature/embedding layout."""
    return HamiltonianBlockExpansion(
        feature_irreps=FEATURE_IRREPS,
        embed_dim=EMBED_DIM,
        rngs=nnx.Rngs(seed),
    )


def _random_inputs(seed: int, leading: tuple[int, ...] = ()) -> tuple[IrrepsArray, jax.Array]:
    """Return a random (feature, embedding) input pair with given leading axes."""
    feat_key, embed_key = jax.random.split(jax.random.PRNGKey(seed))
    feature = IrrepsArray(
        FEATURE_IRREPS, jax.random.normal(feat_key, (*leading, FEATURE_IRREPS.dim))
    )
    embedding = jax.random.normal(embed_key, (*leading, EMBED_DIM))
    return feature, embedding


def test_block_output_shape() -> None:
    """A single feature expands to a ``(14, 14)`` block."""
    module = _make_module()
    feature, embedding = _random_inputs(1)
    block = module(feature, embedding)
    assert block.shape == (FULL_ORBITALS, FULL_ORBITALS)


def test_block_output_is_batched() -> None:
    """A leading axis of ``n`` nodes/edges gives an ``(n, 14, 14)`` stack."""
    module = _make_module()
    feature, embedding = _random_inputs(2, leading=(7,))
    block = module(feature, embedding)
    assert block.shape == (7, FULL_ORBITALS, FULL_ORBITALS)


def test_num_path_weight_and_bias_are_positive_ints() -> None:
    """The head exposes QHNet-style ``num_path_weight`` / ``num_bias`` counts."""
    module = _make_module()
    assert isinstance(module.num_path_weight, int)
    assert isinstance(module.num_bias, int)
    assert module.num_path_weight > 0
    # Bias applies only to scalar (l_out1 = l_out2 = 0) sub-blocks: 3x3 s-shells.
    assert module.num_bias == 9


def test_block_transformation_law() -> None:
    r"""The critical test: ``H(Rx) = D_14(R) H(x) D_14(R)^T`` to float64 tolerance.

    This proves the head assembles a correct equivariant Fock block.
    """
    jax.config.update("jax_enable_x64", True)
    rotation = _random_rotation(7)
    module = _make_module()
    feature, embedding = _random_inputs(3)
    feature = IrrepsArray(FEATURE_IRREPS, feature.array.astype(jnp.float64))
    embedding = embedding.astype(jnp.float64)

    block = module(feature, embedding)
    block_rotated = module(_rotate_irreps(feature, rotation), embedding)
    wigner = _wigner_block(rotation)
    expected = wigner @ block @ wigner.T

    residual = float(jnp.max(jnp.abs(block_rotated - expected)))
    assert residual < 1e-5, f"equivariance residual {residual} exceeds 1e-5"


def test_diagonal_and_offdiagonal_share_module() -> None:
    """The SAME head serves node (diagonal) and pair (off-diagonal) embeddings."""
    module = _make_module()
    node_feature, node_embed = _random_inputs(4, leading=(3,))
    # Off-diagonal: a concatenated pair embedding of the same width.
    edge_feature, edge_embed = _random_inputs(5, leading=(6,))
    diagonal = module(node_feature, node_embed)
    off_diagonal = module(edge_feature, edge_embed)
    assert diagonal.shape == (3, 14, 14)
    assert off_diagonal.shape == (6, 14, 14)


def test_block_expansion_is_jit_grad_vmap_safe() -> None:
    """jit/grad/vmap smoke: the head is transform-clean."""
    module = _make_module()
    feature, embedding = _random_inputs(6, leading=(4,))

    graphdef, state = nnx.split(module)

    def scalar_loss(state_in: nnx.State, array: jax.Array, embed: jax.Array) -> jax.Array:
        rebuilt = nnx.merge(graphdef, state_in)
        block = rebuilt(IrrepsArray(FEATURE_IRREPS, array), embed)
        return jnp.sum(block**2)

    # jit
    jitted = jax.jit(scalar_loss)
    value = jitted(state, feature.array, embedding)
    assert jnp.isfinite(value)
    # grad w.r.t. parameters
    grads = jax.grad(scalar_loss)(state, feature.array, embedding)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves and all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    # vmap over the leading axis
    per_sample = jax.vmap(lambda array, embed: scalar_loss(state, array, embed))(
        feature.array, embedding
    )
    assert per_sample.shape == (4,)


def test_block_expansion_survives_repeated_nnx_jit_calls() -> None:
    """``nnx.jit`` over the head stays valid across repeated calls.

    A training loop calls the jitted step on the same module instance every step;
    on the second call NNX checks the graphdef *metadata* for equality and rejects
    array-valued metadata. The head's static metadata (Clebsch-Gordan tensors,
    path routing) must therefore be hashable tuples, not arrays.
    """
    module = _make_module()
    feature, embedding = _random_inputs(8, leading=(5,))

    @nnx.jit
    def call(head: HamiltonianBlockExpansion, array: jax.Array, embed: jax.Array) -> jax.Array:
        return head(IrrepsArray(FEATURE_IRREPS, array), embed)

    first = call(module, feature.array, embedding)
    # Second call on the same instance triggers the graphdef metadata equality
    # check that rejects array-valued metadata.
    second = call(module, feature.array + 0.01, embedding)
    assert first.shape == second.shape == (5, 14, 14)
    assert jnp.all(jnp.isfinite(first)) and jnp.all(jnp.isfinite(second))


@pytest.mark.parametrize("leading", [(), (1,), (10,)])
def test_block_finite_for_various_batch_shapes(leading: tuple[int, ...]) -> None:
    """Output is finite across scalar and batched leading shapes."""
    module = _make_module()
    feature, embedding = _random_inputs(9, leading=leading)
    block = module(feature, embedding)
    assert jnp.all(jnp.isfinite(block))

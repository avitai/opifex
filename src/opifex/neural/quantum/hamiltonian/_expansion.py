r"""QHNet block expansion: steerable pair feature -> dense Hamiltonian block.

The heart of an equivariant Hamiltonian predictor (Yu et al. 2023, "Efficient and
Equivariant Graph Networks for Predicting Quantum Hamiltonian", QHNet,
arXiv:2306.04922; reference ``divelab/AIRS`` ``OpenDFT/QHBench/QH9/models``
``Expansion``) is the *inverse* use of the Clebsch-Gordan tensor that drives the
tensor product. A tensor product **couples** two irreps ``l_i (x) l_j`` down to a
single output irrep ``L``; the expansion **decouples** a steerable feature -- a
direct sum over ``L`` of ``|l_i - l_j| <= L <= l_i + l_j`` -- back into the
``(2 l_i + 1) x (2 l_j + 1)`` matrix block that couples to those ``L``.

For a shell pair ``(l_i, l_j)`` the block is

.. math::
   B[a, b] = \sum_{L} \sum_{M}
             C^{l_i\,l_j\,L}_{a\,b\,M}\, f^{L}_{M},

i.e. the contraction of the **last** index ``M`` of the real Clebsch-Gordan
tensor :func:`opifex.geometry.algebra.wigner.clebsch_gordan` ``(l_i, l_j, L)``
with the ``L``-chunk ``f^{L}`` of the input feature (QHNet's
``einsum("ijk, ...k -> ...ij")`` with ``k = M`` the CG's last axis). Because
``C`` is the intertwiner between ``D^{l_i} (x) D^{l_j}`` and ``D^{L}`` and the
feature transforms as ``f^{L} -> D^{L} f^{L}``, the block satisfies the QHNet
transformation law

.. math::  B(R \cdot x) = D^{l_i}(R)\, B(x)\, D^{l_j}(R)^{\top},

which is the block-wise statement of ``H(R x) = D(R) H(x) D(R)^T`` that makes the
assembled Hamiltonian equivariant.

The block shapes are static per ``(l_i, l_j)``, so the whole expansion is
``jit``/``grad``/``vmap`` clean. The CG tensors are static constants stored as
nested Python tuples (so they live in ``nnx`` static aux-data, never as parameter
leaves) and rebuilt as compile-time-constant arrays inside ``__call__`` -- the
same pattern as
:class:`opifex.neural.equivariant.FullyConnectedTensorProduct`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.geometry.algebra.wigner import clebsch_gordan
from opifex.neural.equivariant import EquivariantLinear, Irreps, IrrepsArray


def pair_feature_irreps(l_i: int, l_j: int) -> Irreps:
    r"""Return the steerable layout a ``(l_i, l_j)`` block expands from.

    A block coupling shells of degree ``l_i`` and ``l_j`` carries exactly the
    irreps ``L`` permitted by the triangle rule ``|l_i - l_j| <= L <= l_i + l_j``,
    each with multiplicity one. Parity is ``(-1)^L`` (the spherical-harmonic
    convention), matching the steerable node features produced by the trunk.

    Args:
        l_i: Angular momentum of the row (receiver) shell.
        l_j: Angular momentum of the column (sender) shell.

    Returns:
        The single-multiplicity :class:`~opifex.neural.equivariant.Irreps`
        ``sum_L 1 x L{parity}``.
    """
    terms = [
        f"1x{degree}{'e' if degree % 2 == 0 else 'o'}"
        for degree in range(abs(l_i - l_j), l_i + l_j + 1)
    ]
    return Irreps("+".join(terms))


def block_from_irreps(feature: IrrepsArray, l_i: int, l_j: int) -> jax.Array:
    r"""Expand a steerable pair feature into a dense ``(2l_i+1, 2l_j+1)`` block.

    Contracts the **last** index ``M`` of ``clebsch_gordan(l_i, l_j, L)`` with the
    ``L``-chunk of ``feature``, summed over the triangle-rule degrees ``L``. This
    is the QHNet ``Expansion`` mechanism (``einsum("ijk, ...k -> ...ij")``).

    Args:
        feature: A pair feature whose irreps include every degree ``L`` in
            ``|l_i - l_j|..l_i + l_j`` (typically exactly
            :func:`pair_feature_irreps`). Leading (batch) axes are supported.
        l_i: Angular momentum of the row (receiver) shell.
        l_j: Angular momentum of the column (sender) shell.

    Returns:
        The block of shape ``feature.shape[:-1] + (2 l_i + 1, 2 l_j + 1)``.

    Raises:
        ValueError: If ``feature`` is missing a required degree ``L``.
    """
    degrees = {
        irrep.l: chunk
        for (_, irrep), chunk in zip(feature.irreps.blocks, feature.chunks, strict=True)
    }
    leading_shape = feature.array.shape[:-1]
    dtype = feature.array.dtype
    block = jnp.zeros((*leading_shape, 2 * l_i + 1, 2 * l_j + 1), dtype=dtype)
    for degree in range(abs(l_i - l_j), l_i + l_j + 1):
        if degree not in degrees:
            raise ValueError(
                f"pair feature {feature.irreps!r} is missing degree {degree} required "
                f"to expand the (l_i={l_i}, l_j={l_j}) block"
            )
        coupling = clebsch_gordan(l_i, l_j, degree).astype(dtype)
        # ``degrees[degree]`` has a multiplicity axis (``mul = 1``); drop it.
        chunk = degrees[degree][..., 0, :]
        block = block + jnp.einsum("ijM,...M->...ij", coupling, chunk)
    return block


class PairExpansion(nnx.Module):
    r"""Learnable expansion of node/edge features into ``(l_i, l_j)`` blocks.

    Maps a steerable feature to ``mul_i x mul_j`` copies of the pair-feature
    layout with an equivariant linear layer, reshapes to multiplicity axes, and
    expands each copy with the last-index Clebsch-Gordan contraction. The
    multiplicities ``mul_i`` / ``mul_j`` distinguish *shells of the same angular
    momentum* on one atom (e.g. STO-3G oxygen's ``1s`` and ``2s``), exactly as
    QHNet's ``Expansion`` uses ``irrep_out_1 = irrep_out_2 = "n_s x 0e + n_p x
    1e + ..."`` and the ``einsum("wuv, ijk, bwk -> buivj")`` output multiplicity
    axes ``(u, v)``.

    Because every stage is equivariant, the output obeys the QHNet block law per
    multiplicity ``B[u, v](R x) = D(l_i) B[u, v](x) D(l_j)^T``.

    Args:
        node_irreps: Layout of the incoming steerable feature.
        l_i: Angular momentum of the row (receiver) shells.
        l_j: Angular momentum of the column (sender) shells.
        mul_i: Number of distinct row shells of degree ``l_i`` on the atom.
        mul_j: Number of distinct column shells of degree ``l_j`` on the atom.
        rngs: Random number generators (keyword-only) seeding the linear map.
    """

    def __init__(
        self,
        node_irreps: Irreps,
        *,
        l_i: int,
        l_j: int,
        mul_i: int = 1,
        mul_j: int = 1,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the equivariant projection onto ``mul_i x mul_j`` pair layouts."""
        super().__init__()
        self.l_i = int(l_i)
        self.l_j = int(l_j)
        self.mul_i = int(mul_i)
        self.mul_j = int(mul_j)
        copies = self.mul_i * self.mul_j
        # Restrict to degrees the node features can supply, with ``copies``
        # multiplicities of each so the projection learns one pair feature per
        # (row-shell, col-shell) combination.
        available = {irrep.l for _, irrep in Irreps(node_irreps).blocks}
        target_blocks = tuple(
            (copies, irrep)
            for _, irrep in pair_feature_irreps(self.l_i, self.l_j).blocks
            if irrep.l in available
        )
        self._projection_irreps = Irreps(target_blocks)
        self.projection = EquivariantLinear(node_irreps, self._projection_irreps, rngs=rngs)

    def __call__(self, feature: IrrepsArray) -> jax.Array:
        """Expand ``feature`` into a ``(mul_i, mul_j, 2l_i+1, 2l_j+1)`` block tensor.

        Args:
            feature: A steerable node/edge feature with ``self``'s input irreps.
                Leading (batch) axes are supported.

        Returns:
            Blocks of shape
            ``feature.shape[:-1] + (mul_i, mul_j, 2 l_i + 1, 2 l_j + 1)``.
        """
        projected = self.projection(feature)
        leading_shape = feature.array.shape[:-1]
        dtype = feature.array.dtype
        block = jnp.zeros(
            (*leading_shape, self.mul_i, self.mul_j, 2 * self.l_i + 1, 2 * self.l_j + 1),
            dtype=dtype,
        )
        for (_, irrep), chunk in zip(projected.irreps.blocks, projected.chunks, strict=True):
            degree = irrep.l
            if not abs(self.l_i - self.l_j) <= degree <= self.l_i + self.l_j:
                continue
            coupling = clebsch_gordan(self.l_i, self.l_j, degree).astype(dtype)
            # ``chunk`` has shape (..., copies, 2L+1); split copies into (mul_i, mul_j).
            reshaped = chunk.reshape(*leading_shape, self.mul_i, self.mul_j, irrep.dim)
            block = block + jnp.einsum("abM,...uvM->...uvab", coupling, reshaped)
        return block


__all__ = ["PairExpansion", "block_from_irreps", "pair_feature_irreps"]

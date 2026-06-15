r"""QHNet block expansion head: bottleneck feature + embedding -> ``(14, 14)`` block.

This is the equivariant Wigner-3j / Clebsch-Gordan **expansion head** of a
block-form Hamiltonian predictor (Yu et al. 2023, "QHNet", arXiv:2306.04922;
reference ``divelab/AIRS`` ``OpenDFT/QHBench/QH9/models/Expanson.py`` ``Expansion``).
It maps a steerable bottleneck feature (an
:class:`~opifex.neural.equivariant.IrrepsArray`) -- per node for diagonal Fock
blocks, per directed edge for off-diagonal blocks -- to the dense ``(14, 14)``
matrix block of :data:`~opifex.neural.quantum.hamiltonian._orbital_layout.BLOCK_IRREPS`
(``3x0e + 2x1e + 1x2e``).

Mechanism (QHNet ``Expansion``)
-------------------------------
For each ordered pair of output shells ``(s_i, s_j)`` of degrees ``(l_i, l_j)``
in ``BLOCK_IRREPS`` and each input degree ``L`` with
``|l_i - l_j| <= L <= l_i + l_j`` carried by the feature, the ``(2 l_i + 1,
2 l_j + 1)`` sub-block is the **last-index Clebsch-Gordan contraction**

.. math::
   B^{(s_i, s_j)}[a, b] = \sum_{L}\ \sum_{w}\ \sum_{M}
       C^{l_i\,l_j\,L}_{a\,b\,M}\, \big(g^{(s_i, s_j, L)}_{w}\, f^{L}_{w, M}\big),

where ``C = clebsch_gordan(l_i, l_j, L)`` (reused from
:func:`opifex.geometry.algebra.wigner.clebsch_gordan` -- *not* reimplemented), ``f^L``
is the ``L``-chunk of the feature with multiplicity index ``w``, and ``g`` is a
**per-sample path weight** produced by an MLP on a provided invariant embedding
(QHNet's ``weights is not None`` path, ``einsum("bwuv, bwk -> buvk")`` followed by
``einsum("ijk, buvk -> buivj")``; here every output shell has multiplicity one so
``u = v = 1``). Scalar sub-blocks (``l_i = l_j = L = 0``) additionally receive a
per-sample bias, mirroring QHNet's ``bias_weights``. The same MLP-driven head thus
serves diagonal blocks (node embedding) and off-diagonal blocks (concatenated pair
embedding) -- one module, no duplication.

Because ``clebsch_gordan(l_i, l_j, L)`` is the intertwiner between
``D^{l_i} (x) D^{l_j}`` and ``D^{L}`` and the feature transforms as
``f^{L} -> D^{L} f^{L}`` (with the invariant embedding -- hence the path weights --
unchanged under rotation), every sub-block obeys
``B(R x) = D^{l_i}(R) B(x) D^{l_j}(R)^{\top}``. Tiled over the shell grid this is
the QHNet block law ``H(R x) = D_{14}(R) H(x) D_{14}(R)^{\top}`` that makes the
assembled Hamiltonian equivariant.

The opifex ``clebsch_gordan`` is unit-Frobenius normalized (e3nn real basis), so it
differs from ``e3nn.o3.wigner_3j`` only by the scalar factor ``1 / sqrt(2L+1)``.
That constant is absorbed into the learnable per-path weight ``g`` and leaves the
transformation law unchanged (verified numerically by the equivariance test), so
reusing ``clebsch_gordan`` is both correct and DRY.

The Clebsch-Gordan tensors and the path routing are compile-time constants stored
as nested Python tuples (static ``nnx`` aux-data, never parameter leaves or
array-valued metadata), keeping the head ``jit``/``grad``/``vmap`` clean and valid
across repeated ``nnx.jit`` calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.geometry.algebra.wigner import clebsch_gordan
from opifex.neural.equivariant import Irreps, IrrepsArray
from opifex.neural.quantum.hamiltonian._orbital_layout import BLOCK_IRREPS, FULL_ORBITALS


logger = logging.getLogger(__name__)

_DEFAULT_FEATURE_IRREPS = "8x0e + 8x1e + 8x2e + 8x3e + 8x4e"


@dataclass(frozen=True, slots=True, kw_only=True)
class _ExpansionPath:
    """One ``(output shell pair, input degree)`` Clebsch-Gordan contraction path.

    Every field is hashable (ints / a flattened tuple of the constant CG tensor),
    so a tuple of paths is valid Flax-NNX graphdef static metadata -- it survives
    the per-call metadata equality check across repeated ``nnx.jit`` invocations.

    Attributes:
        row_offset: Start row of the sub-block in the ``(14, 14)`` matrix.
        col_offset: Start column of the sub-block in the ``(14, 14)`` matrix.
        l_i: Degree of the row (receiver) output shell.
        l_j: Degree of the column (sender) output shell.
        degree: Input degree ``L`` contracted on this path.
        feature_slice: ``(start, stop)`` slice of the feature axis for the
            ``L``-chunk this path consumes (multiplicity-major, width
            ``mul_L * (2L + 1)``).
        multiplicity: Multiplicity ``mul_L`` of degree ``L`` in the feature.
        weight_offset: Start index of this path's ``mul_L`` weights in the flat
            per-sample weight vector.
        bias_offset: Start index of this path's bias in the flat per-sample bias
            vector, or ``-1`` if the path carries no bias (non-scalar sub-block).
        coupling_flat: The constant ``clebsch_gordan(l_i, l_j, L)`` tensor flattened
            to a row-major tuple (rebuilt as a compile-time array in ``__call__``).
    """

    row_offset: int
    col_offset: int
    l_i: int
    l_j: int
    degree: int
    feature_slice: tuple[int, int]
    multiplicity: int
    weight_offset: int
    bias_offset: int
    coupling_flat: tuple[float, ...]


def _output_shells() -> tuple[tuple[int, int], ...]:
    """Return ``(offset, l)`` for each output shell of ``BLOCK_IRREPS`` in order.

    A ``mul x l`` block contributes ``mul`` consecutive shells of degree ``l``,
    each occupying ``2l + 1`` rows/columns -- the irrep-ordered 14-slot layout.
    """
    shells: list[tuple[int, int]] = []
    offset = 0
    for mul, irrep in BLOCK_IRREPS.blocks:
        for _ in range(mul):
            shells.append((offset, irrep.l))
            offset += irrep.dim
    return tuple(shells)


def _feature_degree_slices(feature_irreps: Irreps) -> dict[int, tuple[int, int, int]]:
    """Map input degree ``L`` -> ``(slice_start, slice_stop, multiplicity)``.

    Args:
        feature_irreps: Layout of the incoming bottleneck feature.

    Returns:
        For every degree present in ``feature_irreps`` (each assumed to appear in a
        single block), the feature-axis slice and its multiplicity.
    """
    degree_slices: dict[int, tuple[int, int, int]] = {}
    for (mul, irrep), feature_slice in zip(
        feature_irreps.blocks, feature_irreps.slices(), strict=True
    ):
        degree_slices[irrep.l] = (feature_slice.start, feature_slice.stop, mul)
    return degree_slices


def _build_paths(feature_irreps: Irreps) -> tuple[tuple[_ExpansionPath, ...], int, int]:
    """Enumerate every Clebsch-Gordan expansion path over the output shell grid.

    Args:
        feature_irreps: Layout of the incoming bottleneck feature.

    Returns:
        The tuple of :class:`_ExpansionPath`, the total per-sample path-weight
        count ``num_path_weight`` and the total per-sample bias count ``num_bias``.
    """
    degree_slices = _feature_degree_slices(feature_irreps)
    shells = _output_shells()
    paths: list[_ExpansionPath] = []
    weight_offset = 0
    bias_offset = 0
    for row_offset, l_i in shells:
        for col_offset, l_j in shells:
            for degree in range(abs(l_i - l_j), l_i + l_j + 1):
                if degree not in degree_slices:
                    continue
                start, stop, multiplicity = degree_slices[degree]
                is_scalar = l_i == 0 and l_j == 0 and degree == 0
                coupling = clebsch_gordan(l_i, l_j, degree)
                paths.append(
                    _ExpansionPath(
                        row_offset=row_offset,
                        col_offset=col_offset,
                        l_i=l_i,
                        l_j=l_j,
                        degree=degree,
                        feature_slice=(start, stop),
                        multiplicity=multiplicity,
                        weight_offset=weight_offset,
                        bias_offset=bias_offset if is_scalar else -1,
                        coupling_flat=tuple(coupling.reshape(-1).tolist()),
                    )
                )
                weight_offset += multiplicity
                if is_scalar:
                    bias_offset += 1
    return tuple(paths), weight_offset, bias_offset


class HamiltonianBlockExpansion(nnx.Module):
    r"""Expand a bottleneck feature + invariant embedding into a ``(14, 14)`` block.

    Implements QHNet's ``Expansion`` (reference
    ``OpenDFT/QHBench/QH9/models/Expanson.py``) over the output shell grid of
    :data:`~opifex.neural.quantum.hamiltonian._orbital_layout.BLOCK_IRREPS`
    (``3x0e + 2x1e + 1x2e``). Per-sample path weights (and scalar-block biases) are
    produced by an MLP on a provided invariant embedding, so the *same* module
    builds diagonal blocks (from a node embedding) and off-diagonal blocks (from a
    concatenated pair embedding). The Clebsch-Gordan contraction reuses
    :func:`opifex.geometry.algebra.wigner.clebsch_gordan` (no reimplementation).

    Args:
        feature_irreps: Layout of the incoming steerable bottleneck feature.
            Defaults to ``8x0e + 8x1e + 8x2e + 8x3e + 8x4e`` -- every degree
            reachable by a ``d``-``d`` (``l = 2, 2``) shell pair (``L`` up to 4).
        embed_dim: Width of the invariant embedding driving the weight/bias MLP.
        mlp_hidden_dim: Hidden width of the weight/bias MLP.
        rngs: Random number generators (keyword-only) seeding the MLP.
    """

    def __init__(
        self,
        *,
        feature_irreps: Irreps | str = _DEFAULT_FEATURE_IRREPS,
        embed_dim: int = 64,
        mlp_hidden_dim: int = 128,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the expansion paths and the embedding -> (weights, bias) MLP."""
        super().__init__()
        self.feature_irreps = Irreps(feature_irreps)
        self.embed_dim = int(embed_dim)
        paths, num_path_weight, num_bias = _build_paths(self.feature_irreps)
        if num_path_weight == 0:
            raise ValueError(
                f"feature irreps {self.feature_irreps!r} supply no degree reachable by "
                f"the {BLOCK_IRREPS!r} shell grid; the block would be identically zero"
            )
        self._paths = paths
        self.num_path_weight = num_path_weight
        self.num_bias = num_bias
        # MLP: invariant embedding -> flat (path weights || biases) per sample.
        self.hidden = nnx.Linear(self.embed_dim, int(mlp_hidden_dim), rngs=rngs)
        self.readout = nnx.Linear(int(mlp_hidden_dim), num_path_weight + num_bias, rngs=rngs)

    def _weights_and_bias(
        self, embedding: Float[Array, "... embed"]
    ) -> tuple[Float[Array, "... weights"], Float[Array, "... bias"]]:
        """Map the invariant embedding to per-sample path weights and biases."""
        hidden = jax.nn.silu(self.hidden(embedding))
        flat = self.readout(hidden)
        return flat[..., : self.num_path_weight], flat[..., self.num_path_weight :]

    def __call__(
        self,
        feature: IrrepsArray,
        embedding: Float[Array, "... embed"],
    ) -> Float[Array, "... 14 14"]:
        """Expand a feature/embedding pair into a dense ``(..., 14, 14)`` block.

        Args:
            feature: Steerable bottleneck feature with ``self.feature_irreps``;
                arbitrary leading (node/edge) axes are supported.
            embedding: Invariant per-sample embedding of width ``self.embed_dim``
                with matching leading axes.

        Returns:
            The dense Hamiltonian block of shape
            ``feature.shape[:-1] + (14, 14)``.

        Raises:
            ValueError: If ``feature.irreps`` does not match ``self.feature_irreps``.
        """
        if feature.irreps != self.feature_irreps:
            raise ValueError(
                f"HamiltonianBlockExpansion expected feature irreps "
                f"{self.feature_irreps!r}, got {feature.irreps!r}"
            )
        leading_shape = feature.array.shape[:-1]
        dtype = feature.array.dtype
        weights, biases = self._weights_and_bias(embedding)
        block = jnp.zeros((*leading_shape, FULL_ORBITALS, FULL_ORBITALS), dtype=dtype)
        for path in self._paths:
            block = block + self._expand_path(path, feature.array, weights, biases, dtype)
        return block

    def _expand_path(
        self,
        path: _ExpansionPath,
        feature_array: Float[Array, "... dim"],
        weights: Float[Array, "... weights"],
        biases: Float[Array, "... bias"],
        dtype: jnp.dtype,
    ) -> Float[Array, "... 14 14"]:
        """Return one path's contribution, padded into the ``(14, 14)`` block.

        Contracts the per-sample-weighted ``L``-chunk with the (constant)
        Clebsch-Gordan tensor (QHNet ``einsum("ijM, ...M -> ...ij")``) and pads the
        ``(2 l_i + 1, 2 l_j + 1)`` sub-block to its ``(row_offset, col_offset)``
        position.
        """
        start, stop = path.feature_slice
        leading_shape = feature_array.shape[:-1]
        # ``(..., mul_L, 2L+1)`` chunk for this input degree.
        chunk = feature_array[..., start:stop].reshape(
            *leading_shape, path.multiplicity, 2 * path.degree + 1
        )
        path_weights = weights[..., path.weight_offset : path.weight_offset + path.multiplicity]
        # Per-sample weighted sum over the multiplicity axis -> ``(..., 2L+1)``.
        feature_vector = jnp.einsum("...w,...wm->...m", path_weights, chunk)
        if path.bias_offset >= 0:
            # Scalar (l_i = l_j = L = 0) sub-block: add the per-sample bias.
            bias = biases[..., path.bias_offset, None]
            feature_vector = feature_vector + bias
        coupling = jnp.asarray(path.coupling_flat, dtype=dtype).reshape(
            2 * path.l_i + 1, 2 * path.l_j + 1, 2 * path.degree + 1
        )
        sub_block = jnp.einsum("ijM,...M->...ij", coupling, feature_vector)
        row_pad = (path.row_offset, FULL_ORBITALS - path.row_offset - (2 * path.l_i + 1))
        col_pad = (path.col_offset, FULL_ORBITALS - path.col_offset - (2 * path.l_j + 1))
        pad_width = [(0, 0)] * len(leading_shape) + [row_pad, col_pad]
        return jnp.pad(sub_block, pad_width)


__all__ = ["HamiltonianBlockExpansion"]

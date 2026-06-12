r"""Heterogeneous-batchable QHNet block Hamiltonian predictor.

A faithful JAX/Flax-NNX port of the QHNet Fock predictor (Yu et al. 2023, "QHNet",
arXiv:2306.04922; reference ``divelab/AIRS``
``OpenDFT/QHBench/QH9/models/QHNet.py``). Given a **flat heterogeneous batch** --
many molecules of differing composition/size concatenated into ``(A, 3)``
positions, ``(A,)`` atomic numbers and a ``(2, E)`` *within-molecule* directed
edge index (already offset so an edge never crosses molecules) -- it emits, for
*every* atom, a fixed ``(14, 14)`` diagonal Fock block and, for *every* directed
edge, a ``(14, 14)`` off-diagonal block (the def2-SVP second-row layout
:data:`~opifex.neural.quantum.hamiltonian._orbital_layout.BLOCK_IRREPS` =
``3x0e + 2x1e + 1x2e``).

Architecture (QHNet ``QHNet.forward``)
--------------------------------------
1. **Embed** atomic numbers into the trunk's scalar channels and lift into the
   parity-correct steerable hidden layout ``Hx0e + Hx1o + Hx2e + Hx3o + Hx4e``.
2. **Message passing**: ``num_interactions``
   :class:`~opifex.neural.quantum.hamiltonian.so2_convolution.SO2ConvolutionLayer`
   layers (the eSCN SO(2)-frame convolution of QHNetV2, arXiv 2506.09398 -- O(L^2)
   per edge vs the dense O(L^3) tensor product; segment/edge-based, hence
   batch-transparent) produce per-atom equivariant features (QHNet's
   ``ConvNetLayer`` stack).
3. **Refinement** (the QHNet expressivity core a NequIP trunk lacks): after the
   ``start_refinement_layer``-th convolution, each subsequent layer feeds the
   parity-relabelled (all-even, matching QHNet's ``hidden_irrep_base``) node
   feature into a
   :class:`~opifex.neural.quantum.hamiltonian._refinement.SelfInteractionLayer`
   (QHNet ``SelfNetLayer`` -- a channel-wise *self* tensor product building the
   diagonal feature ``fii``) and a
   :class:`~opifex.neural.quantum.hamiltonian._refinement.PairInteractionLayer`
   (QHNet ``PairNetLayer`` -- a channel-wise *pair* tensor product over the
   complete edge graph building the off-diagonal feature ``fij``), accumulated
   residually across the stack.
4. **Bottleneck**: ``output_ii`` / ``output_ij``
   (:class:`~opifex.neural.equivariant.EquivariantLinear`) map ``fii`` / ``fij``
   to the even bottleneck layout ``Bx0e + ... + Bx4e``.
5. **Block heads**: the shared
   :class:`~opifex.neural.quantum.hamiltonian._block_expansion.HamiltonianBlockExpansion`
   (QHNet's Wigner-3j ``Expansion``) expands the bottleneck feature + a per-sample
   invariant embedding (the atom embedding for the diagonal head, the concatenated
   pair embedding for the off-diagonal head) into the ``(14, 14)`` block.
6. **Symmetrise** (QHNet ``ret_diagonal = D + D^T``): the diagonal block is made
   symmetric; the off-diagonal block law ``H[i, j] = B_ij + B_ji^T`` is realised at
   assembly because the directed graph carries both ``(i, j)`` and ``(j, i)`` (see
   the *Off-diagonal symmetrisation* note below).

Off-diagonal symmetrisation: QHNet symmetrises with ``ND[transpose_edge].T``,
which requires the per-graph ``transpose_edge_index`` permutation. On a flat
heterogeneous batch that permutation is composition-dependent. Rather than carry
it, this core emits the **raw** per-edge block and lets the consumer symmetrise
the *assembled* matrix (``H = H~ + H~^T``); because a complete molecular graph
contains both directed edges, ``H~[i, j] = B_ij`` and ``H~[j, i] = B_ji`` so the
symmetrised ``H[i, j] = B_ij + B_ji^T`` reproduces QHNet's law.
:meth:`assemble_matrix` performs exactly this symmetrisation for the single
molecule inference path.

All static routing (irreps, offsets, slices) lives in hashable Python
tuples/ints; no ``jax``/``numpy`` array is stored as a plain ``nnx.Module``
attribute, so the module is ``jit``/``grad``/``vmap`` clean and valid across
repeated ``nnx.jit`` calls (the known opifex array-valued-metadata gotcha).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int  # noqa: TC002

from opifex.neural.atomistic.backbones._message_passing import edge_geometry
from opifex.neural.atomistic.backbones.nequip import NequIPConfig
from opifex.neural.equivariant import (
    BesselBasis,
    cosine_cutoff,
    EquivariantLinear,
    Irrep,
    Irreps,
    IrrepsArray,
    spherical_harmonics,
)
from opifex.neural.quantum.hamiltonian._block_expansion import HamiltonianBlockExpansion
from opifex.neural.quantum.hamiltonian._orbital_layout import (
    atom_orbital_counts,
    block_validity_mask,
)
from opifex.neural.quantum.hamiltonian._refinement import (
    PairInteractionLayer,
    SelfInteractionLayer,
)
from opifex.neural.quantum.hamiltonian.so2_convolution import SO2ConvolutionLayer


logger = logging.getLogger(__name__)

_MAX_ATOMIC_NUMBER = 118
"""Highest supported nuclear charge; the embedding table has one row per Z=0..118."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockHamiltonianConfig:
    """Hyper-parameters of a :class:`BlockHamiltonianPredictor`.

    Defaults sit well below the QHNet reference (hidden multiplicity ~128, ``sh_lmax``
    4, 5 interactions) so the documented defaults stay test-fast; production /
    training configs should raise ``hidden_irreps`` to a uniform-multiplicity
    ``Hx0e + Hx1o + Hx2e + Hx3o + Hx4e`` (``sh_lmax`` 4) toward the reference.

    Attributes:
        hidden_irreps: Steerable layout of the per-atom hidden / message-passing
            features (QHNet's ``hidden_irrep``). Must be **uniform multiplicity**
            across all degrees (the channel-wise refinement tensor products require
            it) and contain a ``0e`` scalar channel.
        sh_lmax: Maximum spherical-harmonic degree of the edge embedding.
        num_interactions: Number of NequIP convolution layers (QHNet's
            ``num_gnn_layers`` ``ConvNetLayer`` stack).
        start_refinement_layer: Convolution index after which the self / pair
            refinement layers run (QHNet's ``start_layer``); refinement happens for
            every layer with index strictly greater than it, so there are
            ``num_interactions - 1 - start_refinement_layer`` refinement layers.
        bottleneck_multiplicity: Multiplicity of the even bottleneck feeding the
            block heads (QHNet's ``bottle_hidden_size``).
        num_radial_basis: Number of Bessel radial-basis functions.
        radial_hidden_dim: Hidden width of the radial network MLP.
        cutoff: Connection / cutoff radius ``r_c`` (Bohr). Defaults large so the
            complete within-molecule graph is retained.
        average_num_neighbors: Constant ``sqrt`` normaliser for the aggregated
            message (NequIP's internal normalisation).
        embed_dim: Width of the invariant embedding driving the block head's
            per-sample weight/bias MLP.
        block_mlp_hidden_dim: Hidden width of the block head's weight/bias MLP.
        pair_weight_hidden_dim: Hidden width of the pair layer's per-edge weight MLPs.
    """

    hidden_irreps: str = "16x0e + 16x1o + 16x2e + 16x3o + 16x4e"
    sh_lmax: int = 4
    num_interactions: int = 3
    start_refinement_layer: int = 0
    bottleneck_multiplicity: int = 16
    num_radial_basis: int = 8
    radial_hidden_dim: int = 64
    cutoff: float = 20.0
    average_num_neighbors: float = 1.0
    embed_dim: int = 64
    block_mlp_hidden_dim: int = 128
    pair_weight_hidden_dim: int = 64

    def to_nequip(self) -> NequIPConfig:
        """Return the matching :class:`NequIPConfig` for the reused conv layers."""
        return NequIPConfig(
            hidden_irreps=self.hidden_irreps,
            sh_lmax=self.sh_lmax,
            num_interactions=self.num_interactions,
            num_radial_basis=self.num_radial_basis,
            radial_hidden_dim=self.radial_hidden_dim,
            cutoff=self.cutoff,
            average_num_neighbors=self.average_num_neighbors,
        )


def _even_irreps(irreps: Irreps) -> Irreps:
    """Return ``irreps`` with every block's parity relabelled to even.

    The refinement layers and the block heads operate in QHNet's all-even
    ``hidden_irrep_base`` space; relabelling the trunk's odd irreps (``1o``,
    ``3o``) to even keeps the per-block dimensions and makes the head
    SO(3)-equivariant (parity is forgotten, exactly as in the reference).
    """
    return Irreps(tuple((mul, Irrep(irrep.l, 1)) for mul, irrep in irreps.blocks))


def _bottleneck_irreps(hidden_irreps: Irreps, multiplicity: int) -> Irreps:
    """Return the even bottleneck layout: one ``multiplicity x le`` block per degree.

    The :class:`HamiltonianBlockExpansion` keys its expansion paths on the input
    *degree* ``l`` only, so the bottleneck carries every degree present in
    ``hidden_irreps`` once, at ``multiplicity`` (even parity, ascending degree).
    """
    degrees = sorted({irrep.l for _, irrep in hidden_irreps.blocks})
    return Irreps(tuple((multiplicity, Irrep(degree, 1)) for degree in degrees))


class BlockHamiltonianPredictor(nnx.Module):
    r"""Heterogeneous-batchable per-atom / per-edge QHNet Fock block predictor.

    Consumes a flat concatenated batch (``atomic_numbers``, ``positions``,
    within-molecule ``edge_index``) and emits a fixed ``(14, 14)`` diagonal block
    per atom and ``(14, 14)`` off-diagonal block per directed edge. Reuses the
    NequIP convolution trunk (segment-based, hence batch-transparent), the QHNet
    self / pair interaction refinement layers and the shared
    :class:`HamiltonianBlockExpansion` head (reference ``divelab/AIRS``
    ``OpenDFT/QHBench/QH9/models/QHNet.py``).

    Args:
        config: Hyper-parameters. Defaults to :class:`BlockHamiltonianConfig`.
        rngs: Random number generators (keyword-only) seeding all weights.

    Raises:
        ValueError: If ``config.hidden_irreps`` carries no ``0e`` scalar channel,
            is not uniform multiplicity, or leaves no room for a refinement layer.
    """

    def __init__(
        self,
        *,
        config: BlockHamiltonianConfig | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the embedding, conv trunk, refinement stack and block heads."""
        super().__init__()
        self.config = config if config is not None else BlockHamiltonianConfig()
        self.hidden_irreps = Irreps(self.config.hidden_irreps)
        self._validate_config()
        self._num_scalars = sum(mul for mul, ir in self.hidden_irreps.blocks if ir.l == 0)
        nequip_config = self.config.to_nequip()
        self.sh_irreps = spherical_harmonics(self.config.sh_lmax, jnp.zeros((1, 3))).irreps

        # --- embedding + lift into the parity-correct steerable hidden layout ---
        self.embedding = nnx.Embed(
            num_embeddings=_MAX_ATOMIC_NUMBER + 1, features=self._num_scalars, rngs=rngs
        )
        self._embedding_irreps = Irreps(f"{self._num_scalars}x0e")
        self.embedding_linear = EquivariantLinear(
            self._embedding_irreps, self.hidden_irreps, rngs=rngs
        )
        self.radial_basis = BesselBasis(self.config.num_radial_basis, self.config.cutoff)

        # --- reused NequIP convolution trunk (segment/edge-based -> batch clean) ---
        self.layers = nnx.List(
            [
                SO2ConvolutionLayer(
                    self.hidden_irreps, self.config.sh_lmax, nequip_config, rngs=rngs
                )
                for _ in range(self.config.num_interactions)
            ]
        )

        # --- QHNet refinement stack (all-even base) ---
        self._even_base = _even_irreps(self.hidden_irreps)
        self._num_refinement = self.config.num_interactions - 1 - self.config.start_refinement_layer
        self.self_layers = nnx.List(
            [SelfInteractionLayer(self._even_base, rngs=rngs) for _ in range(self._num_refinement)]
        )
        self.pair_layers = nnx.List(
            [
                PairInteractionLayer(
                    self._even_base,
                    edge_radial_dim=self.config.num_radial_basis,
                    weight_hidden_dim=self.config.pair_weight_hidden_dim,
                    rngs=rngs,
                )
                for _ in range(self._num_refinement)
            ]
        )

        # --- even bottleneck + shared block-expansion heads ---
        self._feature_irreps = _bottleneck_irreps(
            self.hidden_irreps, self.config.bottleneck_multiplicity
        )
        self.output_ii = EquivariantLinear(self._even_base, self._feature_irreps, rngs=rngs)
        self.output_ij = EquivariantLinear(self._even_base, self._feature_irreps, rngs=rngs)
        self.diagonal_head = HamiltonianBlockExpansion(
            feature_irreps=self._feature_irreps,
            embed_dim=self.config.embed_dim,
            mlp_hidden_dim=self.config.block_mlp_hidden_dim,
            rngs=rngs,
        )
        self.off_diagonal_head = HamiltonianBlockExpansion(
            feature_irreps=self._feature_irreps,
            embed_dim=2 * self.config.embed_dim,
            mlp_hidden_dim=self.config.block_mlp_hidden_dim,
            rngs=rngs,
        )
        # Project the (element-identity) atom embedding to the block head's width.
        self.node_embed = nnx.Linear(self._num_scalars, self.config.embed_dim, rngs=rngs)

    def _validate_config(self) -> None:
        """Validate the hidden layout supports the refinement / head requirements."""
        if self._num_scalars_for(self.hidden_irreps) == 0:
            raise ValueError(
                f"hidden_irreps must contain a 0e scalar channel to read out, got "
                f"{self.hidden_irreps!r}"
            )
        multiplicities = {mul for mul, _ in self.hidden_irreps.blocks}
        if len(multiplicities) != 1:
            raise ValueError(
                f"hidden_irreps must be uniform multiplicity for the channel-wise "
                f"refinement tensor products, got {self.hidden_irreps!r}"
            )
        if self.config.num_interactions - 1 - self.config.start_refinement_layer < 1:
            raise ValueError(
                f"need at least one refinement layer: num_interactions="
                f"{self.config.num_interactions} with start_refinement_layer="
                f"{self.config.start_refinement_layer} leaves none"
            )

    @staticmethod
    def _num_scalars_for(irreps: Irreps) -> int:
        """Return the total ``l = 0`` multiplicity of ``irreps``."""
        return sum(mul for mul, irrep in irreps.blocks if irrep.l == 0)

    def _encode(
        self,
        atomic_numbers: Int[Array, " n_atoms"],
        positions: Float[Array, "n_atoms 3"],
        edge_index: Int[Array, "2 n_edges"],
    ) -> tuple[IrrepsArray, IrrepsArray, Float[Array, "n_atoms embed"]]:
        """Run the trunk + refinement; return the diagonal/off-diagonal features.

        Returns:
            ``(diagonal_feature, off_diagonal_feature, node_embedding)`` where the
            first is per-atom ``fii`` (even bottleneck), the second per-edge ``fij``
            (even bottleneck) and the third the per-atom invariant embedding driving
            the block heads.
        """
        n_atoms = atomic_numbers.shape[0]
        senders, receivers = edge_index[0], edge_index[1]
        geometry = edge_geometry(positions, (senders, receivers))
        lengths = geometry.lengths[:, 0]
        radial = self.radial_basis(lengths)
        envelope = (cosine_cutoff(lengths, self.config.cutoff) * geometry.mask[:, 0])[:, None]
        radial_envelope = radial * envelope

        atom_embedding = self.embedding(atomic_numbers)
        node_features = self.embedding_linear(IrrepsArray(self._embedding_irreps, atom_embedding))
        diagonal_feature: IrrepsArray | None = None
        off_diagonal_feature: IrrepsArray | None = None
        refinement_index = 0
        for layer_index, layer in enumerate(self.layers):
            node_features = layer(node_features, geometry, radial, envelope, n_atoms)
            if layer_index > self.config.start_refinement_layer:
                node_even = IrrepsArray(self._even_base, node_features.array)
                diagonal_feature = self.self_layers[refinement_index](node_even, diagonal_feature)
                off_diagonal_feature = self.pair_layers[refinement_index](
                    node_even, senders, receivers, radial_envelope, off_diagonal_feature
                )
                refinement_index += 1

        node_embedding = self.node_embed(atom_embedding)
        if diagonal_feature is None or off_diagonal_feature is None:
            raise RuntimeError(
                "no refinement layer ran; _validate_config should guarantee at least one"
            )
        return (
            self.output_ii(diagonal_feature),
            self.output_ij(off_diagonal_feature),
            node_embedding,
        )

    def __call__(
        self,
        atomic_numbers: Int[Array, " n_atoms"],
        positions: Float[Array, "n_atoms 3"],
        edge_index: Int[Array, "2 n_edges"],
        node_batch: Int[Array, " n_atoms"] | None = None,
    ) -> dict[str, Float[Array, "... 14 14"]]:
        """Predict per-atom diagonal and per-edge off-diagonal Fock blocks.

        Args:
            atomic_numbers: Flat ``(A,)`` atomic numbers of the concatenated batch.
            positions: Flat ``(A, 3)`` atomic positions (Bohr).
            edge_index: ``(2, E)`` within-molecule directed ``(senders, receivers)``
                edge index, already offset so edges never cross molecules
                (``-1`` padding is allowed via the
                :func:`~...backbones._message_passing.edge_geometry` contract).
            node_batch: Optional ``(A,)`` segment id per atom. Accepted for
                completeness; the forward does not need it because the conv trunk
                and refinement scatter only over the within-molecule ``edge_index``.

        Returns:
            ``{"diagonal_blocks": (A, 14, 14), "off_diagonal_blocks": (E, 14, 14)}``;
            the diagonal blocks are symmetrised (QHNet ``D + D^T``) and the
            off-diagonal blocks are raw (symmetrised at assembly, see module docs).
        """
        del node_batch  # Not required: edges are within-molecule (documented).
        senders, receivers = edge_index[0], edge_index[1]
        diagonal_feature, off_diagonal_feature, node_embedding = self._encode(
            atomic_numbers, positions, edge_index
        )

        # --- diagonal blocks (per atom) ---
        diagonal = self.diagonal_head(diagonal_feature, node_embedding)
        diagonal = diagonal + jnp.swapaxes(diagonal, -1, -2)

        # --- off-diagonal blocks (per directed edge) ---
        # QHNet pair embedding: cat([node_attr[dst], node_attr[src]]) = [receiver, sender].
        pair_embedding = jnp.concatenate(
            [node_embedding[receivers], node_embedding[senders]], axis=-1
        )
        off_diagonal = self.off_diagonal_head(off_diagonal_feature, pair_embedding)

        return {"diagonal_blocks": diagonal, "off_diagonal_blocks": off_diagonal}

    def assemble_matrix(
        self,
        diagonal_blocks: Float[Array, "n_atoms 14 14"],
        off_diagonal_blocks: Float[Array, "n_edges 14 14"],
        atomic_numbers: Int[Array, " n_atoms"],
        edge_index: Int[Array, "2 n_edges"],
    ) -> Float[Array, "n_ao n_ao"]:
        r"""Assemble a single molecule's dense, symmetric ``(n_ao, n_ao)`` matrix.

        Masks each block to its element's valid AO slots
        (:func:`~...._orbital_layout.block_validity_mask`) and scatters it into the
        dense matrix at the per-atom AO offsets
        (:func:`~...._orbital_layout.atom_orbital_counts`). The off-diagonal blocks
        are written at both ``(i, j)`` and ``(j, i)``; the directed graph carries
        both edges, so the QHNet off-diagonal law
        ``H[i, j] = B_ij + B_ji^T`` is realised by the final symmetrisation
        ``H = H~ + H~^T``.

        This is a host-side inference helper for a *single* molecule (it builds a
        dense matrix and uses Python sizing), not part of the batched forward.

        Args:
            diagonal_blocks: ``(A, 14, 14)`` per-atom blocks from :meth:`__call__`.
            off_diagonal_blocks: ``(E, 14, 14)`` per-edge blocks.
            atomic_numbers: ``(A,)`` atomic numbers of the single molecule.
            edge_index: ``(2, E)`` directed edge index of the single molecule.

        Returns:
            The symmetric dense AO matrix of shape ``(n_ao, n_ao)``.
        """
        counts = atom_orbital_counts(atomic_numbers)
        offsets = jnp.concatenate([jnp.zeros((1,), counts.dtype), jnp.cumsum(counts)])
        n_ao = int(offsets[-1])
        n_atoms = int(atomic_numbers.shape[0])
        matrix = jnp.zeros((n_ao, n_ao), dtype=diagonal_blocks.dtype)

        # Diagonal blocks: masked, packed to the element's AO slots, placed on-site.
        diag_mask = block_validity_mask(atomic_numbers)
        for atom in range(n_atoms):
            start = int(offsets[atom])
            stop = int(offsets[atom + 1])
            packed = _pack_block(diagonal_blocks[atom], diag_mask[atom], stop - start)
            matrix = matrix.at[start:stop, start:stop].set(packed)

        # Off-diagonal blocks: row = receiver element, col = sender element.
        senders = edge_index[0]
        receivers = edge_index[1]
        n_edges = int(edge_index.shape[1])
        off_mask = block_validity_mask(atomic_numbers[receivers], atomic_numbers[senders])
        for edge in range(n_edges):
            row = int(receivers[edge])
            col = int(senders[edge])
            if row < 0 or col < 0:
                continue
            row_start, row_stop = int(offsets[row]), int(offsets[row + 1])
            col_start, col_stop = int(offsets[col]), int(offsets[col + 1])
            packed = _pack_block(
                off_diagonal_blocks[edge],
                off_mask[edge],
                (row_stop - row_start, col_stop - col_start),
            )
            matrix = matrix.at[row_start:row_stop, col_start:col_stop].set(packed)

        return matrix + matrix.T


def _pack_block(
    block: Float[Array, "14 14"],
    mask: Float[Array, "14 14"],
    shape: int | tuple[int, int],
) -> Float[Array, "rows cols"]:
    """Select the valid AO rows/cols of a ``(14, 14)`` block into a dense sub-block.

    Args:
        block: The full ``(14, 14)`` predicted block.
        mask: The ``(14, 14)`` boolean AO validity mask for this atom/pair.
        shape: Output ``(rows, cols)`` (an int means a square diagonal block).

    Returns:
        The packed dense sub-block of the requested shape.
    """
    rows, cols = (shape, shape) if isinstance(shape, int) else shape
    row_valid = jnp.any(mask, axis=1)
    col_valid = jnp.any(mask, axis=0)
    row_indices = jnp.nonzero(row_valid, size=rows)[0]
    col_indices = jnp.nonzero(col_valid, size=cols)[0]
    return block[jnp.ix_(row_indices, col_indices)]


__all__ = ["BlockHamiltonianConfig", "BlockHamiltonianPredictor"]

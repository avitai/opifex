r"""Heterogeneous-batchable QHNet-style block Hamiltonian predictor.

This is the *segment/concatenated* core of a QHNet Fock predictor (Yu et al. 2023,
"QHNet", arXiv:2306.04922; reference ``divelab/AIRS``
``OpenDFT/QHBench/QH9/models/QHNet.py``). Given a **flat heterogeneous batch** --
many molecules of differing composition/size concatenated into ``(A, 3)``
positions, ``(A,)`` atomic numbers and a ``(2, E)`` *within-molecule* directed
edge index (already offset so an edge never crosses molecules) -- it emits, for
*every* atom, a fixed ``(14, 14)`` diagonal Fock block and, for *every* directed
edge, a ``(14, 14)`` off-diagonal block.

Because the blocks are fixed-size (the def2-SVP second-row layout
:data:`~opifex.neural.quantum.hamiltonian._orbital_layout.BLOCK_IRREPS` =
``3x0e + 2x1e + 1x2e``) and the NequIP convolution scatters messages **only over
within-molecule edges**, one compiled forward (for a given padded ``(A, E)``)
runs over *any* concatenation of molecules -- no per-composition assembly plan and
no per-molecule recompile. This replaces the old per-composition dense assembly of
:class:`~opifex.neural.quantum.hamiltonian.predictor.HamiltonianPredictor`.

Architecture (QHNet forward, ``divelab/AIRS`` ``QHNet.forward``)
---------------------------------------------------------------
1. **Embed** atomic numbers into the trunk's scalar channels and lift into the
   steerable hidden layout (reuses :class:`~...nequip.NequIP`'s embedding pattern).
2. **Message passing**: ``num_interactions`` reused
   :class:`~opifex.neural.atomistic.backbones.nequip._ConvolutionLayer` layers
   (the NequIP Clebsch-Gordan TP convolution) produce per-atom equivariant
   features. The conv layer is segment/edge-based -- it scatters via
   :func:`~opifex.neural.equivariant.scatter_sum` over ``geometry.receivers`` --
   so it runs **unchanged** on the concatenated multi-molecule graph (QHNet's
   ``ConvNetLayer`` stack).
3. **Bottleneck**: an :class:`~opifex.neural.equivariant.EquivariantLinear` maps
   the per-atom features to the diagonal head's steerable feature (QHNet's
   ``output_ii``), and a second one to the off-diagonal head's feature
   (``output_ij``).
4. **Diagonal head** (per atom): the shared
   :class:`~opifex.neural.quantum.hamiltonian._block_expansion.HamiltonianBlockExpansion`
   over the node feature + the per-atom invariant (``0e``) embedding.
5. **Off-diagonal head** (per directed edge): the *same* block-head module over a
   steerable pair feature (sender feature tensored with the edge spherical
   harmonics, radially modulated, plus a mixed receiver feature -- the NequIP edge
   message reused) + a per-edge invariant embedding formed by concatenating the
   sender and receiver scalar embeddings (QHNet's
   ``node_pair_embedding = cat([node_attr[dst], node_attr[src]])``).
6. **Symmetrise** (QHNet ``ret_diagonal = D + D^T``): the diagonal block is made
   symmetric; the off-diagonal block law ``ND + ND[transpose_edge].T`` is realised
   implicitly because the directed graph contains both ``(i, j)`` and ``(j, i)``
   and :meth:`assemble_matrix`/downstream symmetrise the assembled matrix (see the
   *Off-diagonal symmetrisation* note below).

Bounded scope (vs. the full QHNet)
----------------------------------
The QHNet ``SelfNetLayer`` / ``PairNetLayer`` on-site/pair *refinement* blocks are
intentionally **not** included here: this module is the heterogeneous-batch *core*
(embed -> conv trunk -> bottleneck -> shared block head), the part that makes
fixed-size blocks trivially batchable. The refinement layers are an orthogonal
expressivity upgrade (a documented follow-up), not part of the batching contract,
and the equivariance / batch-consistency guarantees here do not depend on them.

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
from opifex.neural.atomistic.backbones.nequip import _ConvolutionLayer, NequIPConfig
from opifex.neural.equivariant import (
    BesselBasis,
    cosine_cutoff,
    EquivariantLinear,
    FullyConnectedTensorProduct,
    Irrep,
    Irreps,
    IrrepsArray,
    spherical_harmonics,
)
from opifex.neural.equivariant._assembly import from_chunks
from opifex.neural.quantum.hamiltonian._block_expansion import HamiltonianBlockExpansion
from opifex.neural.quantum.hamiltonian._orbital_layout import (
    atom_orbital_counts,
    block_validity_mask,
)


logger = logging.getLogger(__name__)

_MAX_ATOMIC_NUMBER = 118
"""Highest supported nuclear charge; the embedding table has one row per Z=0..118."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockHamiltonianConfig:
    """Hyper-parameters of a :class:`BlockHamiltonianPredictor`.

    Defaults sit near QHNet (hidden multiplicity ~128, ``sh_lmax`` 4, 5
    interactions) but are reduced here so the documented defaults stay test-fast;
    production configs should raise them toward the reference.

    Attributes:
        hidden_irreps: Steerable layout of the per-atom hidden / message-passing
            features (QHNet's ``hidden_irrep``). Must contain a ``0e`` scalar
            channel to read out as the invariant embedding.
        sh_lmax: Maximum spherical-harmonic degree of the edge embedding.
        num_interactions: Number of NequIP convolution layers (QHNet's
            ``num_gnn_layers`` ``ConvNetLayer`` stack).
        num_radial_basis: Number of Bessel radial-basis functions.
        radial_hidden_dim: Hidden width of the radial network MLP.
        cutoff: Connection / cutoff radius ``r_c`` (Bohr). Defaults large so the
            complete within-molecule graph is retained.
        average_num_neighbors: Constant ``sqrt`` normaliser for the aggregated
            message (NequIP's internal normalisation).
        embed_dim: Width of the invariant embedding driving the block head's
            per-sample weight/bias MLP.
        block_mlp_hidden_dim: Hidden width of the block head's weight/bias MLP.
    """

    hidden_irreps: str = "32x0e + 16x1o + 8x2e"
    sh_lmax: int = 2
    num_interactions: int = 3
    num_radial_basis: int = 8
    radial_hidden_dim: int = 64
    cutoff: float = 20.0
    average_num_neighbors: float = 1.0
    embed_dim: int = 64
    block_mlp_hidden_dim: int = 128

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


def _bottleneck_feature_irreps(hidden_irreps: Irreps, multiplicity: int) -> Irreps:
    """Return the block-head feature layout: one ``mult x le`` block per degree.

    The :class:`HamiltonianBlockExpansion` keys its expansion paths on the input
    *degree* ``l`` only (parity is irrelevant to the rotation-only block law), so
    the bottleneck carries every degree present in ``hidden_irreps`` once, at the
    given multiplicity. Degrees absent from the trunk simply leave their
    Clebsch-Gordan sub-block components at zero (still equivariant).

    Args:
        hidden_irreps: The trunk's per-atom feature layout.
        multiplicity: Multiplicity assigned to every bottleneck degree.

    Returns:
        The bottleneck / block-head feature :class:`Irreps` (even-parity, ascending
        degree), e.g. ``8x0e + 8x1e + 8x2e``.
    """
    degrees = sorted({irrep.l for _, irrep in hidden_irreps.blocks})
    blocks = tuple((multiplicity, Irrep(degree, 1)) for degree in degrees)
    return Irreps(blocks)


class BlockHamiltonianPredictor(nnx.Module):
    r"""Heterogeneous-batchable per-atom / per-edge QHNet Fock block predictor.

    Consumes a flat concatenated batch (``atomic_numbers``, ``positions``,
    within-molecule ``edge_index``) and emits a fixed ``(14, 14)`` diagonal block
    per atom and ``(14, 14)`` off-diagonal block per directed edge. Reuses the
    NequIP convolution trunk (segment-based, hence batch-transparent) and the
    shared :class:`HamiltonianBlockExpansion` head for both the diagonal (node
    feature + node embedding) and off-diagonal (edge feature + concatenated pair
    embedding) blocks (reference ``divelab/AIRS``
    ``OpenDFT/QHBench/QH9/models/QHNet.py``).

    Args:
        config: Hyper-parameters. Defaults to :class:`BlockHamiltonianConfig`.
        rngs: Random number generators (keyword-only) seeding all weights.

    Raises:
        ValueError: If ``config.hidden_irreps`` carries no ``0e`` scalar channel
            (needed for the invariant embedding driving the block head).
    """

    def __init__(
        self,
        *,
        config: BlockHamiltonianConfig | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the embedding, conv trunk, bottlenecks and the shared block head."""
        super().__init__()
        self.config = config if config is not None else BlockHamiltonianConfig()
        self.hidden_irreps = Irreps(self.config.hidden_irreps)
        self._num_scalars = sum(mul for mul, ir in self.hidden_irreps.blocks if ir.l == 0)
        if self._num_scalars == 0:
            raise ValueError(
                f"hidden_irreps must contain a 0e scalar channel to read out, got "
                f"{self.hidden_irreps!r}"
            )
        nequip_config = self.config.to_nequip()
        self.sh_irreps = spherical_harmonics(self.config.sh_lmax, jnp.zeros((1, 3))).irreps

        # --- embedding + lift into the steerable hidden layout (NequIP pattern) ---
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
                _ConvolutionLayer(
                    self.hidden_irreps,
                    self.sh_irreps,
                    self.hidden_irreps,
                    nequip_config,
                    rngs=rngs,
                )
                for _ in range(self.config.num_interactions)
            ]
        )

        # --- bottleneck features for the diagonal and off-diagonal block heads ---
        block_multiplicity = self._num_scalars
        self._feature_irreps = _bottleneck_feature_irreps(self.hidden_irreps, block_multiplicity)
        self.output_ii = EquivariantLinear(self.hidden_irreps, self._feature_irreps, rngs=rngs)
        self.output_ij = EquivariantLinear(self.hidden_irreps, self._feature_irreps, rngs=rngs)

        # --- off-diagonal edge feature builder (NequIP edge message, reused) ---
        self.edge_tensor_product = FullyConnectedTensorProduct(
            self.hidden_irreps, self.sh_irreps, self.hidden_irreps, rngs=rngs
        )
        self.edge_radial = nnx.Linear(
            self.config.num_radial_basis,
            self.hidden_irreps.num_irreps,
            use_bias=False,
            rngs=rngs,
        )
        self.edge_receiver_mix = EquivariantLinear(
            self.hidden_irreps, self.hidden_irreps, rngs=rngs
        )

        # --- one shared block-expansion head: diagonal embed_dim, off-diag 2x ---
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
        # Project the trunk scalars to the block head's embedding width.
        self.node_embed = nnx.Linear(self._num_scalars, self.config.embed_dim, rngs=rngs)

    def _trunk(
        self,
        atomic_numbers: Int[Array, " n_atoms"],
        positions: Float[Array, "n_atoms 3"],
        edge_index: Int[Array, "2 n_edges"],
    ) -> tuple[IrrepsArray, IrrepsArray, Float[Array, "n_edges num_radial"]]:
        """Embed + run the conv trunk; return node features and reusable edge data.

        Returns:
            ``(node_features, edge_sh, radial_envelope)`` where ``node_features`` is
            the per-atom steerable feature, ``edge_sh`` the per-edge spherical
            harmonics and ``radial_envelope`` the cutoff-enveloped radial basis.
        """
        n_atoms = atomic_numbers.shape[0]
        graph = (edge_index[0], edge_index[1])
        geometry = edge_geometry(positions, graph)
        lengths = geometry.lengths[:, 0]
        radial = self.radial_basis(lengths)
        envelope = (cosine_cutoff(lengths, self.config.cutoff) * geometry.mask[:, 0])[:, None]
        edge_sh = spherical_harmonics(self.sh_irreps, geometry.vectors)
        scalar_embedding = IrrepsArray(self._embedding_irreps, self.embedding(atomic_numbers))
        node_features = self.embedding_linear(scalar_embedding)
        for layer in self.layers:
            node_features = layer(node_features, edge_sh, geometry, radial, envelope, n_atoms)
        return node_features, edge_sh, radial * envelope

    def _edge_feature(
        self,
        node_features: IrrepsArray,
        edge_index: Int[Array, "2 n_edges"],
        edge_sh: IrrepsArray,
        radial_envelope: Float[Array, "n_edges num_radial"],
    ) -> IrrepsArray:
        """Build the steerable per-edge (pair) feature (QHNet pair message reuse).

        The sender feature is tensored with the edge spherical harmonics and
        radially modulated (the NequIP edge message), and a mixed receiver feature
        is added -- the same construction used by
        :class:`~opifex.neural.quantum.hamiltonian.predictor.HamiltonianPredictor`.
        """
        senders, receivers = edge_index[0], edge_index[1]
        sender_features = IrrepsArray(node_features.irreps, node_features.array[senders])
        message = self.edge_tensor_product(sender_features, edge_sh)
        weights = self.edge_radial(radial_envelope)
        message = self._apply_radial_weights(message, weights)
        receiver_features = IrrepsArray(node_features.irreps, node_features.array[receivers])
        mixed = self.edge_receiver_mix(receiver_features)
        return IrrepsArray(self.hidden_irreps, message.array + mixed.array)

    def _apply_radial_weights(
        self, message: IrrepsArray, weights: Float[Array, "n_edges num_irreps"]
    ) -> IrrepsArray:
        """Scale each output multiplicity of ``message`` by its radial weight."""
        scaled: list[Array | None] = []
        cursor = 0
        for (mul, _), chunk in zip(message.irreps.blocks, message.chunks, strict=True):
            block_weights = weights[:, cursor : cursor + mul]
            scaled.append(chunk * block_weights[..., None])
            cursor += mul
        return from_chunks(message.irreps, scaled, message.array.shape[:-1], message.array.dtype)

    def _node_embedding(self, node_features: IrrepsArray) -> Float[Array, "n_atoms embed"]:
        """Project the per-atom invariant scalar channels to the head embedding."""
        scalars = node_features.chunks[0].reshape(node_features.array.shape[0], self._num_scalars)
        return self.node_embed(scalars)

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
                scatters only over the within-molecule ``edge_index``.

        Returns:
            ``{"diagonal_blocks": (A, 14, 14), "off_diagonal_blocks": (E, 14, 14)}``;
            the diagonal blocks are symmetrised (QHNet ``D + D^T``) and the
            off-diagonal blocks are raw (symmetrised at assembly, see module docs).
        """
        del node_batch  # Not required: edges are within-molecule (documented).
        node_features, edge_sh, radial_envelope = self._trunk(atomic_numbers, positions, edge_index)
        node_embedding = self._node_embedding(node_features)

        # --- diagonal blocks (per atom) ---
        diagonal_feature = self.output_ii(node_features)
        diagonal = self.diagonal_head(diagonal_feature, node_embedding)
        diagonal = diagonal + jnp.swapaxes(diagonal, -1, -2)

        # --- off-diagonal blocks (per directed edge) ---
        edge_feature = self._edge_feature(node_features, edge_index, edge_sh, radial_envelope)
        off_feature = self.output_ij(edge_feature)
        senders, receivers = edge_index[0], edge_index[1]
        # QHNet pair embedding: cat([node_attr[dst], node_attr[src]]) = [receiver, sender].
        pair_embedding = jnp.concatenate(
            [node_embedding[receivers], node_embedding[senders]], axis=-1
        )
        off_diagonal = self.off_diagonal_head(off_feature, pair_embedding)

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

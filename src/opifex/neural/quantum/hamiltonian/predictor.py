r"""Equivariant DFT Hamiltonian/overlap predictor (QHNet-style).

:class:`HamiltonianPredictor` predicts the dense atomic-orbital Fock/Hamiltonian
matrix ``H`` (or the overlap matrix ``S``) of a molecular system, in opifex's
exact shell/AO ordering, with guaranteed E(3) equivariance
``H(R r) = D(R) H(r) D(R)^T`` (Yu et al. 2023, "QHNet", arXiv:2306.04922;
reference ``divelab/AIRS`` ``OpenDFT/QHBench/QH9/models/QHNet.py``).

Architecture (every primitive reused from opifex's Q0 kit):

#. **Steerable trunk** -- the NequIP tensor-product message passing
   (:class:`opifex.neural.atomistic.backbones.nequip._ConvolutionLayer`) produces
   a *full steerable* per-atom feature :class:`IrrepsArray` (not just the scalar
   readout the energy contract uses).
#. **Node head (diagonal ``H_ii``)** -- for every intra-atom shell pair
   ``(l_i, l_j)`` an equivariant :class:`PairExpansion` turns the receiver atom's
   node feature into the ``(2 l_i + 1) x (2 l_j + 1)`` on-site block.
#. **Edge head (off-diagonal ``H_ij``)** -- for every directed edge a mixed
   sender/receiver edge feature (a radially-modulated tensor product of the
   sender's features with the edge spherical harmonics, combined with the
   receiver's features) is expanded per shell-pair type into the off-site block.
#. **Scatter** -- blocks are written into the dense matrix at their static
   ``(row_offset, col_offset)`` AO positions.
#. **Symmetrize** -- ``H = H~ + H~^T`` makes the matrix Hermitian; the
   off-diagonal contribution naturally realises ``H_ij = H_ji^T`` because the
   directed graph contains both ``(i, j)`` and ``(j, i)`` and the transpose of the
   ``(j, i)`` block lands on the ``(i, j)`` sub-matrix (QHNet's
   ``transpose_edge_index`` symmetrisation).

The per-block einsum shapes are static (one shared :class:`PairExpansion` per
angular-momentum pair type ``(l_i, l_j)``), so the predictor is ``jit``/``grad``/
``vmap`` clean over geometry. Switching to a different molecule (different atom or
orbital count) only swaps the static shell-pair plan via :meth:`rebind`; the
learnable weights are shared across molecules.

Basis note (STO-3G, the validation target): opifex's STO-3G covers H/C/N/O, whose
shells are ``s`` and ``p`` only (``l in {0, 1}``). For ``s`` and ``p`` the
Cartesian AO components coincide with the real spherical-harmonic (irrep)
components in the identical ``(x, y, z)`` order used by
:func:`opifex.geometry.algebra.wigner.wigner_d`, so **no Cartesian-to-spherical
transform is needed** and the predicted blocks land directly in opifex's AO
ordering.

Documented follow-ups (not built here):

* **def2-SVP / ``d``-orbitals (``l >= 2``) + the QH9 benchmark** require a
  Cartesian-to-spherical solid-harmonic transform on the predicted blocks (the
  ``GaussianShell.n_cartesian`` Cartesian count exceeds ``2 l + 1`` for ``l >=
  2``). That belongs with the basis module (currently being rewritten) and is
  intentionally out of scope.
* **QHNetV2 SO(2)-frame convolution** (Yu et al. 2023, arXiv:2306.04922 v2;
  Passaro & Zitnick 2023, "eSCN", arXiv:2302.03655): rotating each edge into a
  local frame aligned with its direction reduces the ``SO(3)`` tensor product to
  cheaper ``SO(2)`` operations, the main scalability lever for large bases. The
  present trunk uses the full ``SO(3)`` Clebsch-Gordan tensor product; the
  SO(2)-frame upgrade is a drop-in replacement for the edge tensor product.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.basis import AtomicOrbitalBasis  # noqa: TC001
from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001
from opifex.core.quantum.registry import register_property_head
from opifex.neural.atomistic.backbones._message_passing import edge_geometry, EdgeGeometry
from opifex.neural.atomistic.backbones.nequip import _ConvolutionLayer, NequIPConfig
from opifex.neural.equivariant import (
    BesselBasis,
    cosine_cutoff,
    EquivariantLinear,
    FullyConnectedTensorProduct,
    Irreps,
    IrrepsArray,
    spherical_harmonics,
)
from opifex.neural.equivariant._assembly import from_chunks
from opifex.neural.quantum.hamiltonian._expansion import PairExpansion
from opifex.neural.quantum.hamiltonian._shell_pairs import (
    build_shell_pair_plan,
    ShellPairPlan,
)


_MAX_ATOMIC_NUMBER = 118
"""Highest supported nuclear charge; the embedding table has one row per Z."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HamiltonianPredictorConfig:
    """Hyper-parameters of a :class:`HamiltonianPredictor`.

    Attributes:
        hidden_irreps: Steerable layout of the trunk's per-atom features. Must
            carry every degree ``L`` reachable by the basis shell pairs (for
            ``s``/``p`` bases that is ``0e``, ``1o`` and ``2e``).
        sh_lmax: Maximum spherical-harmonic degree of the edge embedding.
        num_interactions: Number of tensor-product convolution layers.
        num_radial_basis: Number of Bessel radial-basis functions.
        radial_hidden_dim: Hidden width of the radial network MLP.
        cutoff: Connection / cutoff radius ``r_c`` (Bohr).
        average_num_neighbors: ``sqrt`` normaliser for aggregated messages.
        property_name: The emitted property name (``"hamiltonian"`` or
            ``"overlap"``); also the registry/output key.
    """

    hidden_irreps: str = "32x0e + 16x1o + 8x2e"
    sh_lmax: int = 2
    num_interactions: int = 3
    num_radial_basis: int = 8
    radial_hidden_dim: int = 64
    cutoff: float = 6.0
    average_num_neighbors: float = 1.0
    property_name: str = "hamiltonian"

    def to_nequip(self) -> NequIPConfig:
        """Return the equivalent :class:`NequIPConfig` for the shared trunk layers."""
        return NequIPConfig(
            hidden_irreps=self.hidden_irreps,
            sh_lmax=self.sh_lmax,
            num_interactions=self.num_interactions,
            num_radial_basis=self.num_radial_basis,
            radial_hidden_dim=self.radial_hidden_dim,
            cutoff=self.cutoff,
            average_num_neighbors=self.average_num_neighbors,
        )


def _pair_type_key(l_i: int, l_j: int) -> str:
    return f"{l_i}_{l_j}"


@dataclass(frozen=True, slots=True, kw_only=True)
class _PairTypeScatterPlan:
    """Static, vectorised assembly plan for all blocks of one ``(l_i, l_j)`` type.

    Every field is a compile-time-constant tuple of ``int`` indices (converted to
    a device array at use), so :meth:`HamiltonianPredictor._assemble` can replace
    the per-block Python loop with one gather, one batched :class:`PairExpansion`,
    one batched rank-select and one scatter per pair type. The number of pair
    types is a small constant (``~len(degrees)^2``), so the outer Python loop over
    types is *not* a compile-time unroll of the block count. Tuple fields (not
    arrays) keep the frozen plan hashable with simple equality, so it is a valid
    Flax-NNX graphdef metadata field across ``nnx.jit`` calls (NNX rejects
    array-valued metadata).

    Attributes:
        key: The ``"{l_i}_{l_j}"`` expansion-dictionary key.
        source_index: For each block, the index into the source-feature array
            (atom index for diagonal blocks, edge slot for off-diagonal blocks);
            shape ``(n_blocks,)``.
        rank_i: Row-shell rank of each block among same-``l_i`` shells; shape
            ``(n_blocks,)``.
        rank_j: Column-shell rank of each block; shape ``(n_blocks,)``.
        row_index: Flattened destination row indices of every block element; shape
            ``(n_blocks * (2 l_i + 1) * (2 l_j + 1),)``.
        col_index: Flattened destination column indices; same shape as
            ``row_index``.
    """

    key: str
    source_index: tuple[int, ...]
    rank_i: tuple[int, ...]
    rank_j: tuple[int, ...]
    row_index: tuple[int, ...]
    col_index: tuple[int, ...]


def _build_scatter_plans(
    specs: tuple[tuple[int, int, int, int, int, int, int], ...],
) -> tuple[_PairTypeScatterPlan, ...]:
    """Group per-block specs by ``(l_i, l_j)`` into vectorised scatter plans.

    Each spec is ``(row_offset, col_offset, l_i, l_j, rank_i, rank_j,
    source_index)``. Blocks of the same angular-momentum pair type share an
    expansion and a fixed ``(2 l_i + 1, 2 l_j + 1)`` shape, so they are batched
    together; the flattened ``(row, col)`` destination grids let one ``.at[].set``
    place every block of the type at once.

    Args:
        specs: The per-block specs (diagonal or off-diagonal).

    Returns:
        One :class:`_PairTypeScatterPlan` per distinct ``(l_i, l_j)`` present,
        in sorted key order for determinism.
    """
    grouped: dict[tuple[int, int], list[tuple[int, int, int, int, int]]] = {}
    for row_offset, col_offset, l_i, l_j, rank_i, rank_j, source_index in specs:
        grouped.setdefault((l_i, l_j), []).append(
            (row_offset, col_offset, rank_i, rank_j, source_index)
        )

    plans: list[_PairTypeScatterPlan] = []
    for (l_i, l_j), entries in sorted(grouped.items()):
        dim_i = 2 * l_i + 1
        dim_j = 2 * l_j + 1
        row_offsets = np.fromiter((e[0] for e in entries), dtype=np.int32)
        col_offsets = np.fromiter((e[1] for e in entries), dtype=np.int32)
        rank_i = np.fromiter((e[2] for e in entries), dtype=np.int32)
        rank_j = np.fromiter((e[3] for e in entries), dtype=np.int32)
        source_index = np.fromiter((e[4] for e in entries), dtype=np.int32)
        # Per-block (di, dj) intra-block grids broadcast against per-block offsets.
        # Both grids span the full (n_blocks, dim_i, dim_j) element space so the
        # row varies along dim_i and the column along dim_j for every block.
        within_rows = np.arange(dim_i, dtype=np.int32)[None, :, None]
        within_cols = np.arange(dim_j, dtype=np.int32)[None, None, :]
        block_shape = (len(entries), dim_i, dim_j)
        row_index = np.broadcast_to(row_offsets[:, None, None] + within_rows, block_shape).reshape(
            -1
        )
        col_index = np.broadcast_to(col_offsets[:, None, None] + within_cols, block_shape).reshape(
            -1
        )
        plans.append(
            _PairTypeScatterPlan(
                key=_pair_type_key(l_i, l_j),
                source_index=tuple(source_index.tolist()),
                rank_i=tuple(rank_i.tolist()),
                rank_j=tuple(rank_j.tolist()),
                row_index=tuple(row_index.tolist()),
                col_index=tuple(col_index.tolist()),
            )
        )
    return tuple(plans)


class HamiltonianPredictor(nnx.Module):
    r"""Equivariant predictor of the dense AO Hamiltonian/overlap matrix.

    Satisfies the :class:`opifex.core.quantum.protocols.PropertyHead` *call*
    contract for the standalone (matrix) use, and also runs directly on a
    :class:`MolecularSystem`. The dense matrix is symmetric and E(3)-equivariant
    by construction.

    Args:
        basis: The atomic-orbital basis whose static shell-pair plan fixes the
            block layout. Use :meth:`rebind` to swap in another molecule's basis
            without changing the learnable weights.
        config: Hyper-parameters. Defaults to :class:`HamiltonianPredictorConfig`.
        rngs: Random number generators (keyword-only) seeding all weights.
    """

    def __init__(
        self,
        *,
        basis: AtomicOrbitalBasis,
        config: HamiltonianPredictorConfig | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the trunk, the per-pair-type expansions and the static block plan."""
        super().__init__()
        self.config = config if config is not None else HamiltonianPredictorConfig()
        self.hidden_irreps = Irreps(self.config.hidden_irreps)
        self._num_scalars = sum(mul for mul, ir in self.hidden_irreps.blocks if ir.l == 0)
        if self._num_scalars == 0:
            raise ValueError(
                f"hidden_irreps must contain a 0e scalar channel, got {self.hidden_irreps!r}"
            )

        nequip_config = self.config.to_nequip()
        self.sh_irreps = spherical_harmonics(self.config.sh_lmax, jnp.zeros((1, 3))).irreps

        # --- trunk: embedding + NequIP convolution layers (full steerable output) ---
        self.embedding = nnx.Embed(
            num_embeddings=_MAX_ATOMIC_NUMBER + 1, features=self._num_scalars, rngs=rngs
        )
        self._embedding_irreps = Irreps(f"{self._num_scalars}x0e")
        self.embedding_linear = EquivariantLinear(
            self._embedding_irreps, self.hidden_irreps, rngs=rngs
        )
        self.radial_basis = BesselBasis(self.config.num_radial_basis, self.config.cutoff)
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

        # --- edge feature builder: radially-modulated sender (x) Y, mixed receiver ---
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

        # --- per-(l_i, l_j) shared expansions for node (diagonal) and edge (off) ---
        # The expansion multiplicities ``(mul_i, mul_j)`` are the max number of
        # shells of each degree on any one atom, so same-``l`` shells (e.g. STO-3G
        # oxygen's 1s/2s) are distinguished by the expansion's multiplicity axes.
        self._max_shells = build_shell_pair_plan(basis).max_shells_per_degree
        pair_types = self._basis_pair_types(basis)
        self.node_expansions = nnx.Dict(
            {
                _pair_type_key(l_i, l_j): PairExpansion(
                    self.hidden_irreps,
                    l_i=l_i,
                    l_j=l_j,
                    mul_i=self._max_shells[l_i],
                    mul_j=self._max_shells[l_j],
                    rngs=rngs,
                )
                for (l_i, l_j) in pair_types
            }
        )
        self.edge_expansions = nnx.Dict(
            {
                _pair_type_key(l_i, l_j): PairExpansion(
                    self.hidden_irreps,
                    l_i=l_i,
                    l_j=l_j,
                    mul_i=self._max_shells[l_i],
                    mul_j=self._max_shells[l_j],
                    rngs=rngs,
                )
                for (l_i, l_j) in pair_types
            }
        )

        self._bind_basis(basis)

    @staticmethod
    def _basis_pair_types(basis: AtomicOrbitalBasis) -> tuple[tuple[int, int], ...]:
        """Return the distinct ``(l_i, l_j)`` angular-momentum pair types in ``basis``."""
        degrees = sorted({shell.angular_momentum for shell in basis.shells})
        return tuple((l_i, l_j) for l_i in degrees for l_j in degrees)

    def _bind_basis(self, basis: AtomicOrbitalBasis) -> None:
        """Store the static shell-pair plan and the complete-graph edge ordering."""
        plan = build_shell_pair_plan(basis)
        self._plan: ShellPairPlan = plan
        self._n_ao = plan.n_atomic_orbitals
        self._n_atoms = len({shell.atom_index for shell in basis.shells})
        self._diagonal_specs = tuple(
            (b.row_offset, b.col_offset, b.l_i, b.l_j, b.rank_i, b.rank_j, b.atom_i)
            for b in plan.diagonal_blocks
        )
        # Off-diagonal blocks index into the *complete directed graph* of ordered
        # atom pairs (canonical sender-major order, self-loops skipped); the block
        # H[i, j] is fed by the edge whose sender is the column atom ``j`` and
        # receiver is the row atom ``i``.
        edge_slot = self._complete_edge_slots(self._n_atoms)
        self._off_specs = tuple(
            (
                b.row_offset,
                b.col_offset,
                b.l_i,
                b.l_j,
                b.rank_i,
                b.rank_j,
                edge_slot[(b.atom_j, b.atom_i)],
            )
            for b in plan.off_diagonal_blocks
        )
        # Vectorised, per-pair-type scatter plans (static numpy index arrays) so
        # ``_assemble`` is one gather + one batched expansion + one scatter per
        # angular-momentum pair type, never a per-block Python unroll.
        self._diagonal_scatter = _build_scatter_plans(self._diagonal_specs)
        self._off_scatter = _build_scatter_plans(self._off_specs)

    @staticmethod
    def _complete_edge_slots(n_atoms: int) -> dict[tuple[int, int], int]:
        """Map each ordered ``(sender, receiver)`` pair to its complete-graph slot."""
        slot: dict[tuple[int, int], int] = {}
        index = 0
        for sender in range(n_atoms):
            for receiver in range(n_atoms):
                if sender != receiver:
                    slot[(sender, receiver)] = index
                    index += 1
        return slot

    @staticmethod
    def _complete_graph(n_atoms: int) -> tuple[Array, Array]:
        """Build the complete directed ``(senders, receivers)`` edge index."""
        senders: list[int] = []
        receivers: list[int] = []
        for sender in range(n_atoms):
            for receiver in range(n_atoms):
                if sender != receiver:
                    senders.append(sender)
                    receivers.append(receiver)
        return jnp.asarray(senders, dtype=jnp.int32), jnp.asarray(receivers, dtype=jnp.int32)

    def rebind(self, basis: AtomicOrbitalBasis) -> HamiltonianPredictor:
        """Return a copy bound to ``basis`` (shared weights, new static plan).

        Switching molecules changes only the static shell-pair plan; the trunk and
        per-pair-type expansion weights are shared, so a single trained predictor
        generalises across molecules. (A jit recompile occurs for the new atom /
        orbital count, as for any static-shape change.)

        Args:
            basis: The new atomic-orbital basis to bind.

        Returns:
            A :class:`HamiltonianPredictor` sharing this one's parameters.

        Raises:
            ValueError: If ``basis`` carries an angular-momentum pair type absent
                from the bound expansions, or needs more same-``l`` shells than the
                bound expansions provide (build on a richer basis first).
        """
        for l_i, l_j in self._basis_pair_types(basis):
            if _pair_type_key(l_i, l_j) not in self.node_expansions:
                raise ValueError(
                    f"basis needs expansion for pair type (l_i={l_i}, l_j={l_j}) which is "
                    "absent from this predictor; build it on a basis that contains the type."
                )
        for degree, count in build_shell_pair_plan(basis).max_shells_per_degree.items():
            if count > self._max_shells.get(degree, 0):
                raise ValueError(
                    f"basis needs {count} shells of degree {degree} but this predictor "
                    f"provides {self._max_shells.get(degree, 0)}; build it on a basis with "
                    "at least that many same-l shells."
                )
        graphdef, state = nnx.split(self)
        clone = nnx.merge(graphdef, state)
        clone._bind_basis(basis)
        return clone

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """The single matrix property this head emits."""
        return (self.config.property_name,)

    # ------------------------------------------------------------------ trunk ---
    def _node_features(
        self, system: MolecularSystem, graph: tuple[Array, Array]
    ) -> tuple[IrrepsArray, EdgeGeometry, IrrepsArray, Array]:
        """Run the trunk; return node features and reusable edge tensors."""
        geometry = edge_geometry(system.positions, graph)
        lengths = geometry.lengths[:, 0]
        radial = self.radial_basis(lengths)
        envelope = (cosine_cutoff(lengths, self.config.cutoff) * geometry.mask[:, 0])[:, None]
        edge_sh = spherical_harmonics(self.sh_irreps, geometry.vectors)
        scalar_embedding = IrrepsArray(
            self._embedding_irreps, self.embedding(system.atomic_numbers)
        )
        node_features = self.embedding_linear(scalar_embedding)
        for layer in self.layers:
            node_features = layer(
                node_features, edge_sh, geometry, radial, envelope, system.n_atoms
            )
        return node_features, geometry, edge_sh, radial * envelope

    def _edge_features(
        self,
        node_features: IrrepsArray,
        geometry: EdgeGeometry,
        edge_sh: IrrepsArray,
        radial_envelope: Array,
    ) -> IrrepsArray:
        """Build a steerable per-edge feature from sender/receiver/direction."""
        senders = geometry.senders
        receivers = geometry.receivers
        sender_features = IrrepsArray(node_features.irreps, node_features.array[senders])
        message = self.edge_tensor_product(sender_features, edge_sh)
        weights = self.edge_radial(radial_envelope)
        message = self._apply_radial_weights(message, weights)
        receiver_features = IrrepsArray(node_features.irreps, node_features.array[receivers])
        mixed = self.edge_receiver_mix(receiver_features)
        return IrrepsArray(self.hidden_irreps, message.array + mixed.array)

    def _apply_radial_weights(self, message: IrrepsArray, weights: Array) -> IrrepsArray:
        """Scale each output multiplicity of ``message`` by its radial weight."""
        scaled: list[Array | None] = []
        cursor = 0
        for (mul, _), chunk in zip(message.irreps.blocks, message.chunks, strict=True):
            block_weights = weights[:, cursor : cursor + mul]
            scaled.append(chunk * block_weights[..., None])
            cursor += mul
        return from_chunks(message.irreps, scaled, message.array.shape[:-1], message.array.dtype)

    # --------------------------------------------------------------- assembly ---
    @staticmethod
    def _scatter_pair_type(
        matrix: Array,
        expansion: PairExpansion,
        source: IrrepsArray,
        plan: _PairTypeScatterPlan,
    ) -> Array:
        """Place every block of one ``(l_i, l_j)`` type into ``matrix`` at once.

        Gathers the ``n_blocks`` source features named by ``plan.source_index`` in
        one indexing op, runs a single batched :class:`PairExpansion` over the
        stack, selects the per-block ``(rank_i, rank_j)`` shell pair with one
        batched gather, then scatters all block elements with a single
        ``.at[].set``. Blocks of distinct shell pairs occupy disjoint AO regions,
        so the scatter never overlaps and ``.set`` is exact.

        Args:
            matrix: The dense AO matrix being assembled.
            expansion: The shared expansion module for this pair type.
            source: The node (diagonal) or edge (off-diagonal) feature array.
            plan: The static scatter plan for this pair type.

        Returns:
            ``matrix`` with this pair type's blocks written in.
        """
        source_index = jnp.asarray(plan.source_index)
        stacked = IrrepsArray(source.irreps, source.array[source_index])
        # (n_blocks, mul_i, mul_j, 2l_i+1, 2l_j+1); leading axis batches the einsum.
        blocks = expansion(stacked)
        block_indices = jnp.arange(blocks.shape[0])
        # Select each block's (rank_i, rank_j) shell pair -> (n_blocks, di, dj).
        selected = blocks[block_indices, jnp.asarray(plan.rank_i), jnp.asarray(plan.rank_j)]
        return matrix.at[jnp.asarray(plan.row_index), jnp.asarray(plan.col_index)].set(
            selected.reshape(-1)
        )

    def _assemble(self, node_features: IrrepsArray, edge_features: IrrepsArray) -> Array:
        """Scatter node (diagonal) and edge (off-diagonal) blocks into a dense matrix.

        Vectorised by angular-momentum pair type: a Python loop over the handful of
        distinct ``(l_i, l_j)`` types (not the thousands of blocks) drives, per
        type, one gather, one batched :class:`PairExpansion` and one scatter. The
        diagonal blocks read node features at the receiver atom; the off-diagonal
        blocks read edge features at the precomputed sender=j -> receiver=i slot;
        the shell ranks select the right same-``l`` shell pair (e.g. oxygen 1s vs
        2s).
        """
        dtype = node_features.array.dtype
        matrix = jnp.zeros((self._n_ao, self._n_ao), dtype=dtype)
        for plan in self._diagonal_scatter:
            matrix = self._scatter_pair_type(
                matrix, self.node_expansions[plan.key], node_features, plan
            )
        for plan in self._off_scatter:
            matrix = self._scatter_pair_type(
                matrix, self.edge_expansions[plan.key], edge_features, plan
            )
        return matrix

    def __call__(self, system: MolecularSystem) -> dict[str, Array]:
        """Predict the dense, symmetric, equivariant matrix for ``system``.

        Args:
            system: The molecular system (atomic numbers + positions in Bohr).

        Returns:
            ``{property_name: matrix}`` with ``matrix`` of shape ``(n_ao, n_ao)``.
            The matrix is symmetric (``H = H~ + H~^T``) and E(3)-equivariant.
        """
        # Both the trunk message passing and the off-diagonal assembly run on the
        # complete directed graph, so every off-diagonal block has a real edge
        # (the smooth radial cutoff still down-weights distant pairs). The matrix
        # symmetrisation ``H = H~ + H~^T`` realises H_ij = H_ji^T because the
        # graph contains both directed edges and the transpose of the (j, i) block
        # lands on the (i, j) sub-matrix.
        graph = self._complete_graph(system.n_atoms)
        node_features, geometry, edge_sh, radial_envelope = self._node_features(system, graph)
        edge_features = self._edge_features(node_features, geometry, edge_sh, radial_envelope)
        raw = self._assemble(node_features, edge_features)
        matrix = raw + raw.T
        return {self.config.property_name: matrix}


HamiltonianPredictor = register_property_head("hamiltonian")(HamiltonianPredictor)


__all__ = ["HamiltonianPredictor", "HamiltonianPredictorConfig"]

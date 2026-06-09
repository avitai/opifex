r"""Tests for :class:`BlockHamiltonianPredictor` (heterogeneous-batchable QHNet core).

The predictor consumes a *flat heterogeneous batch* -- many molecules of different
composition concatenated into ``(A, 3)`` positions, ``(A,)`` atomic numbers and a
``(2, E)`` within-molecule directed edge index -- and emits a fixed ``(14, 14)``
diagonal Fock block per atom and a ``(14, 14)`` off-diagonal block per directed
edge (reference ``divelab/AIRS`` ``OpenDFT/QHBench/QH9/models/QHNet.py``). Because
the blocks are fixed-size and the NequIP convolution scatters only over
within-molecule edges, the same compiled forward runs over any concatenation.

The correctness gates are:

* **Block equivariance** -- under a random rotation ``R`` each diagonal block obeys
  ``B_ii(R r) = D_14(R) B_ii(r) D_14(R)^T`` and each off-diagonal block obeys the
  same law with ``D_14`` the real Wigner-D of ``BLOCK_IRREPS = 3x0e + 2x1e + 1x2e``.
* **Heterogeneous-batch consistency** -- concatenating two different molecules into
  one flat batch yields the same per-molecule blocks as running each alone (the
  whole point of the segment/concatenated design).
* **assemble_matrix** -- the single-molecule inference helper builds a symmetric
  ``(n_ao, n_ao)`` dense matrix with the correct per-element AO sizes.
* ``jit`` / ``grad`` / ``vmap`` cleanliness and a repeated ``nnx.jit`` step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.quantum.hamiltonian._orbital_layout import (
    atom_orbital_counts,
    BLOCK_IRREPS,
    block_validity_mask,
    FULL_ORBITALS,
)
from opifex.neural.quantum.hamiltonian.block_predictor import (
    BlockHamiltonianConfig,
    BlockHamiltonianPredictor,
)


# Small, fast config (lmax 2, two interactions) -- enough to exercise s/p/d shells.
_CONFIG = BlockHamiltonianConfig(
    hidden_irreps="8x0e + 8x1o + 8x2e",
    sh_lmax=2,
    num_interactions=2,
    cutoff=10.0,
    embed_dim=16,
)


def _predictor(seed: int = 0) -> BlockHamiltonianPredictor:
    """Build a small predictor with the shared test config."""
    return BlockHamiltonianPredictor(config=_CONFIG, rngs=nnx.Rngs(seed))


def _complete_edges(offset: int, n_atoms: int) -> tuple[list[int], list[int]]:
    """Return the within-molecule directed (sender, receiver) lists, no self-loops."""
    senders: list[int] = []
    receivers: list[int] = []
    for sender in range(n_atoms):
        for receiver in range(n_atoms):
            if sender != receiver:
                senders.append(offset + sender)
                receivers.append(offset + receiver)
    return senders, receivers


def _water() -> tuple[jax.Array, jax.Array]:
    """Water (O, H, H) atomic numbers and positions in Bohr."""
    atomic_numbers = jnp.array([8, 1, 1])
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]], dtype=jnp.float64
    )
    return atomic_numbers, positions


def _methane_like() -> tuple[jax.Array, jax.Array]:
    """A 5-atom C + 4 H molecule (different composition/size from water)."""
    atomic_numbers = jnp.array([6, 1, 1, 1, 1])
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.19, 1.19, 1.19],
            [-1.19, -1.19, 1.19],
            [-1.19, 1.19, -1.19],
            [1.19, -1.19, -1.19],
        ],
        dtype=jnp.float64,
    )
    return atomic_numbers, positions


def _single_batch(
    atomic_numbers: jax.Array, positions: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build the flat-batch inputs for one molecule (complete within-molecule graph)."""
    n_atoms = int(atomic_numbers.shape[0])
    senders, receivers = _complete_edges(0, n_atoms)
    edge_index = jnp.asarray([senders, receivers], dtype=jnp.int32)
    node_batch = jnp.zeros((n_atoms,), dtype=jnp.int32)
    return atomic_numbers, positions, edge_index, node_batch


def _random_rotation(seed: int) -> jax.Array:
    """Return a uniformly random proper rotation matrix (det = +1)."""
    key = jax.random.PRNGKey(seed)
    gaussian = jax.random.normal(key, (3, 3), dtype=jnp.float64)
    orthogonal, _ = jnp.linalg.qr(gaussian)
    return orthogonal * jnp.sign(jnp.linalg.det(orthogonal))


def _wigner_block(rotation: jax.Array) -> jax.Array:
    """Build the ``(14, 14)`` block-diagonal Wigner-D of ``BLOCK_IRREPS``."""
    matrices: list[jax.Array] = []
    for mul, irrep in BLOCK_IRREPS.blocks:
        wigner = wigner_d(irrep.l, rotation)
        matrices.extend([wigner] * mul)
    return jax.scipy.linalg.block_diag(*matrices)


# --------------------------------------------------------------------------- API


def test_output_keys_and_shapes() -> None:
    """The forward emits per-atom and per-edge fixed ``(14, 14)`` blocks."""
    predictor = _predictor()
    atomic_numbers, positions, edge_index, node_batch = _single_batch(*_water())
    out = predictor(atomic_numbers, positions, edge_index, node_batch)
    n_atoms = int(atomic_numbers.shape[0])
    n_edges = int(edge_index.shape[1])
    assert out["diagonal_blocks"].shape == (n_atoms, FULL_ORBITALS, FULL_ORBITALS)
    assert out["off_diagonal_blocks"].shape == (n_edges, FULL_ORBITALS, FULL_ORBITALS)


def test_node_batch_is_optional() -> None:
    """``node_batch`` is accepted for completeness but not required for the forward."""
    predictor = _predictor()
    atomic_numbers, positions, edge_index, _ = _single_batch(*_water())
    out = predictor(atomic_numbers, positions, edge_index)
    assert out["diagonal_blocks"].shape[0] == int(atomic_numbers.shape[0])


def test_diagonal_blocks_are_symmetric() -> None:
    """Per-atom diagonal blocks are symmetric (QHNet ``D + D^T`` symmetrisation)."""
    predictor = _predictor()
    atomic_numbers, positions, edge_index, node_batch = _single_batch(*_water())
    out = predictor(atomic_numbers, positions, edge_index, node_batch)
    diagonal = out["diagonal_blocks"]
    assert jnp.allclose(diagonal, jnp.swapaxes(diagonal, -1, -2), atol=1e-5)


# ------------------------------------------------------------------ equivariance


def test_block_equivariance() -> None:
    r"""Diagonal and off-diagonal blocks obey ``B(R r) = D_14(R) B(r) D_14(R)^T``."""
    jax.config.update("jax_enable_x64", True)
    predictor = _predictor(seed=1)
    atomic_numbers, positions = _water()
    _, _, edge_index, node_batch = _single_batch(atomic_numbers, positions)
    rotation = _random_rotation(7)
    wigner = _wigner_block(rotation)

    base = predictor(atomic_numbers, positions, edge_index, node_batch)
    rotated = predictor(atomic_numbers, positions @ rotation.T, edge_index, node_batch)

    expected_diag = jnp.einsum("ij,njk,lk->nil", wigner, base["diagonal_blocks"], wigner)
    expected_off = jnp.einsum("ij,njk,lk->nil", wigner, base["off_diagonal_blocks"], wigner)
    diag_residual = float(jnp.max(jnp.abs(rotated["diagonal_blocks"] - expected_diag)))
    off_residual = float(jnp.max(jnp.abs(rotated["off_diagonal_blocks"] - expected_off)))
    assert diag_residual < 1e-5, diag_residual
    assert off_residual < 1e-5, off_residual


# ------------------------------------------------- heterogeneous-batch consistency


def test_heterogeneous_batch_matches_separate() -> None:
    """Concatenating two different molecules gives the same per-molecule blocks.

    This is the key test: it proves the segment/concatenated design batches
    heterogeneous molecules correctly (the whole point of the block predictor).
    """
    jax.config.update("jax_enable_x64", True)
    predictor = _predictor(seed=2)

    water_z, water_pos = _water()
    methane_z, methane_pos = _methane_like()
    n_water = int(water_z.shape[0])
    n_methane = int(methane_z.shape[0])

    # Per-molecule references.
    water_out = predictor(*_single_batch(water_z, water_pos))
    methane_out = predictor(*_single_batch(methane_z, methane_pos))

    # Concatenated heterogeneous batch (edges offset so they never cross molecules).
    atomic_numbers = jnp.concatenate([water_z, methane_z])
    positions = jnp.concatenate([water_pos, methane_pos])
    water_senders, water_receivers = _complete_edges(0, n_water)
    methane_senders, methane_receivers = _complete_edges(n_water, n_methane)
    edge_index = jnp.asarray(
        [water_senders + methane_senders, water_receivers + methane_receivers],
        dtype=jnp.int32,
    )
    node_batch = jnp.concatenate(
        [jnp.zeros((n_water,), jnp.int32), jnp.ones((n_methane,), jnp.int32)]
    )
    batch_out = predictor(atomic_numbers, positions, edge_index, node_batch)

    n_water_edges = n_water * (n_water - 1)
    # Diagonal blocks: first n_water atoms == water, remainder == methane.
    assert jnp.allclose(
        batch_out["diagonal_blocks"][:n_water], water_out["diagonal_blocks"], atol=1e-8
    )
    assert jnp.allclose(
        batch_out["diagonal_blocks"][n_water:], methane_out["diagonal_blocks"], atol=1e-8
    )
    # Off-diagonal blocks split at the molecule's edge boundary.
    assert jnp.allclose(
        batch_out["off_diagonal_blocks"][:n_water_edges],
        water_out["off_diagonal_blocks"],
        atol=1e-8,
    )
    assert jnp.allclose(
        batch_out["off_diagonal_blocks"][n_water_edges:],
        methane_out["off_diagonal_blocks"],
        atol=1e-8,
    )


# ----------------------------------------------------------------- assemble_matrix


def test_assemble_matrix_is_symmetric_and_correctly_sized() -> None:
    """The single-molecule helper builds a symmetric ``(n_ao, n_ao)`` matrix."""
    predictor = _predictor(seed=3)
    atomic_numbers, positions, edge_index, node_batch = _single_batch(*_water())
    out = predictor(atomic_numbers, positions, edge_index, node_batch)
    matrix = predictor.assemble_matrix(
        out["diagonal_blocks"], out["off_diagonal_blocks"], atomic_numbers, edge_index
    )
    expected_ao = int(jnp.sum(atom_orbital_counts(atomic_numbers)))
    # Water def2-svp: O (14) + H (5) + H (5) = 24.
    assert expected_ao == 24
    assert matrix.shape == (expected_ao, expected_ao)
    assert jnp.allclose(matrix, matrix.T, atol=1e-5)


def _assembled_ao_wigner(atomic_numbers: jax.Array, rotation: jax.Array) -> jax.Array:
    """Build the ``(n_ao, n_ao)`` block-diagonal Wigner-D of an assembled matrix.

    The assembled dense matrix orders AOs atom-major, each atom contributing its
    populated slots of the 14-dim ``BLOCK_IRREPS`` block. The matching rotation is
    therefore block-diagonal over atoms, each atom's block being the full
    ``(14, 14)`` ``D_14`` restricted to that atom's valid AO slots (whole shells,
    so the restriction is exact).
    """
    full = _wigner_block(rotation)
    diag_mask = block_validity_mask(atomic_numbers)
    per_atom: list[jax.Array] = []
    for atom in range(int(atomic_numbers.shape[0])):
        valid = jnp.where(jnp.diagonal(diag_mask[atom]))[0]
        per_atom.append(full[jnp.ix_(valid, valid)])
    return jax.scipy.linalg.block_diag(*per_atom)


def test_assembled_matrix_is_equivariant() -> None:
    r"""The assembled dense matrix obeys ``H(R r) = D(R) H(r) D(R)^T`` (Q4 gate)."""
    jax.config.update("jax_enable_x64", True)
    predictor = _predictor(seed=4)
    atomic_numbers, positions, edge_index, node_batch = _single_batch(*_water())
    rotation = _random_rotation(11)

    base_out = predictor(atomic_numbers, positions, edge_index, node_batch)
    rotated_out = predictor(atomic_numbers, positions @ rotation.T, edge_index, node_batch)
    base = predictor.assemble_matrix(
        base_out["diagonal_blocks"], base_out["off_diagonal_blocks"], atomic_numbers, edge_index
    )
    rotated = predictor.assemble_matrix(
        rotated_out["diagonal_blocks"],
        rotated_out["off_diagonal_blocks"],
        atomic_numbers,
        edge_index,
    )
    wigner = _assembled_ao_wigner(atomic_numbers, rotation)
    expected = wigner @ base @ wigner.T
    residual = float(jnp.max(jnp.abs(rotated - expected)))
    assert residual < 1e-5, residual


# ----------------------------------------------------------- transform compatibility


def test_jit_grad_vmap_smoke() -> None:
    """The forward is ``jit``/``grad``/``vmap`` clean."""
    predictor = _predictor(seed=4)
    atomic_numbers, positions, edge_index, node_batch = _single_batch(*_water())

    @nnx.jit
    def forward(module: BlockHamiltonianPredictor, pos: jax.Array) -> jax.Array:
        out = module(atomic_numbers, pos, edge_index, node_batch)
        return out["diagonal_blocks"].sum() + out["off_diagonal_blocks"].sum()

    value = forward(predictor, positions)
    assert jnp.isfinite(value)

    grads = nnx.grad(forward, argnums=1)(predictor, positions)
    assert grads.shape == positions.shape
    assert jnp.all(jnp.isfinite(grads))

    batched_positions = jnp.stack([positions, positions + 0.1])
    blocks = jax.vmap(
        lambda pos: predictor(atomic_numbers, pos, edge_index, node_batch)["diagonal_blocks"]
    )(batched_positions)
    assert blocks.shape == (2, int(atomic_numbers.shape[0]), FULL_ORBITALS, FULL_ORBITALS)


def test_repeated_nnx_jit_step() -> None:
    """A repeated ``nnx.jit`` training-step call works (no array-valued metadata)."""
    predictor = _predictor(seed=5)
    atomic_numbers, positions, edge_index, node_batch = _single_batch(*_water())
    optimizer = nnx.Optimizer(predictor, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit
    def step(module: BlockHamiltonianPredictor, opt: nnx.Optimizer, pos: jax.Array) -> jax.Array:
        def loss_fn(m: BlockHamiltonianPredictor) -> jax.Array:
            out = m(atomic_numbers, pos, edge_index, node_batch)
            return out["diagonal_blocks"].sum() ** 2

        loss, grads = nnx.value_and_grad(loss_fn)(module)
        opt.update(module, grads)
        return loss

    loss_one = step(predictor, optimizer, positions)
    loss_two = step(predictor, optimizer, positions)
    assert jnp.isfinite(loss_one)
    assert jnp.isfinite(loss_two)

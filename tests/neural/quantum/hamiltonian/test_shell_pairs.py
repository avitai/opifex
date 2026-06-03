r"""Tests for the static shell-pair plan derived from an :class:`AtomicOrbitalBasis`."""

from __future__ import annotations

import jax.numpy as jnp

from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.quantum.hamiltonian._shell_pairs import build_shell_pair_plan


def _water_basis() -> AtomicOrbitalBasis:
    system = MolecularSystem(
        atomic_numbers=jnp.array([8, 1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]]),
        basis_set="sto-3g",
    )
    return AtomicOrbitalBasis.from_molecular_system(system, basis_name="sto-3g")


def test_plan_matrix_size_matches_basis() -> None:
    """The plan records the correct dense-matrix side length (7 for STO-3G water)."""
    plan = build_shell_pair_plan(_water_basis())
    assert plan.n_atomic_orbitals == 7


def test_diagonal_and_off_diagonal_partition() -> None:
    """Diagonal blocks are intra-atom; off-diagonal blocks are inter-atom."""
    plan = build_shell_pair_plan(_water_basis())
    assert all(block.atom_i == block.atom_j for block in plan.diagonal_blocks)
    assert all(block.atom_i != block.atom_j for block in plan.off_diagonal_blocks)


def test_oxygen_s_shells_get_distinct_ranks() -> None:
    """Oxygen's two ``s`` shells (1s, 2s) are ranked 0 and 1; ``p`` is rank 0."""
    plan = build_shell_pair_plan(_water_basis())
    # The oxygen-oxygen s/s diagonal blocks span ranks {0, 1} x {0, 1}.
    oxygen_ss = [
        block
        for block in plan.diagonal_blocks
        if block.atom_i == 0 and block.l_i == 0 and block.l_j == 0
    ]
    rank_pairs = {(block.rank_i, block.rank_j) for block in oxygen_ss}
    assert rank_pairs == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_max_shells_per_degree() -> None:
    """Water (STO-3G) has at most 2 ``s`` shells (oxygen) and 1 ``p`` shell per atom."""
    plan = build_shell_pair_plan(_water_basis())
    assert plan.max_shells_per_degree == {0: 2, 1: 1}


def test_block_offsets_within_matrix() -> None:
    """Every block's offsets and angular extent stay inside the dense matrix."""
    plan = build_shell_pair_plan(_water_basis())
    for block in (*plan.diagonal_blocks, *plan.off_diagonal_blocks):
        assert 0 <= block.row_offset + 2 * block.l_i < plan.n_atomic_orbitals + block.l_i
        assert block.row_offset + (2 * block.l_i + 1) <= plan.n_atomic_orbitals
        assert block.col_offset + (2 * block.l_j + 1) <= plan.n_atomic_orbitals

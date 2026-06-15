r"""Static shell-pair plan derived from an :class:`AtomicOrbitalBasis`.

The Hamiltonian predictor scatters fixed-size ``(2 l_i + 1) x (2 l_j + 1)`` blocks
into the dense AO matrix at ``(shell_i.ao_offset, shell_j.ao_offset)``. Which
blocks exist, their angular momenta and their destination offsets are *static*
properties of the basis (they depend only on the element identities and the basis
set, not on the geometry), so this module precomputes them once as plain Python
data. Keeping the plan static is what lets the predictor stay ``jit``-clean: the
per-block einsum shapes never depend on traced values.

Two kinds of pairs:

* **Intra-atom (diagonal) pairs** -- both shells on the same atom. These form the
  on-site blocks ``H_ii`` and are predicted from per-atom (node) features.
* **Inter-atom (off-diagonal) pairs** -- the two shells on different atoms. These
  form the ``H_ij`` blocks and are predicted from directed-edge features; the
  predictor enforces ``H_ij = H_ji^T`` via a reverse-edge permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from opifex.core.quantum.basis import AtomicOrbitalBasis, GaussianShell


@dataclass(frozen=True, slots=True, kw_only=True)
class ShellPairBlock:
    """One ``(shell_i, shell_j)`` block of the dense AO matrix.

    Attributes:
        atom_i: Atom index of the row (receiver) shell.
        atom_j: Atom index of the column (sender) shell.
        l_i: Angular momentum of the row shell.
        l_j: Angular momentum of the column shell.
        rank_i: Index of the row shell among the atom's shells of degree ``l_i``
            (distinguishes e.g. STO-3G oxygen's ``1s`` (rank 0) from ``2s``
            (rank 1)).
        rank_j: Index of the column shell among the atom's shells of degree ``l_j``.
        row_offset: First AO index of the row shell (``shell_i.ao_offset``).
        col_offset: First AO index of the column shell (``shell_j.ao_offset``).
    """

    atom_i: int
    atom_j: int
    l_i: int
    l_j: int
    rank_i: int
    rank_j: int
    row_offset: int
    col_offset: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ShellPairPlan:
    """The full static plan of diagonal and off-diagonal AO blocks for a basis.

    Attributes:
        n_atomic_orbitals: Side length of the dense AO matrix.
        max_shells_per_degree: For each angular momentum ``l`` present, the maximum
            number of shells of that ``l`` on any single atom (the multiplicity an
            expansion needs to distinguish same-``l`` shells).
        diagonal_blocks: Intra-atom ``H_ii`` blocks (predicted from node features).
        off_diagonal_blocks: Inter-atom ``H_ij`` blocks for *ordered* atom pairs
            ``(i, j)`` with ``i != j`` (predicted from directed-edge features).
    """

    n_atomic_orbitals: int
    max_shells_per_degree: dict[int, int]
    diagonal_blocks: tuple[ShellPairBlock, ...]
    off_diagonal_blocks: tuple[ShellPairBlock, ...]


def _shell_ranks(basis: AtomicOrbitalBasis) -> dict[int, int]:
    """Map each shell (by its ``ao_offset``) to its rank among same-``l`` shells.

    The rank is the index of the shell among the shells of the same angular
    momentum on the same atom (in atom-major shell order), distinguishing e.g.
    STO-3G oxygen's ``1s`` (rank 0) from its ``2s`` (rank 1).
    """
    counters: dict[tuple[int, int], int] = {}
    ranks: dict[int, int] = {}
    for shell in basis.shells:
        key = (shell.atom_index, shell.angular_momentum)
        rank = counters.get(key, 0)
        ranks[shell.ao_offset] = rank
        counters[key] = rank + 1
    return ranks


def _max_shells_per_degree(basis: AtomicOrbitalBasis) -> dict[int, int]:
    """Return the maximum count of shells of each degree ``l`` on any one atom."""
    counts: dict[tuple[int, int], int] = {}
    for shell in basis.shells:
        key = (shell.atom_index, shell.angular_momentum)
        counts[key] = counts.get(key, 0) + 1
    maxima: dict[int, int] = {}
    for (_, degree), count in counts.items():
        maxima[degree] = max(maxima.get(degree, 0), count)
    return maxima


def _block(shell_i: GaussianShell, shell_j: GaussianShell, ranks: dict[int, int]) -> ShellPairBlock:
    return ShellPairBlock(
        atom_i=shell_i.atom_index,
        atom_j=shell_j.atom_index,
        l_i=shell_i.angular_momentum,
        l_j=shell_j.angular_momentum,
        rank_i=ranks[shell_i.ao_offset],
        rank_j=ranks[shell_j.ao_offset],
        row_offset=shell_i.ao_offset,
        col_offset=shell_j.ao_offset,
    )


def build_shell_pair_plan(basis: AtomicOrbitalBasis) -> ShellPairPlan:
    """Enumerate every diagonal and off-diagonal AO block of ``basis``.

    Args:
        basis: The atomic-orbital basis (its shells carry ``atom_index``,
            ``angular_momentum`` and ``ao_offset``).

    Returns:
        A :class:`ShellPairPlan` listing all intra-atom blocks (for the node head)
        and all inter-atom blocks for ordered atom pairs (for the edge head), with
        per-shell ranks and the per-degree shell multiplicities.
    """
    ranks = _shell_ranks(basis)
    diagonal: list[ShellPairBlock] = []
    off_diagonal: list[ShellPairBlock] = []
    for shell_i in basis.shells:
        for shell_j in basis.shells:
            block = _block(shell_i, shell_j, ranks)
            if shell_i.atom_index == shell_j.atom_index:
                diagonal.append(block)
            else:
                off_diagonal.append(block)
    return ShellPairPlan(
        n_atomic_orbitals=basis.n_atomic_orbitals,
        max_shells_per_degree=_max_shells_per_degree(basis),
        diagonal_blocks=tuple(diagonal),
        off_diagonal_blocks=tuple(off_diagonal),
    )


__all__ = ["ShellPairBlock", "ShellPairPlan", "build_shell_pair_plan"]

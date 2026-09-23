r"""def2-SVP per-element orbital layout and AO validity masks (QHNet block form).

A heterogeneous-batchable Hamiltonian predictor emits a *fixed* dense block per
atom (diagonal Fock block) and per directed edge (off-diagonal block), then masks
out the AO slots an element does not populate. For the full second-row def2-SVP
basis the block is 14-dimensional: 3 s-shells, 2 p-shells and 1 d-shell
(``3 + 2*3 + 5 = 14``, the irreps ``3x0e + 2x1e + 1x2e``), the def2-SVP
shell composition of C/N/O/F (Weigend & Ahlrichs, Phys. Chem. Chem. Phys. 7, 3297
(2005)) used by the QH9 block form (Yu et al. 2023, "QH9", arXiv:2306.09549;
Yu et al. 2023, "QHNet", arXiv:2306.04922). Hydrogen carries the def2-SVP
2 s + 1 p shells in slots ``[0, 1, 3, 4, 5]`` while C/N/O/F populate all 14. A
directed-edge (pair) block's
validity mask is the outer product of its row element's and column element's
per-atom masks.

The masks are derived from a static ``(max_Z+1, 14)`` boolean lookup table built
from :data:`ORBITAL_MASK`, so :func:`block_validity_mask` and
:func:`atom_orbital_counts` are pure index gathers -- fully ``jit``/``vmap`` clean
(the table is a compile-time-constant array; the static metadata lives in plain
Python tuples).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Int
from numpy.typing import NDArray


FULL_ORBITALS: int = 14
"""def2-SVP full second-row AO count (``3 s + 2 p + 1 d`` = ``3 + 6 + 5``)."""

_HYDROGEN_AO_INDICES: tuple[int, ...] = (0, 1, 3, 4, 5)
"""The def2-SVP 2 s + 1 p slots hydrogen populates in the 14-slot block."""

ORBITAL_MASK: dict[int, tuple[int, ...]] = {
    1: _HYDROGEN_AO_INDICES,
    6: tuple(range(FULL_ORBITALS)),
    7: tuple(range(FULL_ORBITALS)),
    8: tuple(range(FULL_ORBITALS)),
    9: tuple(range(FULL_ORBITALS)),
}
"""Atomic number -> valid AO indices into the 14-slot irrep-ordered block."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AoConvention:
    """How one molecule's def2-SVP AOs are reordered, as four per-element and per-shell tables.

    The structure and the two conventions below are those of the QH9 benchmark (Yu et al.
    2023, "QH9", arXiv:2306.09549), which stores Hamiltonians in its own AO order.

    Attributes:
        shells: Atomic number -> its shells, one letter each, in the source order.
        components: Shell letter -> the order its components take in the target.
        signs: Shell letter -> the sign applied to each of its components.
        shell_order: Atomic number -> the order its shells take in the target.
    """

    shells: Mapping[int, str]
    components: Mapping[str, tuple[int, ...]]
    signs: Mapping[str, tuple[int, ...]]
    shell_order: Mapping[int, tuple[int, ...]]


_DEF2SVP_SHELLS: dict[int, str] = {1: "ssp", 6: "sssppd", 7: "sssppd", 8: "sssppd", 9: "sssppd"}
_UNSIGNED: dict[str, tuple[int, ...]] = {"s": (1,), "p": (1, 1, 1), "d": (1, 1, 1, 1, 1)}
_SHELLS_IN_PLACE: dict[int, tuple[int, ...]] = {
    1: (0, 1, 2),
    6: (0, 1, 2, 3, 4, 5),
    7: (0, 1, 2, 3, 4, 5),
    8: (0, 1, 2, 3, 4, 5),
    9: (0, 1, 2, 3, 4, 5),
}

PYSCF_DEF2SVP: AoConvention = AoConvention(
    shells=_DEF2SVP_SHELLS,
    components={"s": (0,), "p": (1, 2, 0), "d": (0, 1, 2, 3, 4)},
    signs=_UNSIGNED,
    shell_order=_SHELLS_IN_PLACE,
)
"""QH9's native AO order to PySCF's def2-SVP spherical order: the order opifex stores."""

BACK2PYSCF: AoConvention = AoConvention(
    shells=_DEF2SVP_SHELLS,
    components={"s": (0,), "p": (2, 0, 1), "d": (0, 1, 2, 3, 4)},
    signs=_UNSIGNED,
    shell_order=_SHELLS_IN_PLACE,
)
"""The stored order to PySCF's internal spherical order, as its overlap integrals use.

It inverts :data:`PYSCF_DEF2SVP` shell by shell.
"""


def _outside_the_layout(atomic_number: int, table: Mapping[int, object]) -> str:
    """Describe an atomic number the layout does not cover.

    Args:
        atomic_number: The element that was asked for.
        table: A per-element table of the layout.

    Returns:
        The message a refusal carries.
    """
    return (
        f"atomic number {atomic_number} is outside the def2-SVP layout, which covers "
        f"{sorted(table)}"
    )


def ao_permutation(
    atomic_numbers: Sequence[int] | Int[NDArray[np.integer], " n_atoms"],
    convention: AoConvention,
) -> tuple[Int[NDArray[np.int64], " n_ao"], Int[NDArray[np.int64], " n_ao"]]:
    r"""Return the AO permutation and signs that reorder one molecule's def2-SVP matrix.

    Shells are laid out atom by atom, each shell's components are permuted and signed, and
    each atom's shells are then reordered, all by ``convention``. A matrix ``M`` becomes
    ``M'[i, j] = M[I[i], I[j]] * s[i] * s[j]``.

    Args:
        atomic_numbers: Nuclear charges of the molecule, host-side.
        convention: The reordering, such as :data:`PYSCF_DEF2SVP` or :data:`BACK2PYSCF`.

    Returns:
        ``(indices, signs)``, each of shape ``(n_ao,)`` and dtype ``int64``.

    Raises:
        KeyError: If an atomic number is outside the basis (H, C, N, O, F).
    """
    shells = ""
    shell_order: list[int] = []
    for atomic_number in atomic_numbers:
        element = int(atomic_number)
        if element not in convention.shells:
            raise KeyError(_outside_the_layout(element, convention.shells))
        first = len(shell_order)
        shells += convention.shells[element]
        shell_order += [first + shell for shell in convention.shell_order[element]]

    shell_indices: list[NDArray[np.int64]] = []
    shell_signs: list[NDArray[np.int64]] = []
    offset = 0
    for shell in shells:
        order = np.asarray(convention.components[shell], dtype=np.int64)
        shell_indices.append(order + offset)
        shell_signs.append(np.asarray(convention.signs[shell], dtype=np.int64))
        offset += order.size
    indices = np.concatenate([shell_indices[shell] for shell in shell_order])
    signs = np.concatenate([shell_signs[shell] for shell in shell_order])
    return indices, signs


def native_ao_count(atomic_numbers: Sequence[int] | Int[NDArray[np.integer], " n_atoms"]) -> int:
    """Return a molecule's def2-SVP AO count, host-side: 5 per H, 14 per C, N, O or F.

    Args:
        atomic_numbers: Nuclear charges of the molecule.

    Returns:
        The total number of atomic orbitals.

    Raises:
        KeyError: If an atomic number is outside the basis.
    """
    elements = [int(atomic_number) for atomic_number in atomic_numbers]
    for element in elements:
        if element not in ORBITAL_MASK:
            raise KeyError(_outside_the_layout(element, ORBITAL_MASK))
    return sum(len(ORBITAL_MASK[element]) for element in elements)


def _build_validity_table() -> np.ndarray:
    """Return the static ``(max_Z + 1, 14)`` boolean per-element AO validity table.

    Row ``Z`` is the 1-D mask of which of the 14 AO slots element ``Z`` populates;
    unlisted atomic numbers (and the ``Z = 0`` padding row) are all-``False``.
    """
    max_z = max(ORBITAL_MASK)
    table = np.zeros((max_z + 1, FULL_ORBITALS), dtype=bool)
    for atomic_number, indices in ORBITAL_MASK.items():
        table[atomic_number, list(indices)] = True
    return table


_VALIDITY_TABLE: np.ndarray = _build_validity_table()
"""Compile-time-constant per-element AO validity lookup (``numpy``, module-level)."""


def _atom_orbital_validity(atomic_numbers: Int[Array, "..."]) -> Bool[Array, "... 14"]:
    """Gather the per-atom 1-D ``(..., 14)`` AO validity mask for each atom.

    An atomic number outside the table -- above fluorine, or negative -- has an all-``False``
    mask. JAX's default gather mode promises in-bounds indices and in practice clamps a large
    one to the last row and wraps a negative one, which handed any such element fluorine's
    fourteen slots; the gather here fills them instead, as ``jax.numpy.ndarray.at`` documents.

    Args:
        atomic_numbers: Integer atomic numbers ``Z`` (any leading shape).

    Returns:
        Boolean mask of shape ``atomic_numbers.shape + (14,)``.
    """
    atomic_numbers = jnp.asarray(atomic_numbers)
    table = jnp.asarray(_VALIDITY_TABLE)
    return table.at[atomic_numbers].get(mode="fill", fill_value=False, wrap_negative_indices=False)


def atom_orbital_counts(atomic_numbers: Int[Array, "..."]) -> Int[Array, "..."]:
    """Return the number of populated AOs per atom (5 for H, 14 for C/N/O/F).

    Args:
        atomic_numbers: Integer atomic numbers ``Z`` (any leading shape).

    Returns:
        Integer AO counts of shape ``atomic_numbers.shape``.
    """
    return jnp.sum(_atom_orbital_validity(atomic_numbers), axis=-1)


def block_validity_mask(
    row_atomic_numbers: Int[Array, "..."],
    col_atomic_numbers: Int[Array, "..."] | None = None,
) -> Bool[Array, "... 14 14"]:
    r"""Return the ``(..., 14, 14)`` AO validity mask of an atom or directed pair.

    For a single atom (``col_atomic_numbers is None``) the mask is the outer
    product of the atom's per-AO validity with itself (the diagonal Fock block).
    For a directed edge it is the outer product of the **row** element's mask and
    the **column** element's mask -- ``mask[i, j] = row_valid[i] & col_valid[j]``.

    Args:
        row_atomic_numbers: Atomic numbers ``Z`` of the row (receiver) atoms.
        col_atomic_numbers: Atomic numbers ``Z`` of the column (sender) atoms; if
            ``None``, the row atoms are reused (diagonal block).

    Returns:
        Boolean mask of shape ``row_atomic_numbers.shape + (14, 14)``.
    """
    if col_atomic_numbers is None:
        col_atomic_numbers = row_atomic_numbers
    row_valid = _atom_orbital_validity(row_atomic_numbers)
    col_valid = _atom_orbital_validity(col_atomic_numbers)
    return row_valid[..., :, None] & col_valid[..., None, :]


__all__ = [
    "BACK2PYSCF",
    "FULL_ORBITALS",
    "ORBITAL_MASK",
    "PYSCF_DEF2SVP",
    "AoConvention",
    "ao_permutation",
    "atom_orbital_counts",
    "block_validity_mask",
    "native_ao_count",
]

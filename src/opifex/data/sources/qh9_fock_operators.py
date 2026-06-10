r"""GPU-resident QH9 Fock spherical-decode + block-cut as datarax operators.

The QH9 Fock target path has two host-side steps that the NumPy reference
(:mod:`opifex.data.sources.qh9_source` /
:mod:`opifex.data.sources.qh9_blocks`) runs eagerly, per molecule, in a Python
loop:

1. **Spherical decode** -- reorder + sign-flip the QH9-native def2-SVP Fock
   matrix into PySCF spherical AO ordering
   (:func:`~opifex.data.sources.qh9_source.matrix_transform_def2svp`).
2. **Block cut** -- slice the spherical Fock into the fixed ``(14, 14)`` per-atom
   diagonal and per-directed-edge off-diagonal blocks the block predictor
   regresses (:func:`~opifex.data.sources.qh9_blocks.cut_fock_to_blocks`).

Running them eagerly in NumPy pins QH9 training at a small GPU duty: the host
loop is serial and never overlaps the device. This module re-expresses both
steps as **canonical datarax** :class:`~datarax.core.operator.OperatorModule`
operators whose ``apply`` is a pure *per-molecule* (no batch dim) transform built
entirely from index gathers and elementwise multiplies. The framework batches
them by vmapping ``apply`` over the molecule axis
(:meth:`~datarax.core.operator.OperatorModule._apply_on_raw`), so both steps fuse
into the jitted predictor train step and run on device.

Index contract (produced by :class:`~opifex.data.sources.qh9_padded_source.QH9PaddedSource`)
--------------------------------------------------------------------------------
Every per-molecule element carries, padded to a fixed shape:

* ``native_fock`` ``(max_ao, max_ao)`` -- the QH9-native Fock, zero-padded.
* ``decode_perm`` ``(max_ao,)`` / ``decode_sign`` ``(max_ao,)`` -- the
  native->spherical AO permutation and signs decomposed from
  ``matrix_transform_def2svp`` (so ``spherical = native[perm][:, perm] * sign
  (x) sign``). Padding AO rows point at index ``0`` with sign ``1`` (they are
  masked out downstream by the per-element validity mask).
* ``atom_ao_start`` ``(max_atoms,)`` -- the spherical AO start offset of each
  atom's contiguous block.
* ``atom_slot_indices`` ``(max_atoms, 14)`` -- the inverse ``ORBITAL_MASK``:
  for each of the 14 block slots, the within-atom source AO offset to gather
  (valid slots map to their packed AO offset; invalid slots map to ``0`` and are
  masked out).
* ``atomic_numbers`` ``(max_atoms,)`` (padded atoms carry ``Z = 0``),
  ``edge_index`` ``(2, max_edges)`` ``(receiver, sender)``.

The block-cut operator's gather mirrors the NumPy scatter exactly: gathering slot
``s`` from within-atom offset ``atom_slot_indices[s]`` and masking invalid slots
with :func:`~opifex.neural.quantum.hamiltonian._orbital_layout.block_validity_mask`
reproduces ``cut_fock_to_blocks`` bit-for-bit (the equivalence gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx  # noqa: TC002
from jaxtyping import Array, Float, Int  # noqa: TC002

from opifex.neural.quantum.hamiltonian._orbital_layout import block_validity_mask


@dataclass(frozen=True)
class FockSphericalDecodeConfig(OperatorConfig):
    """Deterministic config for :class:`FockSphericalDecodeOperator`.

    The operator is a pure index/sign transform with no learnable parameters and
    no randomness, so the config carries only the inherited deterministic
    defaults.
    """

    def __post_init__(self) -> None:
        """Validate the inherited deterministic operator configuration."""
        super().__post_init__()


class FockSphericalDecodeOperator(OperatorModule):
    """Reorder + sign-flip a QH9-native Fock matrix into spherical AO ordering.

    Per-molecule ``apply`` realises ``spherical = native[perm][:, perm] * sign
    (x) sign`` -- the device-side equivalent of
    :func:`~opifex.data.sources.qh9_source.matrix_transform_def2svp`. The
    framework vmaps it over the molecule axis (Batch-free) via
    :meth:`~datarax.core.operator.OperatorModule._apply_on_raw`.
    """

    config: FockSphericalDecodeConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: FockSphericalDecodeConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialise the deterministic spherical-decode operator."""
        super().__init__(config, rngs=rngs)

    def apply(
        self,
        data: dict[str, Array],
        state: Any,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Array], Any, dict[str, Any] | None]:
        """Decode one molecule's native Fock into spherical AO ordering.

        Args:
            data: Per-molecule element dict carrying ``native_fock``
                ``(max_ao, max_ao)``, ``decode_perm`` ``(max_ao,)`` and
                ``decode_sign`` ``(max_ao,)``.
            state: Per-element state (passed through unchanged).
            metadata: Per-element metadata (passed through unchanged).
            random_params: Unused (the operator is deterministic).
            stats: Unused (the operator is deterministic).

        Returns:
            ``(data | {"fock": spherical}, state, metadata)`` where ``spherical``
            is the ``(max_ao, max_ao)`` spherical-ordered Fock matrix.
        """
        del random_params, stats
        native: Float[Array, "max_ao max_ao"] = data["native_fock"]
        perm: Int[Array, " max_ao"] = data["decode_perm"]
        sign = data["decode_sign"].astype(native.dtype)
        spherical = native[perm][:, perm] * sign[:, None] * sign[None, :]
        return {**data, "fock": spherical}, state, metadata


@dataclass(frozen=True)
class FockBlockCutConfig(OperatorConfig):
    """Deterministic config for :class:`FockBlockCutOperator`.

    The operator is a pure gather/mask transform with no learnable parameters and
    no randomness, so the config carries only the inherited deterministic
    defaults.
    """

    def __post_init__(self) -> None:
        """Validate the inherited deterministic operator configuration."""
        super().__post_init__()


class FockBlockCutOperator(OperatorModule):
    """Cut a spherical Fock matrix into per-atom / per-edge ``(14, 14)`` blocks.

    Per-molecule ``apply`` gathers each atom's and directed edge's spherical AO
    sub-matrix into a fixed ``(14, 14)`` block at its element's valid AO slots and
    masks the invalid slots -- the device-side equivalent of
    :func:`~opifex.data.sources.qh9_blocks.cut_fock_to_blocks`. The framework
    vmaps it over the molecule axis (Batch-free).
    """

    config: FockBlockCutConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: FockBlockCutConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialise the deterministic block-cut operator."""
        super().__init__(config, rngs=rngs)

    def apply(
        self,
        data: dict[str, Array],
        state: Any,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Array], Any, dict[str, Any] | None]:
        """Cut one molecule's spherical Fock into masked per-atom/per-edge blocks.

        Args:
            data: Per-molecule element dict carrying ``fock`` ``(max_ao,
                max_ao)`` (from :class:`FockSphericalDecodeOperator`),
                ``atom_ao_start`` ``(max_atoms,)``, ``atom_slot_indices``
                ``(max_atoms, 14)``, ``atomic_numbers`` ``(max_atoms,)`` and
                ``edge_index`` ``(2, max_edges)`` ``(receiver, sender)``.
            state: Per-element state (passed through unchanged).
            metadata: Per-element metadata (passed through unchanged).
            random_params: Unused (the operator is deterministic).
            stats: Unused (the operator is deterministic).

        Returns:
            ``(data | {"diagonal_blocks", "diagonal_mask", "off_diagonal_blocks",
            "off_diagonal_mask"}, state, metadata)`` with block shapes
            ``(max_atoms, 14, 14)`` (diagonal) and ``(max_edges, 14, 14)``
            (off-diagonal); masked blocks carry zero outside the valid AO slots.
        """
        del random_params, stats
        fock = data["fock"]
        starts = data["atom_ao_start"]
        slots = data["atom_slot_indices"]
        atomic_numbers = data["atomic_numbers"]
        edge_index = data["edge_index"]

        diagonal_mask = block_validity_mask(atomic_numbers).astype(fock.dtype)
        diagonal = _gather_blocks(fock, starts, slots, starts, slots) * diagonal_mask

        receivers, senders = edge_index[0], edge_index[1]
        off_diagonal = _gather_blocks(
            fock, starts[receivers], slots[receivers], starts[senders], slots[senders]
        )
        off_diagonal_mask = block_validity_mask(
            atomic_numbers[receivers], atomic_numbers[senders]
        ).astype(fock.dtype)

        return (
            {
                **data,
                "diagonal_blocks": diagonal,
                "diagonal_mask": diagonal_mask,
                "off_diagonal_blocks": off_diagonal * off_diagonal_mask,
                "off_diagonal_mask": off_diagonal_mask,
            },
            state,
            metadata,
        )


def _gather_blocks(
    fock: Float[Array, "max_ao max_ao"],
    row_start: Int[Array, " n"],
    row_slots: Int[Array, "n 14"],
    col_start: Int[Array, " n"],
    col_slots: Int[Array, "n 14"],
) -> Float[Array, "n 14 14"]:
    """Gather ``(14, 14)`` AO blocks from a spherical Fock matrix.

    For block ``b`` and slots ``(s, t)`` the gathered value is
    ``fock[row_start[b] + row_slots[b, s], col_start[b] + col_slots[b, t]]`` --
    the AO-slot scatter of :func:`~opifex.data.sources.qh9_blocks.cut_fock_to_blocks`
    expressed as a pure gather (invalid slots gather offset ``0`` and are zeroed
    by the caller's validity mask).

    Args:
        fock: Spherical-ordered Fock matrix ``(max_ao, max_ao)``.
        row_start: Per-block row (receiver) AO start offset ``(n,)``.
        row_slots: Per-block within-atom row AO offsets ``(n, 14)``.
        col_start: Per-block column (sender) AO start offset ``(n,)``.
        col_slots: Per-block within-atom column AO offsets ``(n, 14)``.

    Returns:
        The gathered blocks ``(n, 14, 14)``.
    """
    row_indices = row_start[:, None] + row_slots
    col_indices = col_start[:, None] + col_slots
    return fock[row_indices[:, :, None], col_indices[:, None, :]]


__all__ = [
    "FockBlockCutConfig",
    "FockBlockCutOperator",
    "FockSphericalDecodeConfig",
    "FockSphericalDecodeOperator",
]

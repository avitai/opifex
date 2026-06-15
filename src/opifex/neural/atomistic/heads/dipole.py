r"""Dipole head: molecular dipole from total-charge-conserving partial charges.

The :class:`DipoleHead` predicts total-charge-conserving per-atom partial charges
(the same constraint as :class:`~opifex.neural.atomistic.heads.charge.ChargeHead`)
and contracts them with the atomic positions to form the molecular dipole

.. math:: \boldsymbol{\mu} = \sum_i q_i\, \mathbf{r}_i .

This is the partial-charge dipole of PaiNN (Schuett, Unke & Gastegger 2021,
"Equivariant message passing for the prediction of tensorial properties and
molecular spectra", ICML) and the ``l = 0`` branch of ``../mace``'s
``compute_total_charge_dipole_permuted`` (``mace/modules/utils.py``), where
``dipole = scatter_sum(positions * charges)``. Because each :math:`q_i` is a
rotation-invariant scalar, the sum :math:`\sum_i q_i \mathbf{r}_i` transforms as a
vector: rotating the geometry by :math:`R` rotates the dipole by :math:`R` (an
:math:`l = 1` equivariant). For a neutral system (:math:`\sum_i q_i = 0`) the
dipole is additionally origin independent.

Scope: this head uses only the charge-weighted-positions term. ``../mace``'s
``AtomicDipolesMACE`` adds an atomic :math:`l = 1` dipole readout when the
backbone exposes equivariant vector features; the opifex backbones (PaiNN /
NequIP / SchNet) currently expose only invariant ``"node_features"``, so the
atomic-dipole term is omitted (it would be identically zero with no vector
input). The construction here matches PaiNN's default ``dipole_moment`` head.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.registry import register_property_head
from opifex.neural.atomistic.heads.charge import conserve_total_charge


if TYPE_CHECKING:
    from opifex.core.quantum.molecular_system import MolecularSystem


@register_property_head("dipole")
class DipoleHead(nnx.Module):
    """Molecular dipole ``sum_i q_i r_i`` from conserved partial charges.

    Args:
        feature_dim: Width of the backbone's ``"node_features"`` embedding.
        hidden_dim: Hidden width of the per-atom charge MLP. Defaults to
            ``feature_dim``.
        rngs: Random number generators (keyword-only) seeding the MLP weights.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the per-atom partial-charge MLP feeding the dipole sum."""
        super().__init__()
        width = hidden_dim if hidden_dim is not None else feature_dim
        self.hidden = nnx.Linear(feature_dim, width, rngs=rngs)
        self.readout = nnx.Linear(width, 1, rngs=rngs)

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """This head emits the molecular ``"dipole"``."""
        return ("dipole",)

    def __call__(
        self,
        system: MolecularSystem,
        graph: tuple[Array, Array],
        embeddings: dict[str, Array],
    ) -> dict[str, Array]:
        r"""Form the molecular dipole from charge-conserving partial charges.

        Args:
            system: The molecular system providing the positions and the
                conserved total ``charge``.
            graph: The ``(senders, receivers)`` edge index (unused by this head).
            embeddings: Must contain ``"node_features"`` of shape
                ``(n_atoms, feature_dim)``.

        Returns:
            ``{"dipole": Array}`` of shape ``(3,)`` -- the molecular dipole
            :math:`\sum_i q_i \mathbf{r}_i`.
        """
        del graph
        node_features = embeddings["node_features"]
        raw_charges = self.readout(nnx.silu(self.hidden(node_features)))[:, 0]
        charges = conserve_total_charge(raw_charges, system.charge)
        dipole = jnp.sum(charges[:, None] * system.positions, axis=0)
        return {"dipole": dipole}


__all__ = ["DipoleHead"]

r"""Charge head: per-atom partial charges constrained to the total system charge.

The :class:`ChargeHead` reads the backbone's per-atom scalar (``l = 0``)
embeddings, maps each to a raw partial charge with a small invariant MLP, and
then enforces *total-charge conservation* so the per-atom charges sum exactly to
the system's net charge :math:`Q`:

.. math::
   q_i = \tilde q_i - \frac{1}{N}\left(\sum_j \tilde q_j - Q\right) .

This is the per-atom excess correction used by ``../mace``
(``AtomicDipolesMACE`` in ``mace/modules/models.py`` subtracts
``scatter_mean(charges) - total_charge / num_atoms``) and underlies the
charge-equilibration-free partial charges of PaiNN (Schuett, Unke & Gastegger
2021, "Equivariant message passing for the prediction of tensorial properties
and molecular spectra", ICML). Because each charge is a per-atom invariant
scalar (it depends only on the rotation-invariant ``node_features``) and the
correction is permutation invariant, the partial charges are E(3)- and
permutation-equivariant scalars -- the contract a downstream dipole / Coulomb
readout relies on.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001
from opifex.core.quantum.registry import register_property_head


def conserve_total_charge(raw_charges: Array, total_charge: float | Array) -> Array:
    r"""Shift per-atom charges so they sum exactly to ``total_charge``.

    Applies the uniform per-atom excess correction

    .. math:: q_i = \tilde q_i - \frac{1}{N}\Big(\sum_j \tilde q_j - Q\Big),

    so that :math:`\sum_i q_i = Q` (``../mace`` ``AtomicDipolesMACE`` excess
    subtraction; Schuett et al. 2021, PaiNN). The correction is uniform across
    atoms, so it preserves the relative (rotation-invariant) charge pattern.

    Args:
        raw_charges: Unconstrained per-atom charges of shape ``(n_atoms,)``.
        total_charge: The system's net charge :math:`Q` to conserve to.

    Returns:
        Per-atom charges of shape ``(n_atoms,)`` summing to ``total_charge``.
    """
    n_atoms = raw_charges.shape[0]
    excess = (jnp.sum(raw_charges) - total_charge) / n_atoms
    return raw_charges - excess


@register_property_head("charges")
class ChargeHead(nnx.Module):
    """Total-charge-conserving per-atom partial-charge readout.

    Args:
        feature_dim: Width of the backbone's ``"node_features"`` embedding.
        hidden_dim: Hidden width of the per-atom MLP. Defaults to ``feature_dim``.
        rngs: Random number generators (keyword-only) seeding the MLP weights.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the per-atom partial-charge MLP."""
        super().__init__()
        width = hidden_dim if hidden_dim is not None else feature_dim
        self.hidden = nnx.Linear(feature_dim, width, rngs=rngs)
        self.readout = nnx.Linear(width, 1, rngs=rngs)

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """This head emits per-atom ``"charges"``."""
        return ("charges",)

    def __call__(
        self,
        system: MolecularSystem,
        graph: tuple[Array, Array],
        embeddings: dict[str, Array],
    ) -> dict[str, Array]:
        """Map per-atom embeddings to charge-conserving partial charges.

        Args:
            system: The molecular system; its ``charge`` is the conserved total.
            graph: The ``(senders, receivers)`` edge index (unused by this head).
            embeddings: Must contain ``"node_features"`` of shape
                ``(n_atoms, feature_dim)``.

        Returns:
            ``{"charges": Array}`` of shape ``(n_atoms,)`` whose entries sum to
            ``system.charge``.
        """
        del graph
        node_features = embeddings["node_features"]
        raw_charges = self.readout(nnx.silu(self.hidden(node_features)))[:, 0]
        charges = conserve_total_charge(raw_charges, system.charge)
        return {"charges": charges}


__all__ = ["ChargeHead", "conserve_total_charge"]

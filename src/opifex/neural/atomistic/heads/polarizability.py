r"""Polarizability head: molecular polarizability tensor from per-atom readouts.

The :class:`PolarizabilityHead` predicts the molecular electric polarizability
:math:`\boldsymbol{\alpha}`, a **symmetric rank-2 Cartesian tensor** that
transforms under a rotation :math:`R \in \mathrm{SO}(3)` as

.. math:: \boldsymbol{\alpha}(R\mathbf{r}) = R\,\boldsymbol{\alpha}(\mathbf{r})\,R^\top .

Following the standard ML decomposition of a symmetric rank-2 tensor into its
irreducible (:math:`l = 0`) isotropic and (:math:`l = 2`) symmetric-traceless
parts (Schuett, Unke & Gastegger 2021, "Equivariant message passing for the
prediction of tensorial properties and molecular spectra", ICML -- the PaiNN
polarizability readout; mirrored by ``../mace``'s polarizability readout), the
head assembles

.. math::
   \boldsymbol{\alpha}
   = \sum_i \Big[\, s_i\,\mathbf{I}
       + q_i\,\big(3\,\mathbf{r}_i \mathbf{r}_i^\top - |\mathbf{r}_i|^2 \mathbf{I}\big) \Big],

where :math:`s_i` (per-atom isotropic scalar) and :math:`q_i` (per-atom
anisotropic weight) are **rotation-invariant** scalars read from the backbone's
``"node_features"`` by small MLPs. The first term contributes the isotropic
:math:`l = 0` part; the second is the Cartesian symmetric-traceless
(:math:`l = 2`) quadrupole-like form :math:`3\,\mathbf{r}\mathbf{r}^\top -
|\mathbf{r}|^2\mathbf{I}`, which is traceless by construction. Because each
:math:`s_i, q_i` is invariant while :math:`\mathbf{r}_i \mathbf{r}_i^\top`
transforms as :math:`R\,\mathbf{r}_i\mathbf{r}_i^\top R^\top` and
:math:`\mathbf{I}` is invariant, the assembled tensor is the required
:math:`l = 2 \oplus l = 0` equivariant. Each per-atom term is manifestly
symmetric, so the molecular sum is symmetric.

Scope: like :class:`~opifex.neural.atomistic.heads.dipole.DipoleHead`, this head
uses only the position-outer-product (charge-weighted) construction. The opifex
backbones (PaiNN / NequIP / SchNet) currently expose only the invariant
``"node_features"``; building the :math:`l = 2` part directly from backbone
equivariant vector/tensor features (as ``../mace``'s polarizability readout does
when such features exist) is a future variant plugging into the same
:class:`~opifex.core.quantum.protocols.PropertyHead` protocol. Setting
``isotropic_only=True`` drops the anisotropic term, yielding a multiple of the
identity (a pure :math:`l = 0` polarizability).
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001
from opifex.core.quantum.registry import register_property_head


logger = logging.getLogger(__name__)


@register_property_head("polarizability")
class PolarizabilityHead(nnx.Module):
    r"""Symmetric ``3x3`` molecular polarizability from per-atom invariants.

    Reads the backbone's per-atom invariant ``"node_features"`` and assembles the
    polarizability as an isotropic (:math:`l = 0`) part plus a symmetric-traceless
    (:math:`l = 2`) part built from position outer products (Schuett et al. 2021,
    PaiNN; ``../mace`` polarizability readout).

    Args:
        feature_dim: Width of the backbone's ``"node_features"`` embedding.
        hidden_dim: Hidden width of the per-atom MLP. Defaults to ``feature_dim``.
        isotropic_only: If ``True``, emit only the isotropic :math:`l = 0` part
            (a multiple of the identity), dropping the anisotropic term.
        rngs: Random number generators (keyword-only) seeding the MLP weights.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int | None = None,
        isotropic_only: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the per-atom isotropic/anisotropic scalar MLP."""
        super().__init__()
        width = hidden_dim if hidden_dim is not None else feature_dim
        # Two invariant scalar channels per atom: isotropic s_i and anisotropic q_i.
        n_channels = 1 if isotropic_only else 2
        # Static structural config (it also fixes the readout width), so a plain
        # Python attribute -- not an nnx leaf -- keeps it jit-static.
        self.isotropic_only = isotropic_only
        self.hidden = nnx.Linear(feature_dim, width, rngs=rngs)
        self.readout = nnx.Linear(width, n_channels, rngs=rngs)

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """This head emits the molecular ``"polarizability"`` tensor."""
        return ("polarizability",)

    def __call__(
        self,
        system: MolecularSystem,
        graph: tuple[Array, Array],
        embeddings: dict[str, Array],
    ) -> dict[str, Array]:
        r"""Assemble the symmetric ``3x3`` molecular polarizability tensor.

        Args:
            system: The molecular system providing the atomic ``positions``.
            graph: The ``(senders, receivers)`` edge index (unused by this head).
            embeddings: Must contain ``"node_features"`` of shape
                ``(n_atoms, feature_dim)``.

        Returns:
            ``{"polarizability": Array}`` of shape ``(3, 3)`` -- the symmetric
            molecular polarizability tensor
            :math:`\sum_i [s_i\mathbf{I} + q_i(3\,\mathbf{r}_i\mathbf{r}_i^\top -
            |\mathbf{r}_i|^2\mathbf{I})]`.
        """
        del graph
        node_features = embeddings["node_features"]
        per_atom = self.readout(nnx.silu(self.hidden(node_features)))
        positions = system.positions
        identity = jnp.eye(3, dtype=positions.dtype)

        isotropic_scalar = per_atom[:, 0]
        isotropic = jnp.sum(isotropic_scalar)[..., None, None] * identity

        if self.isotropic_only:
            return {"polarizability": isotropic}

        anisotropic_weight = per_atom[:, 1]
        # Cartesian l=2 symmetric-traceless form: 3 r r^T - |r|^2 I per atom.
        outer = positions[:, :, None] * positions[:, None, :]
        squared_norm = jnp.sum(positions**2, axis=-1)
        traceless = 3.0 * outer - squared_norm[:, None, None] * identity[None, :, :]
        anisotropic = jnp.sum(anisotropic_weight[:, None, None] * traceless, axis=0)

        return {"polarizability": isotropic + anisotropic}


__all__ = ["PolarizabilityHead"]

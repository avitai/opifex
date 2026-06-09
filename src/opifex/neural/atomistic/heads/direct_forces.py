r"""Direct (non-conservative) force head: an equivariant ``l = 1`` readout.

The :class:`DirectForcesHead` implements the *direct* force strategy: it predicts
the per-atom force vector

.. math:: \mathbf{F}_i = \sum_c g_c(\mathbf{s}_i)\, \mathbf{v}_{i,c}

directly from the backbone's per-atom **equivariant vector** features
:math:`\mathbf{v}_{i,c} \in \mathbb{R}^3` (channel ``c``), gated by an invariant
scalar :math:`g_c(\mathbf{s}_i)` read from the invariant ``node_features``. It
**never differentiates an energy** -- there is no :data:`ENERGY_FN_KEY` closure
and no :func:`jax.grad` call -- contrasting with the conservative
:class:`~opifex.neural.atomistic.heads.forces.ForcesHead` (forces as
``-grad(E)``).

Both strategies satisfy the same
:class:`opifex.core.quantum.protocols.PropertyHead` protocol and emit
``("forces",)``; which one a model uses is a *strategy* choice, not hardcoded
(Orb proves both matter -- Neumann et al. 2024, "Orb: A Fast, Scalable Neural
Network Potential"; fairchem's GemNet / eSCN direct-force heads).

**Equivariance / conservativeness trade-off (documented):**

* *Equivariant.* A scalar-times-vector contraction of an equivariant ``l = 1``
  channel by an invariant ``l = 0`` gate is itself ``l = 1`` equivariant:
  rotating the geometry by :math:`R` (so :math:`\mathbf{v}_{i,c} \to R\,
  \mathbf{v}_{i,c}` while :math:`\mathbf{s}_i` is unchanged) rotates the force by
  :math:`R`. This is the PaiNN gated vector readout (Schuett, Unke & Gastegger
  2021, eq. 9-10), used here as a force head rather than a dipole head.
* *Not guaranteed conservative.* Because the force is read out directly rather
  than as :math:`-\partial E/\partial \mathbf{r}`, the predicted field has no
  scalar potential in general (:math:`\nabla \times \mathbf{F} \neq 0`), so
  energy is not exactly conserved in long molecular-dynamics trajectories. This
  is the well-documented Orb / fairchem trade-off: direct forces are cheaper
  (one network pass, no second-order autodiff) and often more accurate
  pointwise, at the cost of strict energy conservation. Choose
  :class:`ForcesHead` when a conserved Hamiltonian is required.

**Backbone requirement.** This head consumes a per-atom equivariant vector
channel under the reserved ``embeddings`` key :data:`VECTOR_FEATURES_KEY`, of
shape ``(n_atoms, 3, feature_dim)`` (the PaiNN vector-feature layout). The
opifex backbones currently surface only invariant ``node_features`` in their
public output (the PaiNN/NequIP vector channels are internal state); an
equivariant backbone must expose its vector channel under this key for the
direct head to be usable. Translation invariance follows from the backbone
building those vectors from relative geometry (edge directions), exactly as
:class:`ForcesHead` derives translation-invariant gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.core.quantum.registry import register_property_head


if TYPE_CHECKING:
    from opifex.core.quantum.molecular_system import MolecularSystem


VECTOR_FEATURES_KEY = "node_vectors"
"""Reserved ``embeddings`` key for per-atom equivariant vectors ``(n_atoms, 3, F)``."""


@register_property_head("direct_forces")
class DirectForcesHead(nnx.Module):
    r"""Direct equivariant force readout ``F_i = sum_c g_c(s_i) v_{i,c}``.

    Maps the backbone's invariant scalars to a per-channel gate, mixes the
    equivariant vector channels with a bias-free linear map (a rotation-commuting
    combination over channels), and contracts the two into a per-atom ``(3,)``
    force. Owns parameters (the gate MLP and the vector mixing), unlike the
    stateless conservative :class:`ForcesHead`.

    Args:
        feature_dim: Width ``F`` of the backbone's ``"node_features"`` and of the
            equivariant vector channel under :data:`VECTOR_FEATURES_KEY`.
        hidden_dim: Hidden width of the invariant gate MLP. Defaults to
            ``feature_dim``.
        rngs: Random number generators (keyword-only) seeding the weights.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the invariant gate MLP and the equivariant vector mixing map."""
        super().__init__()
        width = hidden_dim if hidden_dim is not None else feature_dim
        self.gate_hidden = nnx.Linear(feature_dim, width, rngs=rngs)
        self.gate_out = nnx.Linear(width, feature_dim, rngs=rngs)
        # Bias-free linear over channels (spatial axis untouched) stays equivariant.
        self.vector_mix = nnx.Linear(feature_dim, feature_dim, use_bias=False, rngs=rngs)

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """This head emits ``"forces"``."""
        return ("forces",)

    def __call__(
        self,
        system: MolecularSystem,
        graph: tuple[Array, Array],
        embeddings: dict[str, Array],
    ) -> dict[str, Array]:
        r"""Return ``{"forces": F}`` of shape ``(n_atoms, 3)`` as a direct readout.

        Args:
            system: The molecular system (unused beyond shape; the force comes
                from features, not from differentiating the geometry).
            graph: The ``(senders, receivers)`` edge index (unused by this head).
            embeddings: Must contain ``"node_features"`` of shape
                ``(n_atoms, feature_dim)`` and the equivariant vector channel
                :data:`VECTOR_FEATURES_KEY` of shape ``(n_atoms, 3, feature_dim)``.

        Returns:
            ``{"forces": Array}`` with the direct, equivariant per-atom forces.

        Raises:
            KeyError: If the equivariant vector channel was not injected; the
                message names the required equivariant backbone.
        """
        del system, graph
        if VECTOR_FEATURES_KEY not in embeddings:
            raise KeyError(
                f"DirectForcesHead requires per-atom equivariant vectors under "
                f"{VECTOR_FEATURES_KEY!r} (shape (n_atoms, 3, feature_dim)); only an "
                "equivariant backbone (PaiNN/NequIP) exposing its vector channel "
                "supports the direct-force strategy."
            )
        node_features: Float[Array, "n_atoms feature_dim"] = embeddings["node_features"]
        vectors: Float[Array, "n_atoms 3 feature_dim"] = embeddings[VECTOR_FEATURES_KEY]
        gate = self.gate_out(nnx.silu(self.gate_hidden(node_features)))
        mixed = self.vector_mix(vectors)
        forces = jnp.sum(gate[:, None, :] * mixed, axis=-1)
        return {"forces": forces}


__all__ = ["VECTOR_FEATURES_KEY", "DirectForcesHead"]

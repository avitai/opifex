r"""Evidential energy head: single-pass Deep-Evidential-Regression UQ for energy.

The :class:`EvidentialEnergyHead` is the uncertainty-aware analogue of
:class:`~opifex.neural.atomistic.heads.energy.EnergyHead`. It reads the
backbone's per-atom scalar (``l = 0``) embeddings, maps each atom to a
Normal-Inverse-Gamma evidential output :math:`(\gamma_i, \nu_i, \alpha_i,
\beta_i)` (Amini et al. 2020, *Deep Evidential Regression*, NeurIPS) with a
small invariant MLP, and aggregates the per-atom evidential distributions into a
**total-energy** NIG distribution. A single deterministic forward pass therefore
yields the total energy *and* its aleatoric / epistemic uncertainty -- no
ensemble, no sampling (Tan et al. 2025, *Evidential deep learning for
interatomic potentials* (eIP), Nat. Commun.; arXiv:2407.13994).

Aggregation choice (documented, bounded)
----------------------------------------
The eIP paper (arXiv:2407.13994) keeps uncertainty at the **per-atom** level and
does *not* prescribe a closed-form rule for combining per-atom NIG *parameters*
into a single molecular NIG (its configuration-level metric, Eq. 7, merely
*averages* per-atom :math:`\sigma_i`). The total energy is, however, the
extensive sum of atomic energies :math:`E = \sum_i \gamma_i`. We therefore
adopt the standard *variance-of-a-sum* aggregation under the atomic-independence
assumption:

.. math::
   \gamma = \sum_i \gamma_i,\qquad
   \mathrm{Var}_{\mathrm{ale}} = \sum_i \frac{\beta_i}{\alpha_i - 1},\qquad
   \mathrm{Var}_{\mathrm{epi}} = \sum_i \frac{\beta_i}{\nu_i(\alpha_i - 1)} .

These summed variances are the load-bearing uncertainty outputs. To expose a
*self-consistent* total NIG :math:`(\gamma, \nu, \alpha, \beta)` for downstream
evidential losses / :func:`nig_to_predictive_distribution`, we back out total
parameters that reproduce exactly the summed moments above:

.. math::
   \alpha = 1 + \sum_i(\alpha_i - 1),\qquad
   \beta = \mathrm{Var}_{\mathrm{ale}}\,(\alpha - 1),\qquad
   \nu = \frac{\mathrm{Var}_{\mathrm{ale}}}{\mathrm{Var}_{\mathrm{epi}}} .

By construction :math:`\beta/(\alpha-1) = \mathrm{Var}_{\mathrm{ale}}` and
:math:`\beta/(\nu(\alpha-1)) = \mathrm{Var}_{\mathrm{epi}}`, so the total NIG and
the summed variances agree and ``total = aleatoric + epistemic`` holds (the
:class:`~opifex.uncertainty.types.PredictiveDistribution` additivity contract).
The ``alpha`` rule keeps the total evidence extensive (the sum of per-atom
evidences :math:`\alpha_i - 1`), consistent with energy being extensive.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001
from opifex.core.quantum.registry import register_property_head
from opifex.neural.dtypes import default_float_dtype
from opifex.uncertainty.evidential import (
    aleatoric_variance,
    epistemic_variance,
    NIGParams,
    positive_evidential_params,
)


# Output keys this head writes. Centralised so the head's ``__call__`` and
# ``implemented_properties`` stay in lock-step (single source of truth).
_ENERGY_KEY = "energy"
_NU_KEY = "energy_nu"
_ALPHA_KEY = "energy_alpha"
_BETA_KEY = "energy_beta"
_ALEATORIC_KEY = "energy_aleatoric_var"
_EPISTEMIC_KEY = "energy_epistemic_var"
_VARIANCE_KEY = "energy_variance"

# Number of NIG channels emitted per atom: (gamma, nu, alpha, beta).
_NIG_CHANNELS = 4


def aggregate_total_energy_nig(per_atom: NIGParams) -> NIGParams:
    r"""Aggregate per-atom NIG parameters into a self-consistent total-energy NIG.

    Implements the documented *variance-of-a-sum* aggregation (see module
    docstring): the total mean is the sum of per-atom means; the total aleatoric
    / epistemic variances are the sums of the per-atom variances; and the
    returned :math:`(\nu, \alpha, \beta)` are the unique parameters reproducing
    those summed moments with extensive total evidence
    :math:`\alpha - 1 = \sum_i(\alpha_i - 1)`.

    Args:
        per_atom: Per-atom NIG parameters of shape ``(n_atoms,)`` each.

    Returns:
        Scalar (0-d) :class:`NIGParams` for the total energy.
    """
    gamma_total = jnp.sum(per_atom.gamma)
    aleatoric_total = jnp.sum(aleatoric_variance(per_atom))
    epistemic_total = jnp.sum(epistemic_variance(per_atom))

    alpha_total = 1.0 + jnp.sum(per_atom.alpha - 1.0)
    beta_total = aleatoric_total * (alpha_total - 1.0)
    nu_total = aleatoric_total / epistemic_total
    return NIGParams(gamma=gamma_total, nu=nu_total, alpha=alpha_total, beta=beta_total)


@register_property_head("evidential_energy")
class EvidentialEnergyHead(nnx.Module):
    r"""Single-pass evidential (NIG) total-energy readout with built-in UQ.

    Emits the total energy and its evidential parameters / uncertainty
    decomposition in one deterministic forward pass (Amini 2020; eIP
    arXiv:2407.13994). See the module docstring for the aggregation rule.

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
        """Build the per-atom evidential MLP emitting 4 NIG channels per atom."""
        super().__init__()
        width = hidden_dim if hidden_dim is not None else feature_dim
        dtype = default_float_dtype()
        self.hidden = nnx.Linear(feature_dim, width, param_dtype=dtype, rngs=rngs)
        self.readout = nnx.Linear(width, _NIG_CHANNELS, param_dtype=dtype, rngs=rngs)

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """Total energy plus its evidential-NIG and uncertainty-variance keys."""
        return (
            _ENERGY_KEY,
            _NU_KEY,
            _ALPHA_KEY,
            _BETA_KEY,
            _ALEATORIC_KEY,
            _EPISTEMIC_KEY,
            _VARIANCE_KEY,
        )

    def per_atom_params(self, node_features: Array) -> NIGParams:
        """Map per-atom embeddings to per-atom NIG parameters.

        Args:
            node_features: Per-atom invariant embeddings of shape
                ``(n_atoms, feature_dim)``.

        Returns:
            Per-atom :class:`NIGParams` (each field shape ``(n_atoms,)``) with
            ``nu, beta > 0`` and ``alpha > 1``.
        """
        raw = self.readout(nnx.silu(self.hidden(node_features)))
        return positive_evidential_params(raw)

    def __call__(
        self,
        system: MolecularSystem,
        graph: tuple[Array, Array],
        embeddings: dict[str, Array],
    ) -> dict[str, Array]:
        r"""Map per-atom embeddings to a total-energy evidential distribution.

        Args:
            system: The molecular system (unused; the energy is the bare sum of
                per-atom evidential means).
            graph: The ``(senders, receivers)`` edge index (unused by this head).
            embeddings: Must contain ``"node_features"`` of shape
                ``(n_atoms, feature_dim)``.

        Returns:
            A dict with the scalar total ``"energy"`` (:math:`\\gamma`), its
            total NIG parameters (``"energy_nu"``, ``"energy_alpha"``,
            ``"energy_beta"``) and the variance decomposition
            (``"energy_aleatoric_var"``, ``"energy_epistemic_var"``,
            ``"energy_variance"``).
        """
        del system, graph
        per_atom = self.per_atom_params(embeddings["node_features"])
        total = aggregate_total_energy_nig(per_atom)
        aleatoric = aleatoric_variance(total)
        epistemic = epistemic_variance(total)
        return {
            _ENERGY_KEY: total.gamma,
            _NU_KEY: total.nu,
            _ALPHA_KEY: total.alpha,
            _BETA_KEY: total.beta,
            _ALEATORIC_KEY: aleatoric,
            _EPISTEMIC_KEY: epistemic,
            _VARIANCE_KEY: aleatoric + epistemic,
        }


__all__ = ["EvidentialEnergyHead", "aggregate_total_energy_nig"]

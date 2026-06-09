r"""Evidential (Deep Evidential Regression) distribution adapter.

Wraps the Normal-Inverse-Gamma (NIG) evidential parameters emitted by a
single-pass evidential head -- Amini et al. 2020 "Deep Evidential Regression"
(NeurIPS); evidential interatomic potentials, eIP arXiv:2407.13994 -- into an
Opifex :class:`~opifex.uncertainty.types.PredictiveDistribution` using the
closed-form NIG moments, attaching ``source_package`` provenance.

This is the model->UQ seam by which an atomistic potential's optional
:class:`~opifex.neural.atomistic.heads.evidential.EvidentialEnergyHead`
registers with the existing ``uncertainty/adapters`` mechanism, exactly like
:class:`~opifex.uncertainty.distributions.ArtifexDistributionAdapter` wraps an
Artifex distribution. The numerical mapping lives once in
:func:`opifex.uncertainty.evidential.nig_to_predictive_distribution` (DRY); this
adapter only marshals a :class:`NIGParams` through it and tags provenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opifex.uncertainty.evidential import nig_to_predictive_distribution


if TYPE_CHECKING:
    from opifex.uncertainty.evidential import NIGParams
    from opifex.uncertainty.types import PredictiveDistribution


class EvidentialAdapter:
    """Wrap NIG evidential parameters into a :class:`PredictiveDistribution`.

    Satisfies :class:`~opifex.uncertainty.adapters.base.DistributionAdapterProtocol`
    structurally: :meth:`from_distribution` accepts the :class:`NIGParams`
    produced by an evidential head and returns the predictive distribution with
    the epistemic / aleatoric variance decomposition.
    """

    def from_distribution(self, distribution: NIGParams) -> PredictiveDistribution:
        """Map NIG parameters to a predictive distribution (Amini 2020; eIP 2024).

        Args:
            distribution: The predicted Normal-Inverse-Gamma parameters.

        Returns:
            A predictive distribution carrying the evidential mean and the
            epistemic / aleatoric / total variance split, tagged with the
            ``deep_evidential_regression`` method and ``opifex`` source package.
        """
        return nig_to_predictive_distribution(
            distribution, metadata=(("source_package", "opifex"),)
        )

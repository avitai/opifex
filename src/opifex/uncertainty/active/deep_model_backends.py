r"""Deep-model backends for active learning — Slice 23 (audit finding #4b).

Phase 8 Task 8.3 (``08-...:586-601``) requires
``active/deep_model_backends.py`` wiring deep-GP / deep-ensemble
model surfaces from :mod:`opifex.neural.bayesian` into the active-
learning acquisition loop. Reference: ``../trieste/models/gpflow``,
``../trieste/models/gpflux``, ``../trieste/models/keras``.

The opifex port is intentionally lightweight: the AL loop only needs
a ``predict(x) -> PredictiveDistribution`` callable. The two backend
classes here adapt user-supplied prediction functions (one per
ensemble member, or one for the whole deep model) into that shape.

References
----------
* Lakshminarayanan, Pritzel, Blundell 2017 — *Simple and Scalable
  Predictive Uncertainty Estimation using Deep Ensembles*, NIPS.
* Damianou, Lawrence 2013 — *Deep Gaussian Processes*, AISTATS.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from opifex.uncertainty.types import PredictiveDistribution


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepModelBackend:
    """Adapt one deep-model ``predict`` function to the AL backend protocol.

    The user supplies a callable
    ``predict_fn: (x: jax.Array) -> PredictiveDistribution`` and an
    optional ``source_package`` provenance string. The backend's
    :meth:`predict` simply forwards to ``predict_fn``.

    Attributes:
        predict_fn: User-supplied prediction function.
        source_package: Provenance tag (e.g.
            ``"opifex.neural.bayesian"``) for metadata wiring.
    """

    predict_fn: Callable[[jax.Array], PredictiveDistribution]
    source_package: str = "opifex.neural.bayesian"

    def predict(self, x: jax.Array) -> PredictiveDistribution:
        """Forward ``x`` to the wrapped prediction function."""
        return self.predict_fn(x)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepEnsembleBackend:
    """Aggregate predictions across ``M`` deep-ensemble members.

    Ensemble mean = ``(1/M) Σ_m μ_m(x)``.
    Ensemble variance = ``(1/M) Σ_m σ²_m(x) + Var_m(μ_m(x))`` —
    Lakshminarayanan+ 2017 eq. 5 (the variance decomposition that
    captures both aleatoric and epistemic uncertainty).
    """

    member_predict_fns: tuple[Callable[[jax.Array], PredictiveDistribution], ...]
    source_package: str = "opifex.neural.bayesian"
    metadata: tuple[tuple[str, object], ...] = field(default_factory=tuple)

    def predict(self, x: jax.Array) -> PredictiveDistribution:
        """Return the ensemble-aggregated predictive distribution at ``x``."""
        if len(self.member_predict_fns) == 0:
            raise ValueError("DeepEnsembleBackend requires at least one member.")
        member_means = []
        member_variances = []
        for predict_fn in self.member_predict_fns:
            pd = predict_fn(x)
            member_means.append(pd.mean)
            if pd.variance is None:
                raise ValueError(
                    "DeepEnsembleBackend requires each member to expose a "
                    "non-None PredictiveDistribution.variance."
                )
            member_variances.append(pd.variance)
        stacked_means = jnp.stack(member_means, axis=0)
        stacked_variances = jnp.stack(member_variances, axis=0)
        ensemble_mean = jnp.mean(stacked_means, axis=0)
        epistemic = jnp.var(stacked_means, axis=0)
        aleatoric = jnp.mean(stacked_variances, axis=0)
        total = epistemic + aleatoric
        return PredictiveDistribution(
            mean=ensemble_mean,
            variance=total,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total_uncertainty=total,
            metadata=self.metadata,
        )


__all__ = ["DeepEnsembleBackend", "DeepModelBackend"]

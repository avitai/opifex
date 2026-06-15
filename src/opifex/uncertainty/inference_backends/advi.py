r"""Automatic Differentiation Variational Inference (ADVI) peer backend.

ADVI fits a mean-field Gaussian variational posterior on an
unconstrained reparametrisation of the model parameters. The ELBO is
optimised via stochastic gradient ascent over Monte-Carlo estimates of
the expectation.

Canonical reference:
* Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., Blei, D. M. 2017
  — *Automatic Differentiation Variational Inference*, JMLR 18(14).

The algorithm itself is vendored in
:mod:`opifex.uncertainty.inference_backends._advi_algorithm` as a
line-by-line port of ``../blackjax/blackjax/vi/meanfield_vi.py`` plus
its shared Gaussian-VI helpers at
``../blackjax/blackjax/vi/_gaussian_vi.py``.

The ``predict_distribution`` and ``posterior_predictive`` hooks re-fit the
backend from the stored ``target_log_prob`` and route the parameter-space
draws through
:func:`opifex.uncertainty._predictive.predictive_from_parameter_samples`
(model-aware when a ``predict_fn`` forward model is supplied, lightweight
parameter-moment stand-in otherwise).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable  # noqa: TC003 — kept eager per opifex convention
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx  # noqa: TC002

from opifex.uncertainty.inference_backends._advi_algorithm import (
    approximate,
    draw,
)
from opifex.uncertainty.inference_backends._peer_predictive import (
    peer_predictive_from_refit,
)
from opifex.uncertainty.inference_backends.base import (
    BackendDiagnostics,
    BackendResult,
)
from opifex.uncertainty.types import (
    PredictiveDistribution,  # noqa: TC001
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ADVIBackend:
    """ADVI mean-field variational inference backend (Kucukelbir+ 2017).

    Attributes:
        name: Backend identifier used by the router.
        source_package: Backend implementation lives in opifex.
        method_names: Sampler-method identifiers handled by this backend.
        family: ``"meanfield"`` (the implemented family). ``"fullrank"``
            is reserved for the matching follow-up slice and currently
            raises ``NotImplementedError`` from ``fit``.
        notes: Free-text rationale shown in capability reports.
        init_state: Starting position used to infer the parameter
            PyTree shape, shape ``(d,)``.
        target_log_prob: Optional stored log-density callable. When set, the
            ``predict_distribution`` / ``posterior_predictive`` hooks re-fit
            the backend from it; ``fit`` also falls back to it when no
            ``target_log_prob`` is threaded through the call.
        num_samples: Number of draws to return from the fitted Gaussian.
        num_iterations: Number of Adam steps minimising the ELBO.
        num_mc_samples: Number of Monte Carlo samples per ELBO step.
        learning_rate: Adam learning rate.
        stl_estimator: Whether to use the stick-the-landing gradient
            estimator (Roeder+ 2017). Lowers gradient variance under
            reverse-KL; recommended on by Agrawal+ 2020.
    """

    name: str = "advi"
    source_package: str = "opifex"
    method_names: tuple[str, ...] = ("advi",)
    family: Literal["meanfield", "fullrank"] = "meanfield"
    notes: str = (
        "ADVI (Kucukelbir+ 2017) — mean-field Gaussian variational "
        "posterior optimised over the ELBO via automatic-differentiation "
        "reparametrisation. Algorithm ported from "
        "blackjax/vi/meanfield_vi.py."
    )
    init_state: jax.Array = dataclasses.field(default_factory=lambda: jnp.zeros(1))
    target_log_prob: Callable[[jax.Array], jax.Array] | None = None
    num_samples: int = 512
    num_iterations: int = 400
    num_mc_samples: int = 8
    learning_rate: float = 0.05
    stl_estimator: bool = True

    def fit(
        self,
        target_log_prob: Callable[[jax.Array], jax.Array] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> BackendResult:
        """Run mean-field ADVI and return ``num_samples`` draws from the fit.

        ``target_log_prob`` overrides the stored :attr:`target_log_prob` when
        supplied; if both are ``None`` a :class:`ValueError` is raised (mirroring
        the BlackJAX backend's fit contract).

        ``sampler_state`` is the array of drawn samples with shape
        ``(num_samples, d)``; diagnostics carries no MCMC mixing
        statistics for VI.
        """
        if self.family != "meanfield":
            raise NotImplementedError(
                f"{self.name!r} backend only ships the mean-field family; "
                f"got family={self.family!r}. Full-rank ADVI lands in a "
                f"follow-up slice."
            )
        log_density_fn = target_log_prob if target_log_prob is not None else self.target_log_prob
        if log_density_fn is None:
            raise ValueError(
                f"{self.name!r} backend.fit requires a 'target_log_prob': pass one "
                "to fit(...) or construct the backend with target_log_prob=<callable>."
            )

        approx_key = rngs.sampler() if "sampler" in rngs else rngs.default()
        sample_key = rngs.sampler() if "sampler" in rngs else rngs.default()
        state = approximate(
            rng_key=approx_key,
            log_density_fn=log_density_fn,
            initial_position=self.init_state,
            num_iterations=self.num_iterations,
            num_mc_samples=self.num_mc_samples,
            learning_rate=self.learning_rate,
            stl_estimator=self.stl_estimator,
        )
        samples = draw(sample_key, state, num_samples=self.num_samples)
        return BackendResult(
            sampler_state=samples,
            diagnostics=BackendDiagnostics(),
        )

    def predict_distribution(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs,
        predict_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    ) -> PredictiveDistribution:
        """Re-fit ADVI and map the posterior draws to a predictive at ``x``.

        Model-aware when ``predict_fn`` is supplied (the genuine posterior
        predictive, marginalising the forward model over the variational draws);
        otherwise the lightweight parameter-moment stand-in. Requires a stored
        :attr:`target_log_prob` (re-fit needs a log density).
        """
        return peer_predictive_from_refit(
            self,
            x,
            rngs=rngs,
            predict_fn=predict_fn,
            metadata=(
                ("method", "predict_distribution"),
                ("backend", self.name),
                ("num_samples", self.num_samples),
            ),
        )

    def posterior_predictive(
        self,
        rngs: nnx.Rngs,
        x: jax.Array,
        predict_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    ) -> PredictiveDistribution:
        """Re-fit ADVI and return the posterior predictive at ``x``.

        Same parameter-draw marginalisation as :meth:`predict_distribution`
        with a ``posterior_predictive`` method tag.
        """
        return peer_predictive_from_refit(
            self,
            x,
            rngs=rngs,
            predict_fn=predict_fn,
            metadata=(
                ("method", "posterior_predictive"),
                ("backend", self.name),
                ("num_samples", self.num_samples),
            ),
        )


__all__ = ["ADVIBackend"]

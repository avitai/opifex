r"""Pathfinder variational-inference peer-sampler backend.

Pathfinder (Zhang+ 2022, arXiv:2108.03782) runs a quasi-Newton L-BFGS
optimisation along the negative log-density of the target, fits a
Gaussian variational approximation at each step using the inverse
Hessian factors produced by L-BFGS, scores each by an ELBO, and
returns draws from the best one.

The algorithm itself is vendored in
:mod:`opifex.uncertainty.inference_backends._pathfinder_algorithm`
as a line-by-line port of ``../blackjax/blackjax/vi/pathfinder.py``
plus its L-BFGS helpers at ``../blackjax/blackjax/optimizers/lbfgs.py``.

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

import jax
import jax.numpy as jnp
from flax import nnx  # noqa: TC002

from opifex.uncertainty.inference_backends._pathfinder_algorithm import (
    pathfinder_approximate,
    pathfinder_sample,
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
class PathfinderBackend:
    """Pathfinder variational inference backend (Zhang+ 2022).

    Attributes:
        name: Backend identifier used by the router.
        source_package: Backend implementation lives in opifex.
        method_names: Sampler-method identifiers handled by this backend.
        notes: Free-text rationale shown in capability reports.
        init_state: L-BFGS starting position, shape ``(d,)``.
        target_log_prob: Optional stored log-density callable. When set, the
            ``predict_distribution`` / ``posterior_predictive`` hooks re-fit
            the backend from it; ``fit`` also falls back to it when no
            ``target_log_prob`` is threaded through the call.
        num_samples: Number of draws to return from the selected
            Pathfinder Gaussian.
        num_elbo_samples: Number of Gaussian draws used to estimate
            each iteration's ELBO during the path-evaluation phase.
        maxiter: Maximum L-BFGS iterations.
        maxcor: L-BFGS memory size (history length).
        ftol: L-BFGS function-tolerance stopping criterion.
        gtol: L-BFGS gradient-norm stopping criterion.
        maxls: Line-search iteration cap per L-BFGS step.
    """

    name: str = "pathfinder"
    source_package: str = "opifex"
    method_names: tuple[str, ...] = ("pathfinder",)
    notes: str = (
        "Pathfinder (Zhang+ 2022, arXiv:2108.03782) — quasi-Newton "
        "variational inference along an L-BFGS trajectory. Algorithm "
        "ported from blackjax/vi/pathfinder.py."
    )
    init_state: jax.Array = dataclasses.field(default_factory=lambda: jnp.zeros(1))
    target_log_prob: Callable[[jax.Array], jax.Array] | None = None
    num_samples: int = 200
    num_elbo_samples: int = 64
    maxiter: int = 30
    maxcor: int = 6
    ftol: float = 1e-5
    gtol: float = 1e-8
    maxls: int = 1000

    def fit(
        self,
        target_log_prob: Callable[[jax.Array], jax.Array] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> BackendResult:
        """Run Pathfinder and return ``num_samples`` draws from the best Gaussian.

        ``target_log_prob`` overrides the stored :attr:`target_log_prob` when
        supplied; if both are ``None`` a :class:`ValueError` is raised (mirroring
        the BlackJAX backend's fit contract).

        ``sampler_state`` is the array of drawn samples with shape
        ``(num_samples, d)``; diagnostics carries no MCMC mixing
        statistics for VI.
        """
        log_density_fn = target_log_prob if target_log_prob is not None else self.target_log_prob
        if log_density_fn is None:
            raise ValueError(
                f"{self.name!r} backend.fit requires a 'target_log_prob': pass one "
                "to fit(...) or construct the backend with target_log_prob=<callable>."
            )
        approx_key = rngs.sampler() if "sampler" in rngs else rngs.default()
        sample_key = rngs.sampler() if "sampler" in rngs else rngs.default()
        state = pathfinder_approximate(
            rng_key=approx_key,
            log_density_fn=log_density_fn,
            initial_position=self.init_state,
            num_samples=self.num_elbo_samples,
            maxiter=self.maxiter,
            maxcor=self.maxcor,
            ftol=self.ftol,
            gtol=self.gtol,
            maxls=self.maxls,
        )
        samples, _ = pathfinder_sample(
            rng_key=sample_key, state=state, num_samples=self.num_samples
        )
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
        """Re-fit Pathfinder and map the posterior draws to a predictive at ``x``.

        Model-aware when ``predict_fn`` is supplied (the forward model is
        marginalised over the Pathfinder draws); otherwise the lightweight
        parameter-moment stand-in. Requires a stored :attr:`target_log_prob`.
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
        """Re-fit Pathfinder and return the posterior predictive at ``x``.

        Same draw marginalisation as :meth:`predict_distribution` with a
        ``posterior_predictive`` method tag.
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


__all__ = ["PathfinderBackend"]

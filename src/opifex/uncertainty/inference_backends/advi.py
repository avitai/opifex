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

The ``predict_distribution`` and ``posterior_predictive`` hooks remain
deferred — they require a model-aware adapter that maps
parameter-space samples to predictive distributions. Concretizing
``fit`` unblocks downstream use of the algorithm directly via
:func:`opifex.uncertainty.inference_backends._advi_algorithm.approximate`.
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
    num_samples: int = 512
    num_iterations: int = 400
    num_mc_samples: int = 8
    learning_rate: float = 0.05
    stl_estimator: bool = True

    def fit(
        self,
        target_log_prob: Callable[[jax.Array], jax.Array],
        *,
        rngs: nnx.Rngs,
    ) -> BackendResult:
        """Run mean-field ADVI and return ``num_samples`` draws from the fit.

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

        approx_key = rngs.sampler() if "sampler" in rngs else rngs.default()
        sample_key = rngs.sampler() if "sampler" in rngs else rngs.default()
        state = approximate(
            rng_key=approx_key,
            log_density_fn=target_log_prob,
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

    def predict_distribution(self, x: jax.Array, *, rngs: nnx.Rngs) -> PredictiveDistribution:
        """Deferred until a model-aware adapter maps samples → predictions."""
        del x, rngs
        raise NotImplementedError(
            f"{self.name!r} backend hook 'predict_distribution' is not yet wired. "
            f"ADVI samples are parameter-space draws; predictive distributions "
            f"require a model-aware adapter that maps parameters → predictions."
        )

    def posterior_predictive(self, rngs: nnx.Rngs, x: jax.Array) -> PredictiveDistribution:
        """Deferred until a model-aware adapter maps samples → predictions."""
        del rngs, x
        raise NotImplementedError(
            f"{self.name!r} backend hook 'posterior_predictive' is not yet wired. "
            f"ADVI samples are parameter-space draws; predictive distributions "
            f"require a model-aware adapter that maps parameters → predictions."
        )


__all__ = ["ADVIBackend"]

r"""Stein Variational Gradient Descent (SVGD) peer-sampler backend.

SVGD evolves a finite set of particles according to a kernelised
Stein-gradient flow that minimises the KL divergence to the target
posterior. It produces deterministic non-IID samples that approximate
the posterior in expectation.

Canonical reference:

* Liu, Q. & Wang, D. 2016 — *Stein Variational Gradient Descent: A
  General Purpose Bayesian Inference Algorithm*, NeurIPS 29.

The algorithm itself is vendored in
:mod:`opifex.uncertainty.inference_backends._svgd_algorithm` as a
line-by-line port of ``../blackjax/blackjax/vi/svgd.py``. This module
wires the algorithm into the :class:`InferenceBackendProtocol` surface.

The ``predict_distribution`` and ``posterior_predictive`` hooks re-fit the
backend from the stored ``target_log_prob`` and route the particle cloud
through
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

from opifex.uncertainty.inference_backends._peer_predictive import (
    peer_predictive_from_refit,
)
from opifex.uncertainty.inference_backends._svgd_algorithm import svgd_fit
from opifex.uncertainty.inference_backends.base import (
    BackendDiagnostics,
    BackendResult,
)
from opifex.uncertainty.types import (
    PredictiveDistribution,  # noqa: TC001
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SVGDBackend:
    """Stein Variational Gradient Descent backend (Liu+Wang 2016).

    Attributes:
        name: Backend identifier used by the router.
        source_package: Backend implementation lives in opifex.
        method_names: Sampler-method identifiers handled by this backend.
        notes: Free-text rationale shown in capability reports.
        init_state: Reference state used to size the particle cloud
            (shape ``(d,)``). The constructor initialises the particle
            cloud as ``init_state`` plus zero-mean Gaussian noise of
            scale :attr:`init_scale`.
        target_log_prob: Optional stored log-density callable. When set, the
            ``predict_distribution`` / ``posterior_predictive`` hooks re-fit
            the backend from it; ``fit`` also falls back to it when no
            ``target_log_prob`` is threaded through the call.
        num_particles: Number of particles ``n`` to evolve.
        num_iterations: Number of Adam steps to run.
        learning_rate: Adam learning rate.
        length_scale: Optional fixed RBF bandwidth. ``None`` uses the
            median heuristic (recomputed each step).
        init_scale: Standard deviation of the Gaussian particle
            initialisation around ``init_state``.
    """

    name: str = "svgd"
    source_package: str = "opifex"
    method_names: tuple[str, ...] = ("svgd",)
    notes: str = (
        "SVGD (Liu+Wang 2016) — kernelised Stein-gradient particle flow "
        "minimising KL to the target posterior. Algorithm ported from "
        "blackjax/vi/svgd.py."
    )
    init_state: jax.Array = dataclasses.field(default_factory=lambda: jnp.zeros(1))
    target_log_prob: Callable[[jax.Array], jax.Array] | None = None
    num_particles: int = 16
    num_iterations: int = 200
    learning_rate: float = 0.1
    length_scale: float | None = None
    init_scale: float = 1.0

    def fit(
        self,
        target_log_prob: Callable[[jax.Array], jax.Array] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> BackendResult:
        """Run SVGD for ``num_iterations`` Adam steps over ``num_particles`` particles.

        ``target_log_prob`` overrides the stored :attr:`target_log_prob` when
        supplied; if both are ``None`` a :class:`ValueError` is raised (mirroring
        the BlackJAX backend's fit contract).

        Initial particles are drawn from ``N(init_state, init_scale²·I)``.
        Returns the final particle cloud as the opaque ``sampler_state``.
        """
        log_density_fn = target_log_prob if target_log_prob is not None else self.target_log_prob
        if log_density_fn is None:
            raise ValueError(
                f"{self.name!r} backend.fit requires a 'target_log_prob': pass one "
                "to fit(...) or construct the backend with target_log_prob=<callable>."
            )
        # The blackjax peer accepts the "sampler" stream by convention; fall
        # back to "default" if no sampler stream exists.
        key = rngs.sampler() if "sampler" in rngs else rngs.default()
        perturbation_shape = (self.num_particles, *self.init_state.shape)
        perturbation = jax.random.normal(key, perturbation_shape) * self.init_scale
        initial_particles = self.init_state + perturbation
        final_particles = svgd_fit(
            initial_particles=initial_particles,
            target_log_prob_fn=log_density_fn,
            num_iterations=self.num_iterations,
            learning_rate=self.learning_rate,
            length_scale=self.length_scale,
        )
        return BackendResult(
            sampler_state=final_particles,
            diagnostics=BackendDiagnostics(),
        )

    def predict_distribution(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs,
        predict_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    ) -> PredictiveDistribution:
        """Re-fit SVGD and map the particle cloud to a predictive at ``x``.

        Model-aware when ``predict_fn`` is supplied (the forward model is
        marginalised over the Stein particles); otherwise the lightweight
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
                ("num_particles", self.num_particles),
            ),
        )

    def posterior_predictive(
        self,
        rngs: nnx.Rngs,
        x: jax.Array,
        predict_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    ) -> PredictiveDistribution:
        """Re-fit SVGD and return the posterior predictive at ``x``.

        Same particle marginalisation as :meth:`predict_distribution` with a
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
                ("num_particles", self.num_particles),
            ),
        )


__all__ = ["SVGDBackend"]

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

The ``predict_distribution`` and ``posterior_predictive`` hooks remain
deferred — they require a model-aware adapter that maps
parameter-space samples to predictive distributions. Concretizing
``fit`` unblocks downstream use of the algorithm directly via
:func:`opifex.uncertainty.inference_backends._svgd_algorithm.svgd_fit`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable  # noqa: TC003 — kept eager per opifex convention

import jax
import jax.numpy as jnp
from flax import nnx  # noqa: TC002

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
    num_particles: int = 16
    num_iterations: int = 200
    learning_rate: float = 0.1
    length_scale: float | None = None
    init_scale: float = 1.0

    def fit(
        self,
        target_log_prob: Callable[[jax.Array], jax.Array],
        *,
        rngs: nnx.Rngs,
    ) -> BackendResult:
        """Run SVGD for ``num_iterations`` Adam steps over ``num_particles`` particles.

        Initial particles are drawn from ``N(init_state, init_scale²·I)``.
        Returns the final particle cloud as the opaque ``sampler_state``.
        """
        # The blackjax peer accepts the "sampler" stream by convention; fall
        # back to "default" if no sampler stream exists.
        key = rngs.sampler() if "sampler" in rngs else rngs.default()
        perturbation_shape = (self.num_particles, *self.init_state.shape)
        perturbation = jax.random.normal(key, perturbation_shape) * self.init_scale
        initial_particles = self.init_state + perturbation
        final_particles = svgd_fit(
            initial_particles=initial_particles,
            target_log_prob_fn=target_log_prob,
            num_iterations=self.num_iterations,
            learning_rate=self.learning_rate,
            length_scale=self.length_scale,
        )
        return BackendResult(
            sampler_state=final_particles,
            diagnostics=BackendDiagnostics(),
        )

    def predict_distribution(self, x: jax.Array, *, rngs: nnx.Rngs) -> PredictiveDistribution:
        """Deferred until a model-aware adapter maps particles → predictions."""
        del x, rngs
        raise NotImplementedError(
            f"{self.name!r} backend hook 'predict_distribution' is not yet wired. "
            f"SVGD particles are parameter-space samples; predictive distributions "
            f"require a model-aware adapter that maps parameters → predictions."
        )

    def posterior_predictive(self, rngs: nnx.Rngs, x: jax.Array) -> PredictiveDistribution:
        """Deferred until a model-aware adapter maps particles → predictions."""
        del rngs, x
        raise NotImplementedError(
            f"{self.name!r} backend hook 'posterior_predictive' is not yet wired. "
            f"SVGD particles are parameter-space samples; predictive distributions "
            f"require a model-aware adapter that maps parameters → predictions."
        )


__all__ = ["SVGDBackend"]

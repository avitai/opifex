r"""Diagonal Laplace posterior approximation.

For a MAP estimate ``θ*`` and a Gaussian prior with precision
``prior_precision`` (scalar applied isotropically), the diagonal Laplace
posterior is

.. math::

    \\theta\\,|\\,\\mathcal{D} \\sim \\mathcal{N}\\bigl(
        \\theta^{*},\\,
        \\operatorname{diag}\\bigl(\\tau\\,\\mathbf{1} + F(\\theta^{*})\\bigr)^{-1}
    \\bigr),

where ``F(θ*)`` is the empirical-Fisher diagonal at the MAP point and
``τ`` is the scalar prior precision.

This module also houses :class:`LaplaceAdapterSpec` — the public adapter
that turns a fitted :class:`LaplaceState` (MAP estimate + diagonal
precision + model function) into a
:class:`~opifex.uncertainty.types.PredictiveDistribution` provider via
Monte-Carlo sampling from ``N(θ*, diag(1/precision))``.

Canonical reference:
* Daxberger Laplace package and bayesian-torch use this same diagonal
  precision formula. opifex computes the empirical-Fisher term via
  :func:`empirical_fisher_diagonal` and adds ``τ * 1`` for the prior.

References
----------
* Daxberger, E. et al. 2021 — *Laplace Redux — Effortless Bayesian Deep
  Learning*, arXiv:2106.14806.
* MacKay, D. J. C. 1992 — *A practical Bayesian framework for
  backpropagation networks*, Neural Computation 4(3).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable  # noqa: TC003 — kept eager for consistency
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx, struct

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.curvature.fisher import empirical_fisher_diagonal
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import MetadataItems, PredictiveDistribution


_LAPLACE_STREAMS = ("params", "sample", "default")


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagonalLaplacePosterior:
    """Mean + diagonal precision of a Laplace posterior.

    Attributes:
        mean: MAP estimate ``θ*``.
        precision_diagonal: Per-parameter precision ``τ + F_ii``.
    """

    mean: jax.Array
    precision_diagonal: jax.Array


def diagonal_laplace_posterior(
    *,
    per_sample_loss: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    map_estimate: jax.Array,
    inputs: jax.Array,
    targets: jax.Array,
    prior_precision: float,
) -> DiagonalLaplacePosterior:
    """Build a diagonal-Fisher Laplace posterior at a MAP point.

    Args:
        per_sample_loss: Maps ``(parameters, input, target) -> scalar``.
        map_estimate: MAP estimate ``θ*``.
        inputs: Batched inputs.
        targets: Batched targets.
        prior_precision: Scalar prior precision ``τ`` (must be ``> 0``).

    Returns:
        :class:`DiagonalLaplacePosterior` with ``precision_diagonal`` of
        the same shape as ``map_estimate``.

    Raises:
        ValueError: If ``prior_precision`` is not strictly positive.
    """
    if prior_precision <= 0.0:
        raise ValueError(f"prior_precision must be positive; got {prior_precision!r}")
    fisher_diagonal = empirical_fisher_diagonal(per_sample_loss, map_estimate, inputs, targets)
    precision_diagonal = prior_precision + fisher_diagonal
    return DiagonalLaplacePosterior(mean=map_estimate, precision_diagonal=precision_diagonal)


class _LaplaceModelProtocol(Protocol):
    """Callable signature required by LaplaceState members.

    ``parameters`` is the parameter vector sampled from the diagonal
    Laplace posterior; ``x`` is the input batch.
    """

    def __call__(self, parameters: jax.Array, x: jax.Array) -> jax.Array:
        """Evaluate the model on inputs ``x`` for the given parameter vector."""
        ...


@struct.dataclass(slots=True, kw_only=True)
class LaplaceState:
    """Fitted-state pytree for diagonal-Laplace posterior inference.

    Bundles the model function with a pre-computed
    :class:`DiagonalLaplacePosterior` (MAP estimate + per-parameter
    precision) so that ``LaplaceAdapterSpec.wrap`` can build a wrapper
    that samples parameters and produces a Monte-Carlo
    :class:`PredictiveDistribution`.

    Build the posterior with :func:`diagonal_laplace_posterior` using
    your per-sample loss + dataset, then pass the result here.
    """

    model_fn: _LaplaceModelProtocol = struct.field(pytree_node=False)
    posterior: DiagonalLaplacePosterior = struct.field()
    num_samples: int = struct.field(pytree_node=False, default=32)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Public hook; never called from ``__post_init__``/``tree_unflatten``."""
        if self.num_samples <= 1:
            raise ValueError(
                f"LaplaceState.num_samples must be > 1 to yield a non-trivial "
                f"variance estimate; got {self.num_samples!r}."
            )
        if not jnp.all(self.posterior.precision_diagonal > 0.0):
            raise ValueError(
                "LaplaceState.posterior.precision_diagonal must be entry-wise "
                "positive; got a non-positive entry."
            )


class _WrappedLaplaceModel:
    """Bookkeeping wrapper around a fitted diagonal-Laplace posterior.

    Draws parameter samples from ``N(θ*, diag(1/precision))`` and
    evaluates the model function on each draw, then summarises the
    Monte-Carlo predictive distribution.
    """

    def __init__(self, state: LaplaceState, capability: UQCapability) -> None:
        """Store the fitted Laplace state and its declared UQ capability."""
        self._state = state
        self._capability = capability

    def predict_distribution(self, x: jax.Array, *, rngs: nnx.Rngs) -> PredictiveDistribution:
        """Sample parameters from the diagonal Laplace posterior and predict."""
        key = extract_rng_key(
            rngs,
            streams=_LAPLACE_STREAMS,
            context="LaplaceAdapterSpec.predict_distribution",
        )
        mean = self._state.posterior.mean
        standard_deviation = 1.0 / jnp.sqrt(self._state.posterior.precision_diagonal)
        noise = jax.random.normal(key, (self._state.num_samples, *mean.shape))
        parameter_samples = mean + noise * standard_deviation

        def _predict(parameters: jax.Array) -> jax.Array:
            """Evaluate the model at one sampled parameter vector."""
            return self._state.model_fn(parameters, x)

        samples = jax.vmap(_predict)(parameter_samples)
        sample_mean = jnp.mean(samples, axis=0)
        variance = jnp.var(samples, axis=0)
        return PredictiveDistribution(
            mean=sample_mean,
            samples=samples,
            variance=variance,
            epistemic=variance,
            total_uncertainty=variance,
            metadata=compose_method_metadata(
                method=self._capability.default_strategy.value,
                source_package=self._capability.source_package,
                extra=(("num_samples", int(self._state.num_samples)),),
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LaplaceAdapterSpec:
    """Concrete diagonal-Laplace posterior adapter.

    Wraps a fitted :class:`LaplaceState` (MAP estimate, per-parameter
    precision, model function) into a Monte-Carlo
    :class:`PredictiveDistribution` provider. The posterior diagonal is
    built with :func:`diagonal_laplace_posterior` from the user's
    per-sample loss and dataset.

    References
    ----------
    * Daxberger, E. et al. 2021 — *Laplace Redux*, arXiv:2106.14806.
    * MacKay, D. J. C. 1992 — *A practical Bayesian framework for
      backpropagation networks*, Neural Computation 4(3).
    """

    default_strategy: DefaultStrategy = DefaultStrategy.LAPLACE
    source_package: str = "opifex.uncertainty.curvature"
    required_capabilities: tuple[str, ...] = ("native_nnx_module",)

    def wrap(self, state: LaplaceState, capability: UQCapability) -> _WrappedLaplaceModel:
        """Wrap a :class:`LaplaceState`; rejects non-``LAPLACE`` capabilities."""
        if capability.default_strategy is not DefaultStrategy.LAPLACE:
            raise ValueError(
                f"LaplaceAdapterSpec requires default_strategy="
                f"{DefaultStrategy.LAPLACE!r}; got "
                f"{capability.default_strategy!r}."
            )
        return _WrappedLaplaceModel(state=state, capability=capability)


__all__ = [
    "DiagonalLaplacePosterior",
    "LaplaceAdapterSpec",
    "LaplaceState",
    "diagonal_laplace_posterior",
]

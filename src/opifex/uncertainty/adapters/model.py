"""Concrete model-uncertainty adapters for deterministic-model UQ.

Includes deterministic, MC-dropout, and Bayesian-last-layer adapters,
plus the deferred SNGP / VBLL specs. The concrete diagonal-Laplace
adapter (``LaplaceAdapterSpec`` + ``LaplaceState``) lives in
:mod:`opifex.uncertainty.curvature.laplace` — it is co-located with the
curvature primitives it consumes.

The deterministic adapter is the bookkeeping wrapper that produces a
zero-epistemic :class:`PredictiveDistribution` from a deterministic
callable. The MC-dropout adapter exposes a stochastic model through
caller-owned ``nnx.Rngs`` — no hidden dropout seed — and returns the
sample-mean and sample-variance.

The :class:`BayesianLastLayerAdapter` applies a Bayesian treatment to
ONLY the final linear layer over frozen backbone features ``phi(x)``.
A Gaussian posterior ``N(weight_mean, weight_covariance)`` over the
last-layer weights yields a CLOSED-FORM (analytic) predictive — no
Monte-Carlo sampling. This is the GLM / linearised-Laplace pushforward
for a linear head, equivalently the neural-linear model. It is DISTINCT
from the full-network diagonal-Laplace adapter in
:mod:`opifex.uncertainty.curvature.laplace`, which Monte-Carlo-samples
parameters; this adapter evaluates the analytic last-layer predictive.

References for the Bayesian-last-layer predictive:

* Ober, S. W. & Rasmussen, C. E. 2019 — *Benchmarking the Neural Linear
  Model for Regression*, arXiv:1912.08416.
* Snoek, J. et al. 2015 — *Scalable Bayesian Optimization Using Deep
  Neural Networks*, ICML 2015.
* Daxberger, E. et al. 2021 — *Laplace Redux — Effortless Bayesian Deep
  Learning* (last-layer Laplace), arXiv:2106.14806.
* GLM / linearised pushforward output covariance ``J Sigma J^T``:
  ``../laplax/laplax/eval/pushforward.py``.

Spec dataclasses for backends that are not yet wired (SNGP, VBLL)
declare their capability metadata and raise an actionable
:class:`NotImplementedError` from ``wrap`` until the underlying
implementation lands.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, TYPE_CHECKING

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx, struct

from opifex.uncertainty.adapters._specs import _DeferredAdapterSpec
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import MetadataItems, PredictiveDistribution


if TYPE_CHECKING:
    from collections.abc import Callable


_MCDROPOUT_STREAMS = ("dropout", "sample", "default")


# ---------------------------------------------------------------------------
# Wrapped-model objects (the value ``adapter.wrap(...)`` returns)
# ---------------------------------------------------------------------------


class _WrappedDeterministicModel:
    """Bookkeeping wrapper around a deterministic callable.

    ``predict_distribution`` runs one forward pass and returns a
    :class:`PredictiveDistribution` whose ``epistemic`` field is zero by
    construction (the deterministic strategy advertises no model
    uncertainty).
    """

    def __init__(self, model: Callable[[jax.Array], jax.Array], capability: UQCapability) -> None:
        self._model = model
        self._capability = capability

    def predict_distribution(self, x: jax.Array) -> PredictiveDistribution:
        pred = self._model(x)
        zeros = jnp.zeros_like(pred)
        return PredictiveDistribution(
            mean=pred,
            samples=pred[None, ...],
            variance=zeros,
            epistemic=zeros,
            total_uncertainty=zeros,
            metadata=compose_method_metadata(
                method=self._capability.default_strategy.value,
                source_package=self._capability.source_package,
                extra=(("num_samples", 1),),
            ),
        )


class _WrappedMCDropoutModel:
    """Bookkeeping wrapper around an MC-dropout-style stochastic callable."""

    def __init__(self, state: MCDropoutState, capability: UQCapability) -> None:
        self._state = state
        self._capability = capability

    def predict_distribution(self, x: jax.Array, *, rngs: nnx.Rngs) -> PredictiveDistribution:
        # Caller-owned RNG: extract a single key and split into per-sample
        # keys. No hidden default — passing nothing raises TypeError at
        # the method boundary.
        key = extract_rng_key(
            rngs, streams=_MCDROPOUT_STREAMS, context="MCDropoutAdapter.predict_distribution"
        )
        sample_keys = jax.random.split(key, self._state.num_samples)

        def _draw(_carry: None, sample_key: jax.Array) -> tuple[None, jax.Array]:
            pred = self._state.model_fn(x, rngs=nnx.Rngs(dropout=sample_key))
            return None, pred

        _, samples = jax.lax.scan(_draw, None, sample_keys)
        mean = jnp.mean(samples, axis=0)
        variance = jnp.var(samples, axis=0)
        return PredictiveDistribution(
            mean=mean,
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


class _FeatureFnProtocol(Protocol):
    """Frozen deterministic backbone mapping ``x -> phi(x)``.

    ``phi`` has shape ``(batch, n_features)``; the last-layer weights map
    those features to ``(batch, n_outputs)``.
    """

    def __call__(self, x: jax.Array) -> jax.Array: ...


@struct.dataclass(slots=True, kw_only=True)
class BayesianLastLayerState:
    """Fitted-state pytree for the Bayesian-last-layer predictive.

    Bundles a frozen feature backbone with a Gaussian posterior over the
    final linear layer's input weights. The posterior is ``N(weight_mean,
    weight_covariance)`` where ``weight_covariance`` is a shared SPD
    covariance over the ``n_features`` input weights.

    Attributes:
        feature_fn: Frozen deterministic backbone ``x -> phi(x)`` with
            ``phi`` of shape ``(batch, n_features)``. Static (not a
            pytree leaf).
        weight_mean: Posterior mean of the last-layer weights with shape
            ``(n_features, n_outputs)`` (a pytree leaf).
        weight_covariance: Shared SPD posterior covariance over the
            input weights with shape ``(n_features, n_features)`` (a
            pytree leaf).
        observation_noise_variance: Aleatoric noise variance ``sigma^2``
            (``>= 0``), broadcast over outputs. Static aux_data so it can
            serve as a JIT-cache key.
        metadata: Immutable, hashable static aux_data.
    """

    feature_fn: _FeatureFnProtocol = struct.field(pytree_node=False)
    weight_mean: jax.Array = struct.field()
    weight_covariance: jax.Array = struct.field()
    observation_noise_variance: float = struct.field(pytree_node=False, default=0.0)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Public hook; never called from ``__post_init__``/``tree_unflatten``."""
        covariance_shape = self.weight_covariance.shape
        if self.weight_covariance.ndim != 2 or covariance_shape[0] != covariance_shape[1]:
            raise ValueError(
                "BayesianLastLayerState.weight_covariance must be a square "
                f"(n_features, n_features) matrix; got shape {covariance_shape}."
            )
        n_features = self.weight_mean.shape[0]
        if covariance_shape[0] != n_features:
            raise ValueError(
                "BayesianLastLayerState.weight_covariance dimension "
                f"{covariance_shape[0]} must equal weight_mean.shape[0]="
                f"{n_features}."
            )
        if self.observation_noise_variance < 0.0:
            raise ValueError(
                "BayesianLastLayerState.observation_noise_variance must be "
                f">= 0; got {self.observation_noise_variance!r}."
            )


class _WrappedBayesianLastLayerModel:
    """Bookkeeping wrapper around a fitted Bayesian-last-layer posterior.

    Evaluates the CLOSED-FORM (analytic) predictive of the GLM /
    linearised-Laplace pushforward for a linear head over frozen
    backbone features.
    """

    def __init__(self, state: BayesianLastLayerState, capability: UQCapability) -> None:
        self._state = state
        self._capability = capability

    def predict_distribution(self, x: jax.Array) -> PredictiveDistribution:
        """Return the analytic last-layer predictive distribution.

        For ``phi = feature_fn(x)`` (shape ``(batch, n_features)``):

        * ``mean = phi @ weight_mean`` — shape ``(batch, n_outputs)``.
        * epistemic: per-sample scalar ``q[b] = phi[b] @ Sigma @ phi[b]``
          (the GLM ``J Sigma J^T`` diagonal for a linear head, where the
          Jacobian wrt the input weights is ``phi`` itself) broadcast
          over outputs.
        * aleatoric: ``observation_noise_variance`` broadcast to
          ``(batch, n_outputs)``.
        * ``total_uncertainty = epistemic + aleatoric`` (also ``variance``).
        * ``samples = mean[None, ...]`` — one representative draw.
        """
        phi = self._state.feature_fn(x)
        mean = phi @ self._state.weight_mean
        ones = jnp.ones((1, mean.shape[-1]))
        quadratic = jnp.einsum("bi,ij,bj->b", phi, self._state.weight_covariance, phi)
        epistemic = quadratic[:, None] * ones
        aleatoric = self._state.observation_noise_variance * jnp.ones_like(mean)
        total_uncertainty = epistemic + aleatoric
        return PredictiveDistribution(
            mean=mean,
            samples=mean[None, ...],
            variance=total_uncertainty,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total_uncertainty=total_uncertainty,
            metadata=compose_method_metadata(
                method=self._capability.default_strategy.value,
                source_package=self._capability.source_package,
                extra=(("num_samples", 1),),
            ),
        )


# ---------------------------------------------------------------------------
# Adapters (the objects with ``.wrap(model, capability) -> Wrapped*``)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ModelUncertaintyAdapter:
    """Deterministic-strategy adapter for an arbitrary callable.

    Wrapping is the explicit way to advertise "this surface advertises
    no epistemic uncertainty"; the alternative is a real adapter (deep
    ensemble, MC-dropout, Laplace, SNGP, etc.) — see
    :class:`DeepEnsembleAdapter` and :class:`MCDropoutAdapter`.
    """

    def wrap(
        self, model: Callable[[jax.Array], jax.Array], capability: UQCapability
    ) -> _WrappedDeterministicModel:
        """Wrap a deterministic callable; rejects any non-deterministic capability."""
        if capability.default_strategy is not DefaultStrategy.DETERMINISTIC:
            raise ValueError(
                f"ModelUncertaintyAdapter requires default_strategy="
                f"{DefaultStrategy.DETERMINISTIC!r}; got "
                f"{capability.default_strategy!r}. Use a real adapter "
                f"(DeepEnsembleAdapter / MCDropoutAdapter / LaplaceAdapterSpec / "
                f"BayesianLastLayerAdapter / SNGPAdapterSpec / VBLLAdapterSpec) "
                f"to advertise epistemic uncertainty."
            )
        return _WrappedDeterministicModel(model=model, capability=capability)


class _MCDropoutModelProtocol(Protocol):
    """Callable signature required by MCDropoutAdapter members."""

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array: ...


@struct.dataclass(slots=True, kw_only=True)
class MCDropoutState:
    """Fitted-state pytree for MC-dropout inference.

    ``model_fn`` is a stochastic callable accepting caller-owned
    ``rngs`` and the input batch; ``num_samples`` is the MC sample
    budget per ``predict_distribution`` call.
    """

    model_fn: _MCDropoutModelProtocol = struct.field(pytree_node=False)
    num_samples: int = struct.field(pytree_node=False, default=10)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Public hook; never called from ``__post_init__``/``tree_unflatten``."""
        if self.num_samples <= 1:
            raise ValueError(
                f"MCDropoutState.num_samples must be > 1 to yield a non-trivial "
                f"variance estimate; got {self.num_samples!r}."
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MCDropoutAdapter:
    """MC-dropout adapter; caller-owned RNG at the predict-time boundary."""

    def wrap(self, model: MCDropoutState, capability: UQCapability) -> _WrappedMCDropoutModel:
        """Wrap an :class:`MCDropoutState`; rejects non-``MC_DROPOUT`` capabilities."""
        if capability.default_strategy is not DefaultStrategy.MC_DROPOUT:
            raise ValueError(
                f"MCDropoutAdapter requires default_strategy="
                f"{DefaultStrategy.MC_DROPOUT!r}; got "
                f"{capability.default_strategy!r}."
            )
        return _WrappedMCDropoutModel(state=model, capability=capability)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BayesianLastLayerAdapter:
    """Bayesian-last-layer adapter — analytic last-layer predictive.

    Applies a Bayesian treatment to ONLY the final linear layer over
    frozen backbone features ``phi(x)``. A Gaussian posterior
    ``N(weight_mean, weight_covariance)`` over the last-layer weights
    yields a CLOSED-FORM (analytic) predictive — no Monte-Carlo
    sampling. This is the GLM / linearised-Laplace pushforward for a
    linear head, equivalently the neural-linear model.

    References
    ----------
    * Ober, S. W. & Rasmussen, C. E. 2019 — *Benchmarking the Neural
      Linear Model for Regression*, arXiv:1912.08416.
    * Snoek, J. et al. 2015 — *Scalable Bayesian Optimization Using Deep
      Neural Networks*, ICML 2015.
    * Daxberger, E. et al. 2021 — *Laplace Redux* (last-layer Laplace),
      arXiv:2106.14806.
    """

    def wrap(
        self, model: BayesianLastLayerState, capability: UQCapability
    ) -> _WrappedBayesianLastLayerModel:
        """Wrap a :class:`BayesianLastLayerState`; rejects wrong capabilities."""
        if capability.default_strategy is not DefaultStrategy.BAYESIAN_LAST_LAYER:
            raise ValueError(
                f"BayesianLastLayerAdapter requires default_strategy="
                f"{DefaultStrategy.BAYESIAN_LAST_LAYER!r}; got "
                f"{capability.default_strategy!r}."
            )
        return _WrappedBayesianLastLayerModel(state=model, capability=capability)


# ---------------------------------------------------------------------------
# Spec stubs for deferred backends
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SNGPAdapterSpec(_DeferredAdapterSpec):
    """Spectral-Normalized Neural Gaussian Process last layer."""

    default_strategy: DefaultStrategy = DefaultStrategy.SNGP
    required_capabilities: tuple[str, ...] = ("native_nnx_module",)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class VBLLAdapterSpec(_DeferredAdapterSpec):
    """Variational Bayesian Last Layer."""

    default_strategy: DefaultStrategy = DefaultStrategy.VBLL
    required_capabilities: tuple[str, ...] = ("native_nnx_module",)


__all__ = [
    "BayesianLastLayerAdapter",
    "BayesianLastLayerState",
    "MCDropoutAdapter",
    "MCDropoutState",
    "ModelUncertaintyAdapter",
    "SNGPAdapterSpec",
    "VBLLAdapterSpec",
]

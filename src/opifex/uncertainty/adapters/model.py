"""Concrete model-uncertainty adapters for deterministic-model UQ.

Includes deterministic, MC-dropout, Bayesian-last-layer,
Variational-Bayesian-last-layer (VBLL), and DUE (Deterministic Uncertainty
Estimation) adapters, plus the deferred SNGP spec. The concrete
diagonal-Laplace adapter (``LaplaceAdapterSpec`` + ``LaplaceState``) lives in
:mod:`opifex.uncertainty.curvature.laplace` — it is co-located with the
curvature primitives it consumes.

The deterministic adapter is the bookkeeping wrapper that produces a
zero-epistemic :class:`PredictiveDistribution` from a deterministic
callable. The MC-dropout adapter exposes a stochastic model through
caller-owned ``nnx.Rngs`` — no hidden dropout seed — and returns the
sample-mean and sample-variance.

The :class:`BayesianLastLayerAdapter` and :class:`VBLLAdapter` both
apply a Gaussian-posterior treatment to ONLY the final linear layer
over frozen backbone features ``phi(x)`` and share the same
closed-form Gaussian-linear-head predictive assembly
(:func:`_gaussian_linear_head_predictive`). They differ only in how the
epistemic term is parameterised:

* :class:`BayesianLastLayerAdapter` carries a full SPD covariance
  ``Sigma`` and computes ``phi Sigma phi^T`` directly — the GLM /
  linearised-Laplace pushforward for a linear head (the neural-linear
  model). It is DISTINCT from the full-network diagonal-Laplace adapter
  in :mod:`opifex.uncertainty.curvature.laplace`, which
  Monte-Carlo-samples parameters; this adapter evaluates the analytic
  last-layer predictive.
* :class:`VBLLAdapter` carries the lower-triangular Cholesky factor
  ``L`` of ``Sigma = L L^T`` and computes the per-sample epistemic
  scalar as ``sum((phi @ L)**2)`` — the numerically stabler L-form used
  by the JAX reference's ``DenseNormal.covariance_weighted_inner_prod``.

References for the Bayesian-last-layer predictive:

* Ober, S. W. & Rasmussen, C. E. 2019 — *Benchmarking the Neural Linear
  Model for Regression*, arXiv:1912.08416.
* Snoek, J. et al. 2015 — *Scalable Bayesian Optimization Using Deep
  Neural Networks*, ICML 2015.
* Daxberger, E. et al. 2021 — *Laplace Redux — Effortless Bayesian Deep
  Learning* (last-layer Laplace), arXiv:2106.14806.
* GLM / linearised pushforward output covariance ``J Sigma J^T``:
  ``../laplax/laplax/eval/pushforward.py``.

References for the VBLL predictive:

* Harrison, J., Willes, J. & Snoek, J. 2024 — *Variational Bayesian Last
  Layers*, arXiv:2404.11599.
* JAX reference (regression only): ``../vbll/vbll/jax/layers/regression.py``
  and ``../vbll/vbll/jax/utils/distributions.py`` (``DenseNormal``).

The :class:`DUEAdapter` wraps an ALREADY-FITTED deep-kernel feature
extractor + inducing-point sparse variational GP (SVGP). DUE (van Amersfoort
et al. 2021) feeds a spectral-normalized / bi-Lipschitz feature map ``f(x)``
into an SVGP and reads off the GP predictive mean + variance in a single
forward pass. This adapter REUSES opifex's
:func:`opifex.uncertainty.gp.predict_svgp` (the Titsias-collapsed SVGP
predictive, grounded in GPJax's ``VariationalGaussian.predict``) over the
features — it does NOT reimplement GP math. The adapter does NOT train the
feature extractor either: the spectral-normalization / bi-Lipschitz feature
training is upstream
(:mod:`opifex.neural.operators.specialized.spectral_normalization` —
``SpectralNorm`` / ``SpectralLinear``). The wrapped state carries the frozen
feature map and the fitted :class:`opifex.uncertainty.gp.SVGPState` over the
features; ``predict_distribution`` re-tags the SVGP predictive's provenance to
the DUE strategy.

References for the DUE predictive:

* van Amersfoort, J., Smith, L., Jesson, A., Key, O. & Gal, Y. 2021 —
  *On Feature Collapse and Deep Kernel Learning for Single Forward Pass
  Uncertainty* (DUE), arXiv:2102.11409.
* SVGP predictive reused as-is: :func:`opifex.uncertainty.gp.predict_svgp`
  (Titsias 2009 collapsed SVGP, GPJax-grounded).
* Spectral-normalization / bi-Lipschitz feature primitives (upstream, used
  at TRAIN time):
  :mod:`opifex.neural.operators.specialized.spectral_normalization`.

Spec dataclasses for backends that are not yet wired (SNGP) declare
their capability metadata and raise an actionable
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
from opifex.uncertainty.gp.svgp import predict_svgp, SVGPState
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


def _gaussian_linear_head_predictive(
    mean: jax.Array,
    epistemic_scalar: jax.Array,
    observation_noise_variance: float,
    *,
    capability: UQCapability,
    num_samples: int = 1,
) -> PredictiveDistribution:
    """Assemble the closed-form Gaussian-linear-head predictive.

    Single source of truth shared by :class:`_WrappedBayesianLastLayerModel`
    and :class:`_WrappedVBLLModel`. Each wrapper computes its own
    per-sample epistemic variance scalar (BLL: ``phi Sigma phi^T`` via
    ``einsum``; VBLL: ``sum((phi @ L)**2)``) and delegates the common
    ``PredictiveDistribution`` construction here, so the
    mean/epistemic/aleatoric/total/samples/metadata assembly lives in one
    place (Rule 1: DRY).

    Args:
        mean: Predictive mean ``phi @ weight_mean`` of shape
            ``(batch, n_outputs)``.
        epistemic_scalar: Per-sample epistemic *variance* scalar of shape
            ``(batch,)`` (equals ``phi Sigma phi^T``). Broadcast over the
            ``n_outputs`` axis.
        observation_noise_variance: Aleatoric noise variance ``sigma^2``
            (``>= 0``), broadcast to ``(batch, n_outputs)``.
        capability: Declared UQ capability — supplies the ``method`` +
            ``source_package`` provenance metadata.
        num_samples: Number of representative draws recorded in metadata
            (the closed-form predictive emits the mean as a single draw).

    Returns:
        A :class:`PredictiveDistribution` with ``variance ==
        total_uncertainty == epistemic + aleatoric`` and
        ``samples = mean[None, ...]``.
    """
    ones = jnp.ones((1, mean.shape[-1]))
    epistemic = epistemic_scalar[:, None] * ones
    aleatoric = observation_noise_variance * jnp.ones_like(mean)
    total_uncertainty = epistemic + aleatoric
    return PredictiveDistribution(
        mean=mean,
        samples=mean[None, ...],
        variance=total_uncertainty,
        epistemic=epistemic,
        aleatoric=aleatoric,
        total_uncertainty=total_uncertainty,
        metadata=compose_method_metadata(
            method=capability.default_strategy.value,
            source_package=capability.source_package,
            extra=(("num_samples", num_samples),),
        ),
    )


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

        The mean/epistemic/aleatoric/total assembly is delegated to the
        shared :func:`_gaussian_linear_head_predictive` helper; only the
        epistemic ``phi Sigma phi^T`` quadratic is computed here.
        """
        phi = self._state.feature_fn(x)
        mean = phi @ self._state.weight_mean
        epistemic_scalar = jnp.einsum("bi,ij,bj->b", phi, self._state.weight_covariance, phi)
        return _gaussian_linear_head_predictive(
            mean,
            epistemic_scalar,
            self._state.observation_noise_variance,
            capability=self._capability,
        )


@struct.dataclass(slots=True, kw_only=True)
class VBLLState:
    """Fitted-state pytree for the Variational-Bayesian-last-layer predictive.

    Bundles a frozen feature backbone with a Gaussian variational
    posterior ``q(W) = N(weight_mean, Sigma_W)`` over the final linear
    layer's input weights, where ``Sigma_W = L L^T`` and ``L`` is the
    lower-triangular Cholesky factor ``weight_covariance_cholesky``.
    Carrying ``L`` (rather than the dense ``Sigma``) matches the JAX
    reference's ``DenseNormal`` parameterisation
    (``../vbll/vbll/jax/utils/distributions.py:128-130``) and keeps the
    epistemic computation numerically stable.

    Attributes:
        feature_fn: Frozen deterministic backbone ``x -> phi(x)`` with
            ``phi`` of shape ``(batch, n_features)``. Static (not a
            pytree leaf).
        weight_mean: Posterior mean of the last-layer weights with shape
            ``(n_features, n_outputs)`` (a pytree leaf).
        weight_covariance_cholesky: Lower-triangular Cholesky factor
            ``L`` of the shared posterior covariance ``Sigma_W = L L^T``
            over the input weights, with shape
            ``(n_features, n_features)`` (a pytree leaf).
        observation_noise_variance: Aleatoric noise variance ``sigma^2``
            (``>= 0``), broadcast over outputs. Static aux_data so it can
            serve as a JIT-cache key.
        metadata: Immutable, hashable static aux_data.
    """

    feature_fn: _FeatureFnProtocol = struct.field(pytree_node=False)
    weight_mean: jax.Array = struct.field()
    weight_covariance_cholesky: jax.Array = struct.field()
    observation_noise_variance: float = struct.field(pytree_node=False, default=0.0)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Public hook; never called from ``__post_init__``/``tree_unflatten``."""
        cholesky = self.weight_covariance_cholesky
        cholesky_shape = cholesky.shape
        if cholesky.ndim != 2 or cholesky_shape[0] != cholesky_shape[1]:
            raise ValueError(
                "VBLLState.weight_covariance_cholesky must be a square "
                f"(n_features, n_features) matrix; got shape {cholesky_shape}."
            )
        if not bool(jnp.allclose(cholesky, jnp.tril(cholesky))):
            raise ValueError(
                "VBLLState.weight_covariance_cholesky must be lower-triangular "
                "(it is the Cholesky factor L of Sigma_W = L @ L.T)."
            )
        n_features = self.weight_mean.shape[0]
        if cholesky_shape[0] != n_features:
            raise ValueError(
                "VBLLState.weight_covariance_cholesky dimension "
                f"{cholesky_shape[0]} must equal weight_mean.shape[0]="
                f"{n_features}."
            )
        if self.observation_noise_variance < 0.0:
            raise ValueError(
                "VBLLState.observation_noise_variance must be "
                f">= 0; got {self.observation_noise_variance!r}."
            )


class _WrappedVBLLModel:
    """Bookkeeping wrapper around a fitted VBLL variational posterior.

    Evaluates the CLOSED-FORM (analytic) regression predictive of the
    Variational Bayesian Last Layer over frozen backbone features. This
    matches the JAX reference's regression predictive; classification
    (MC-softmax marginalization) is out of scope — see :class:`VBLLAdapter`.
    """

    def __init__(self, state: VBLLState, capability: UQCapability) -> None:
        self._state = state
        self._capability = capability

    def predict_distribution(self, x: jax.Array) -> PredictiveDistribution:
        """Return the analytic VBLL regression predictive distribution.

        For ``phi = feature_fn(x)`` (shape ``(batch, n_features)``) and the
        lower-triangular Cholesky factor ``L`` of ``Sigma_W = L L^T``:

        * ``mean = phi @ weight_mean`` — shape ``(batch, n_outputs)``.
        * epistemic: per-sample scalar ``q[b] = sum_k ((L^T phi[b])_k)**2``
          computed as ``Lt_phi = phi @ L``;
          ``epistemic_scalar = sum(Lt_phi**2, axis=-1)`` — the
          ``DenseNormal.covariance_weighted_inner_prod`` form
          (``../vbll/vbll/jax/utils/distributions.py:157-160``), equal to
          ``phi Sigma_W phi^T`` but numerically stabler. Broadcast over
          outputs.
        * aleatoric: ``observation_noise_variance`` broadcast to
          ``(batch, n_outputs)`` — the additive Gaussian ``noise()`` term
          of the reference predictive
          (``../vbll/vbll/jax/layers/regression.py:61-62``).
        * ``total_uncertainty = epistemic + aleatoric`` (also ``variance``).
        * ``samples = mean[None, ...]`` — one representative draw.

        The mean/epistemic/aleatoric/total assembly is delegated to the
        shared :func:`_gaussian_linear_head_predictive` helper.
        """
        phi = self._state.feature_fn(x)
        mean = phi @ self._state.weight_mean
        # L-form epistemic: covariance_weighted_inner_prod via L^T phi.
        lt_phi = phi @ self._state.weight_covariance_cholesky
        epistemic_scalar = jnp.sum(lt_phi**2, axis=-1)
        return _gaussian_linear_head_predictive(
            mean,
            epistemic_scalar,
            self._state.observation_noise_variance,
            capability=self._capability,
        )


@struct.dataclass(slots=True, kw_only=True)
class DUEState:
    """Fitted-state pytree for the DUE predictive (deep kernel + SVGP).

    Bundles a FROZEN deterministic feature extractor with the fitted
    inducing-point sparse variational GP posterior over the FEATURES. DUE
    (van Amersfoort et al. 2021, arXiv:2102.11409) feeds a spectral-normalized
    / bi-Lipschitz feature map ``f(x)`` into an SVGP and reads off the GP
    predictive mean + variance in a single forward pass.

    The feature extractor is trained UPSTREAM (the spectral-normalization /
    bi-Lipschitz training lives in
    :mod:`opifex.neural.operators.specialized.spectral_normalization`); this
    state carries the already-fitted ``feature_fn`` (static) and the
    already-fitted :class:`opifex.uncertainty.gp.SVGPState` over the features
    (a nested pytree leaf carrier, NOT static).

    Attributes:
        feature_fn: Frozen deterministic feature map ``x -> f(x)`` with
            ``f(x)`` of shape ``(batch, n_features)``. Static (not a pytree
            leaf): it carries no traced parameters at predict time.
        svgp_state: The fitted :class:`opifex.uncertainty.gp.SVGPState` whose
            posterior lives over the FEATURES ``f(x)``. A nested pytree node
            (its Cholesky factors / inducing inputs / coefficient vector ride
            ``jax.tree_util`` transforms).
        metadata: Immutable, hashable static aux_data.
    """

    feature_fn: _FeatureFnProtocol = struct.field(pytree_node=False)
    svgp_state: SVGPState = struct.field()
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


class _WrappedDUEModel:
    """Bookkeeping wrapper around a fitted DUE deep-kernel + SVGP posterior.

    Evaluates the SVGP predictive over the frozen features in a single forward
    pass and re-tags the predictive's provenance metadata to the DUE strategy.
    The GP predictive is REUSED as-is from
    :func:`opifex.uncertainty.gp.predict_svgp` (Titsias-collapsed SVGP); no GP
    math is reimplemented here.
    """

    def __init__(self, state: DUEState, capability: UQCapability) -> None:
        self._state = state
        self._capability = capability

    def predict_distribution(self, x: jax.Array) -> PredictiveDistribution:
        """Return the SVGP-over-features predictive, re-tagged to DUE.

        For ``features = feature_fn(x)`` (shape ``(batch, n_features)``):

        * ``dist = predict_svgp(state=svgp_state, x_test=features)`` — the
          Titsias-collapsed SVGP predictive mean / variance / epistemic.
        * the returned distribution is IDENTICAL to ``dist`` in every array
          field; only the static ``metadata`` tuple is re-stamped, advertising
          ``method == "due"``, the DUE ``source_package``, and the inducing
          count, via :func:`dataclasses.replace` (the typed equivalent of the
          flax-struct ``.replace`` immutable update, which copies every array
          field unchanged and re-stamps only metadata).
        """
        features = self._state.feature_fn(x)
        dist = predict_svgp(state=self._state.svgp_state, x_test=features)
        return dataclasses.replace(
            dist,
            metadata=compose_method_metadata(
                method=self._capability.default_strategy.value,
                source_package=self._capability.source_package,
                extra=(("num_inducing", int(self._state.svgp_state.x_inducing.shape[0])),),
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
                f"BayesianLastLayerAdapter / VBLLAdapter / SNGPAdapterSpec) "
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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class VBLLAdapter:
    """Variational Bayesian Last Layer adapter — analytic regression predictive.

    Applies a variational Bayesian treatment to ONLY the final linear
    layer over frozen backbone features ``phi(x)``. A Gaussian
    variational posterior ``q(W) = N(weight_mean, L L^T)`` over the
    last-layer weights yields a CLOSED-FORM (analytic) regression
    predictive — no Monte-Carlo sampling. The epistemic term uses the
    Cholesky L-form ``sum((phi @ L)**2)`` of
    ``DenseNormal.covariance_weighted_inner_prod``.

    **Scope (bounded honestly).** This is the regression closed-form
    predictive matching the JAX reference. Classification (MC-softmax
    marginalization) is OUT OF SCOPE for this adapter — the JAX reference
    (``../vbll/vbll/jax/layers/regression.py``) implements regression
    only. There is intentionally no classification path here.

    References
    ----------
    * Harrison, J., Willes, J. & Snoek, J. 2024 — *Variational Bayesian
      Last Layers*, arXiv:2404.11599.
    * ``DenseNormal.covariance_weighted_inner_prod``:
      ``../vbll/vbll/jax/utils/distributions.py:157-160``.
    * Closed-form predictive ``(W() @ x).squeeze + noise()``:
      ``../vbll/vbll/jax/layers/regression.py:61-62``.
    """

    def wrap(self, model: VBLLState, capability: UQCapability) -> _WrappedVBLLModel:
        """Wrap a :class:`VBLLState`; rejects any non-``VBLL`` capability."""
        if capability.default_strategy is not DefaultStrategy.VBLL:
            raise ValueError(
                f"VBLLAdapter requires default_strategy="
                f"{DefaultStrategy.VBLL!r}; got "
                f"{capability.default_strategy!r}."
            )
        return _WrappedVBLLModel(state=model, capability=capability)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DUEAdapter:
    """DUE adapter — deep-kernel feature extractor + SVGP, single forward pass.

    Wraps an ALREADY-FITTED frozen feature map ``f(x)`` and the fitted
    inducing-point sparse variational GP posterior over the features. A single
    forward pass yields the GP predictive mean + variance (van Amersfoort et
    al. 2021). The GP predictive is REUSED from
    :func:`opifex.uncertainty.gp.predict_svgp` (Titsias-collapsed SVGP,
    GPJax-grounded) — no GP math is reimplemented.

    **Scope (bounded honestly).** This adapter does NOT train the feature
    extractor. The spectral-normalization / bi-Lipschitz feature training that
    makes the deep kernel distance-aware is upstream
    (:mod:`opifex.neural.operators.specialized.spectral_normalization` —
    ``SpectralNorm`` / ``SpectralLinear``); the caller fits the feature map +
    SVGP and passes both through :class:`DUEState`.

    References
    ----------
    * van Amersfoort, J., Smith, L., Jesson, A., Key, O. & Gal, Y. 2021 —
      *On Feature Collapse and Deep Kernel Learning for Single Forward Pass
      Uncertainty* (DUE), arXiv:2102.11409.
    * SVGP predictive reused as-is: :func:`opifex.uncertainty.gp.predict_svgp`.
    """

    def wrap(self, model: DUEState, capability: UQCapability) -> _WrappedDUEModel:
        """Wrap a :class:`DUEState`; rejects any non-``DUE`` capability."""
        if capability.default_strategy is not DefaultStrategy.DUE:
            raise ValueError(
                f"DUEAdapter requires default_strategy="
                f"{DefaultStrategy.DUE!r}; got "
                f"{capability.default_strategy!r}."
            )
        return _WrappedDUEModel(state=model, capability=capability)


# ---------------------------------------------------------------------------
# Spec stubs for deferred backends
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SNGPAdapterSpec(_DeferredAdapterSpec):
    """Spectral-Normalized Neural Gaussian Process last layer."""

    default_strategy: DefaultStrategy = DefaultStrategy.SNGP
    required_capabilities: tuple[str, ...] = ("native_nnx_module",)


__all__ = [
    "BayesianLastLayerAdapter",
    "BayesianLastLayerState",
    "DUEAdapter",
    "DUEState",
    "MCDropoutAdapter",
    "MCDropoutState",
    "ModelUncertaintyAdapter",
    "SNGPAdapterSpec",
    "VBLLAdapter",
    "VBLLState",
]

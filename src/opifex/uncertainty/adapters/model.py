"""Concrete model-uncertainty adapters for deterministic-model UQ.

Includes deterministic, MC-dropout, Bayesian-last-layer,
Variational-Bayesian-last-layer (VBLL), DUE (Deterministic Uncertainty
Estimation), and SNGP (Spectral-normalized Neural Gaussian Process)
adapters. The concrete diagonal-Laplace adapter (``LaplaceAdapterSpec`` +
``LaplaceState``) lives in :mod:`opifex.uncertainty.curvature.laplace` — it is
co-located with the curvature primitives it consumes.

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

The :class:`SNGPAdapter` wraps an ALREADY-FITTED Spectral-normalized
Neural Gaussian Process last layer (Liu et al. 2020, arXiv:2006.10108). A
fixed random-Fourier-feature (RFF) map ``phi(x)`` feeds a
Laplace-approximated GP whose predictive variance is a distance-aware
uncertainty signal. The wrapped state carries the frozen ``feature_fn``,
the last-layer weights ``beta`` (``output_weights``), and the Laplace
precision matrix ``Sigma^{-1}`` (``precision_matrix``). The predictive mean
is ``phi @ beta`` and the per-sample epistemic variance is computed by the
edward2 Cholesky-solve formula ``ridge * sum((L^{-1} phi^T)**2)`` for
``L = cholesky(Sigma^{-1})`` — then delegated to the SHARED
:func:`_gaussian_linear_head_predictive` assembly (Rule 1: DRY), exactly as
the Bayesian-last-layer and VBLL adapters do. The Laplace precision is built
faithfully by :func:`fit_sngp_precision` (a direct port of edward2's
``LaplaceRandomFeatureCovariance`` initial + exact-update logic), NOT from
scratch. SNGP emits a regression mean + variance predictive (consistent with
the other model adapters); the classification mean-field-logit adjustment is
exposed honestly as :func:`sngp_mean_field_logits` (an edward2 port) but is
NOT fabricated into the regression predict path.

References for the SNGP predictive:

* Liu, J., Lin, Z., Padhy, S., Tran, D., Bedrax-Weiss, T. & Lakshminarayanan,
  B. 2020 — *Simple and Principled Uncertainty Estimation with Deterministic
  Deep Learning via Distance Awareness* (SNGP), arXiv:2006.10108.
* JAX reference (ported, not invented): ``LaplaceRandomFeatureCovariance``
  (``../edward2/edward2/jax/nn/random_feature.py``:217-405 —
  ``update_precision_matrix``:311-362 and
  ``compute_predictive_covariance``:364-405) and ``mean_field_logits``
  (``../edward2/edward2/jax/nn/utils.py``:54-101).
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, TYPE_CHECKING

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx, struct

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


_SNGP_SUPPORTED_LIKELIHOODS = ("gaussian", "binary_logistic", "poisson")
_MEAN_FIELD_SUPPORTED_LIKELIHOODS = ("logistic", "binary_logistic", "poisson")


def fit_sngp_precision(
    features: jax.Array,
    *,
    ridge_penalty: float = 1.0,
    logits: jax.Array | None = None,
    likelihood: str = "gaussian",
) -> jax.Array:
    """Build the SNGP Laplace precision matrix from training random features.

    Faithful port of edward2's ``LaplaceRandomFeatureCovariance`` initial +
    exact (momentum-free) update path
    (``../edward2/edward2/jax/nn/random_feature.py``:
    ``initial_precision_matrix``:307-309 and
    ``update_precision_matrix``:311-362). With training random features
    ``phi_tr`` of shape ``(n, D)`` the precision is built as a single
    exact pass over the data (the edward2 ``momentum is None`` branch,
    :354-356)::

        initial    = ridge_penalty * I_D
        phi_adj    = sqrt(prob_multiplier) * phi_tr
        batch_prec = phi_adj.T @ phi_adj
        precision  = initial + batch_prec

    where ``prob_multiplier`` is the Laplace-approximation weight per the
    likelihood (edward2 :342-348):

    * ``gaussian`` -> ``1``;
    * ``binary_logistic`` -> ``p * (1 - p)`` with ``p = sigmoid(logits)``;
    * ``poisson`` -> ``exp(logits)``.

    Args:
        features: Training random features ``phi_tr`` of shape ``(n, D)``.
        ridge_penalty: Ridge factor ``s`` stabilising the precision
            eigenvalues (``> 0``); also the predictive-variance scale.
        logits: Predictive logits of shape ``(n,)`` or ``(n, 1)``. Required
            for the non-Gaussian likelihoods; unused for ``gaussian``.
        likelihood: One of ``gaussian`` / ``binary_logistic`` / ``poisson``.

    Returns:
        The symmetric positive-definite Laplace precision matrix
        ``Sigma^{-1}`` of shape ``(D, D)``.

    Raises:
        ValueError: If ``likelihood`` is unsupported, or if ``logits`` is
            ``None`` for a non-Gaussian likelihood.
    """
    if likelihood not in _SNGP_SUPPORTED_LIKELIHOODS:
        raise ValueError(
            f"SNGP likelihood must be one of {_SNGP_SUPPORTED_LIKELIHOODS}; got {likelihood!r}."
        )
    hidden_features = features.shape[-1]
    initial = ridge_penalty * jnp.eye(hidden_features, dtype=features.dtype)
    # prob_multiplier — the Laplace-approximation weight (edward2 :342-348).
    if likelihood == "gaussian":
        prob_multiplier: jax.Array | float = 1.0
    else:
        if logits is None:
            raise ValueError(f"SNGP likelihood={likelihood!r} requires logits; got None.")
        logits_column = jnp.reshape(logits, (features.shape[0], -1))
        if likelihood == "binary_logistic":
            probability = jax.nn.sigmoid(logits_column)
            prob_multiplier = probability * (1.0 - probability)
        else:  # poisson
            prob_multiplier = jnp.exp(logits_column)
    features_adjusted = jnp.sqrt(prob_multiplier) * features
    batch_precision = features_adjusted.T @ features_adjusted
    return initial + batch_precision


def sngp_mean_field_logits(
    logits: jax.Array,
    variances: jax.Array,
    *,
    mean_field_factor: float,
    likelihood: str = "logistic",
) -> jax.Array:
    """Adjust classification logits by the SNGP mean-field approximation.

    Faithful port of edward2's ``mean_field_logits``
    (``../edward2/edward2/jax/nn/utils.py``:54-101). Scales the logits down
    by a per-sample factor so the softmax approximates the posterior mean of
    the Gaussian-approximated latent posterior (Liu et al. 2020). For a
    negative ``mean_field_factor`` the logits are returned unchanged (edward2
    :85-86). The scaling coefficient is (edward2 :92-95)::

        logistic / binary_logistic -> sqrt(1 + variances * mean_field_factor)
        poisson                    -> exp(-variances * mean_field_factor / 2)

    and is broadcast over the trailing logit-class axis (edward2 :97-99).

    This helper is the honest classification counterpart to the SNGP
    regression predictive; the regression predict path
    (:class:`_WrappedSNGPModel`) does NOT call it (binding rule 4).

    Args:
        logits: Predictive logits of shape ``(batch, num_classes)``.
        variances: Per-sample predictive variances of shape ``(batch,)``.
        mean_field_factor: Mean-field scale factor; ``< 0`` disables the
            adjustment.
        likelihood: One of ``logistic`` / ``binary_logistic`` / ``poisson``.

    Returns:
        Uncertainty-adjusted logits of shape ``(batch, num_classes)``.

    Raises:
        ValueError: If ``likelihood`` is unsupported.
    """
    if likelihood not in _MEAN_FIELD_SUPPORTED_LIKELIHOODS:
        raise ValueError(
            "SNGP mean-field likelihood must be one of "
            f"{_MEAN_FIELD_SUPPORTED_LIKELIHOODS}; got {likelihood!r}."
        )
    if mean_field_factor < 0:
        return logits
    if likelihood == "poisson":
        logits_scale = jnp.exp(-variances * mean_field_factor / 2.0)
    else:
        logits_scale = jnp.sqrt(1.0 + variances * mean_field_factor)
    # Broadcast the per-sample scale over the trailing logit-class axis.
    while logits_scale.ndim < logits.ndim:
        logits_scale = jnp.expand_dims(logits_scale, axis=-1)
    return logits / logits_scale


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


@struct.dataclass(slots=True, kw_only=True)
class SNGPState:
    """Fitted-state pytree for the SNGP predictive (RFF + Laplace GP head).

    Bundles a FROZEN deterministic random-Fourier-feature (RFF) map
    ``phi(x)`` with the fitted last-layer weights and the Laplace precision
    matrix of the GP head. SNGP (Liu et al. 2020, arXiv:2006.10108) replaces a
    deterministic network's final layer with a random-feature Gaussian-process
    approximation whose predictive variance is a distance-aware uncertainty
    signal.

    The RFF map is FIXED (not trained at predict time); the precision matrix is
    built once over the training features via :func:`fit_sngp_precision` (a
    faithful port of edward2's ``LaplaceRandomFeatureCovariance``). At predict
    time the mean is ``phi @ output_weights`` and the per-sample epistemic
    variance is the edward2 Cholesky-solve diagonal ``ridge * sum((L^{-1}
    phi^T)**2)`` for ``L = cholesky(precision_matrix, lower=True)``.

    Attributes:
        feature_fn: Frozen deterministic RFF map ``x -> phi(x)`` with ``phi``
            of shape ``(batch, D)``. Static (not a pytree leaf): it carries no
            traced parameters at predict time.
        output_weights: Last-layer weights ``beta`` of shape ``(D, n_outputs)``
            (a pytree leaf).
        precision_matrix: The Laplace precision matrix ``Sigma^{-1}`` of shape
            ``(D, D)`` (a pytree leaf).
        ridge_penalty: Ridge factor ``s`` (``> 0``) stabilising the precision
            and scaling the predictive variance. Static aux_data so it can
            serve as a JIT-cache key.
        observation_noise_variance: Aleatoric noise variance ``sigma^2``
            (``>= 0``), broadcast over outputs. Static aux_data.
        metadata: Immutable, hashable static aux_data.
    """

    feature_fn: _FeatureFnProtocol = struct.field(pytree_node=False)
    output_weights: jax.Array = struct.field()
    precision_matrix: jax.Array = struct.field()
    ridge_penalty: float = struct.field(pytree_node=False, default=1.0)
    observation_noise_variance: float = struct.field(pytree_node=False, default=0.0)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Public hook; never called from ``__post_init__``/``tree_unflatten``."""
        precision_shape = self.precision_matrix.shape
        if self.precision_matrix.ndim != 2 or precision_shape[0] != precision_shape[1]:
            raise ValueError(
                "SNGPState.precision_matrix must be a square (D, D) matrix; "
                f"got shape {precision_shape}."
            )
        hidden_features = self.output_weights.shape[0]
        if precision_shape[0] != hidden_features:
            raise ValueError(
                "SNGPState.precision_matrix dimension "
                f"{precision_shape[0]} must equal output_weights.shape[0]="
                f"{hidden_features}."
            )
        if self.ridge_penalty <= 0.0:
            raise ValueError(f"SNGPState.ridge_penalty must be > 0; got {self.ridge_penalty!r}.")
        if self.observation_noise_variance < 0.0:
            raise ValueError(
                "SNGPState.observation_noise_variance must be "
                f">= 0; got {self.observation_noise_variance!r}."
            )


class _WrappedSNGPModel:
    """Bookkeeping wrapper around a fitted SNGP RFF + Laplace-GP head.

    Evaluates the CLOSED-FORM SNGP regression predictive over the frozen RFF
    features. The per-sample epistemic variance is the edward2
    ``compute_predictive_covariance`` Cholesky-solve diagonal
    (``../edward2/edward2/jax/nn/random_feature.py``:393-404); the
    mean/epistemic/aleatoric/total assembly is delegated to the shared
    :func:`_gaussian_linear_head_predictive` helper (Rule 1: DRY).
    """

    def __init__(self, state: SNGPState, capability: UQCapability) -> None:
        self._state = state
        self._capability = capability

    def predict_distribution(self, x: jax.Array) -> PredictiveDistribution:
        """Return the analytic SNGP regression predictive distribution.

        For ``phi = feature_fn(x)`` (shape ``(batch, D)``):

        * ``mean = phi @ output_weights`` — shape ``(batch, n_outputs)``.
        * epistemic: per-sample scalar from the edward2 chol-solve formula
          (``compute_predictive_covariance``:393-404)::

              chol = cholesky(precision_matrix, lower=True)
              y    = solve_triangular(chol, phi.T, lower=True)
              q    = ridge_penalty * sum(y**2, axis=0)

          equal to ``ridge_penalty * diag(phi @ inv(precision) @ phi.T)`` but
          computed via the stabler triangular solve. Broadcast over outputs.
        * aleatoric: ``observation_noise_variance`` broadcast to
          ``(batch, n_outputs)``.
        * ``total_uncertainty = epistemic + aleatoric`` (also ``variance``).
        * ``samples = mean[None, ...]`` — one representative draw.

        This adapter emits the regression mean + variance form (consistent with
        the other model adapters). The classification mean-field-logit
        adjustment is available as :func:`sngp_mean_field_logits` but is NOT
        applied here (binding rule 4): there is intentionally no fabricated
        classification path.
        """
        phi = self._state.feature_fn(x)
        mean = phi @ self._state.output_weights
        # edward2 compute_predictive_covariance: chol-solve diagonal.
        chol = jax.scipy.linalg.cholesky(self._state.precision_matrix, lower=True)
        y = jax.scipy.linalg.solve_triangular(chol, phi.T, lower=True)
        epistemic_scalar = self._state.ridge_penalty * jnp.sum(y**2, axis=0)
        return _gaussian_linear_head_predictive(
            mean,
            epistemic_scalar,
            self._state.observation_noise_variance,
            capability=self._capability,
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
                f"BayesianLastLayerAdapter / VBLLAdapter / SNGPAdapter) "
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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SNGPAdapter:
    """SNGP adapter — RFF + Laplace-GP last layer, single forward pass.

    Wraps an ALREADY-FITTED Spectral-normalized Neural Gaussian Process last
    layer (Liu et al. 2020). A FIXED random-Fourier-feature map ``phi(x)``
    feeds a Laplace-approximated GP head whose predictive variance is a
    distance-aware uncertainty signal. A single forward pass yields the
    predictive mean ``phi @ output_weights`` and the per-sample epistemic
    variance via the edward2 Cholesky-solve diagonal; the
    mean/epistemic/aleatoric/total assembly is delegated to the SHARED
    :func:`_gaussian_linear_head_predictive` helper (Rule 1: DRY), exactly as
    :class:`BayesianLastLayerAdapter` and :class:`VBLLAdapter` do.

    **Scope (bounded honestly).** This adapter does NOT train the feature map
    or fit the precision matrix. The RFF map is fixed; the Laplace precision is
    built once via :func:`fit_sngp_precision` (a faithful edward2 port) and
    passed through :class:`SNGPState`. The adapter emits a regression mean +
    variance predictive (consistent with the other model adapters); the
    classification mean-field-logit adjustment is exposed honestly as
    :func:`sngp_mean_field_logits` but is NOT applied in the predict path.

    References
    ----------
    * Liu, J. et al. 2020 — *Simple and Principled Uncertainty Estimation with
      Deterministic Deep Learning via Distance Awareness* (SNGP),
      arXiv:2006.10108.
    * JAX reference (ported, not invented): ``LaplaceRandomFeatureCovariance``
      (``../edward2/edward2/jax/nn/random_feature.py``:217-405) and
      ``mean_field_logits`` (``../edward2/edward2/jax/nn/utils.py``:54-101).
    """

    def wrap(self, model: SNGPState, capability: UQCapability) -> _WrappedSNGPModel:
        """Wrap an :class:`SNGPState`; rejects any non-``SNGP`` capability."""
        if capability.default_strategy is not DefaultStrategy.SNGP:
            raise ValueError(
                f"SNGPAdapter requires default_strategy="
                f"{DefaultStrategy.SNGP!r}; got "
                f"{capability.default_strategy!r}."
            )
        return _WrappedSNGPModel(state=model, capability=capability)


__all__ = [
    "BayesianLastLayerAdapter",
    "BayesianLastLayerState",
    "DUEAdapter",
    "DUEState",
    "MCDropoutAdapter",
    "MCDropoutState",
    "ModelUncertaintyAdapter",
    "SNGPAdapter",
    "SNGPState",
    "VBLLAdapter",
    "VBLLState",
    "fit_sngp_precision",
    "sngp_mean_field_logits",
]

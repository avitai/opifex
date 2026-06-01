r"""Shared :class:`PredictiveDistribution` construction factory.

Canonical constructors that every Gaussian-process / state-space estimator
uses to assemble a :class:`opifex.uncertainty.types.PredictiveDistribution`
and to refresh its provenance metadata. Centralising construction here keeps
the field set and metadata shape in a single place (Rule 1 — DRY) so that
adding or renaming a field touches exactly one site.

Three functions:

* :func:`gaussian_process_predictive` — the canonical constructor for a
  Gaussian-process predictive carrying ``mean`` / ``variance`` plus optional
  ``epistemic`` / ``total_uncertainty`` variance decompositions and metadata.
* :func:`sample_based_predictive` — the canonical constructor for a
  Monte-Carlo / sampler predictive carrying ``samples`` plus the empirical
  ``mean`` / ``variance`` reduced over the leading sample axis (the shape the
  SBI ``predict_distribution`` paths build).
* :func:`replace_predictive_metadata` — immutable metadata refresh that keeps
  every array field untouched and only re-stamps the
  ``compose_method_metadata`` provenance tuple via :func:`dataclasses.replace`.

Both reproduce, byte-for-byte, the objects the hand-written estimator sites
build today (the markov ``predict_*`` paths and the per-likelihood response
wrappers).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable  # noqa: TC003 — kept eager per opifex convention

import jax
import jax.numpy as jnp

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import MetadataItems, PredictiveDistribution


def gaussian_process_predictive(
    mean: jax.Array,
    variance: jax.Array,
    *,
    epistemic: jax.Array | None = None,
    total_uncertainty: jax.Array | None = None,
    metadata: MetadataItems = (),
) -> PredictiveDistribution:
    """Assemble a Gaussian-process :class:`PredictiveDistribution`.

    Args:
        mean: Predictive mean, shape ``(batch, ...)``.
        variance: Marginal predictive *variance* (not std-dev) matching
            ``mean.shape``.
        epistemic: Optional epistemic-variance component. Defaults to ``None``.
        total_uncertainty: Optional total-variance component. Defaults to
            ``None``.
        metadata: Immutable, hashable provenance tuple. Defaults to ``()``.

    Returns:
        A :class:`PredictiveDistribution` carrying the supplied moments and
        metadata; all other container fields take their dataclass defaults.
    """
    return PredictiveDistribution(
        mean=mean,
        variance=variance,
        epistemic=epistemic,
        total_uncertainty=total_uncertainty,
        metadata=metadata,
    )


def sample_based_predictive(
    samples: jax.Array,
    *,
    metadata: MetadataItems = (),
) -> PredictiveDistribution:
    """Assemble a sampler :class:`PredictiveDistribution` from posterior draws.

    Reduces ``samples`` to its empirical ``mean`` and ``variance`` over the
    leading sample axis (axis ``0``) and stores the raw draws on the
    ``samples`` field — byte-for-byte the object the SBI ``predict_distribution``
    paths (NPE flow-sampling, NLE/NRE MCMC) build today. Centralising the
    moment reduction here keeps the sample-to-moment convention in one place
    (Rule 1 — DRY).

    Args:
        samples: Posterior draws, shape ``(num_samples, ...)``; axis ``0`` is
            the sample axis reduced for ``mean`` / ``variance``.
        metadata: Immutable, hashable provenance tuple. Defaults to ``()``.

    Returns:
        A :class:`PredictiveDistribution` carrying ``samples`` plus the
        empirical ``mean`` / ``variance``; all other container fields take
        their dataclass defaults.
    """
    return PredictiveDistribution(
        mean=jnp.mean(samples, axis=0),
        variance=jnp.var(samples, axis=0),
        samples=samples,
        metadata=metadata,
    )


def ensemble_predictive(
    samples: jax.Array,
    *,
    method: str,
    source_package: str = "opifex",
    extra_metadata: MetadataItems = (),
    include_zero_aleatoric: bool = False,
) -> PredictiveDistribution:
    """Aggregate a stack of member predictions into a :class:`PredictiveDistribution`.

    Members live on ``axis=0`` (shape ``(num_members, batch, ...)``). The
    predictive mean and (sample) variance are taken across that member axis;
    the variance is reported as the marginal ``variance`` *and* the
    ``epistemic`` / ``total_uncertainty`` components — the deep-ensemble
    member-aggregation contract. This is the single source of truth for every
    "stack-of-predictions → epistemic-decomposed predictive" site (the ensemble
    model adapters and the quantum-chemistry UQ surfaces), keeping the
    mean/variance reduction and the variance-decomposition layout in one place
    (Rule 1 — DRY).

    Unlike :func:`sample_based_predictive` (which leaves the variance
    decomposition unset), this constructor populates the epistemic / total
    variance fields because the spread *across members* is, by construction,
    epistemic uncertainty.

    Args:
        samples: Member predictions, shape ``(num_members, ...)``; axis ``0`` is
            the member axis reduced for ``mean`` / ``variance``.
        method: Method tag stamped into the provenance metadata under
            ``"method"``.
        source_package: ``source_package`` provenance tag. Defaults to
            ``"opifex"``.
        extra_metadata: Additional immutable, hashable metadata pairs appended
            after the standard ``method`` / ``source_package`` entries.
            Defaults to ``()``.
        include_zero_aleatoric: When set, an explicit zero ``aleatoric`` array
            is emitted so the variance-additivity invariant
            ``total_uncertainty == epistemic + aleatoric`` is satisfied with a
            materialised aleatoric term. Defaults to ``False``.

    Returns:
        A :class:`PredictiveDistribution` carrying ``mean`` / ``samples`` plus
        the across-member ``variance`` reported as both the marginal variance
        and the ``epistemic`` / ``total_uncertainty`` decomposition.
    """
    mean = jnp.mean(samples, axis=0)
    variance = jnp.var(samples, axis=0)
    return PredictiveDistribution(
        mean=mean,
        samples=samples,
        variance=variance,
        epistemic=variance,
        aleatoric=jnp.zeros_like(mean) if include_zero_aleatoric else None,
        total_uncertainty=variance,
        metadata=compose_method_metadata(
            method=method,
            source_package=source_package,
            extra=extra_metadata,
        ),
    )


def predictive_from_parameter_samples(
    parameter_samples: jax.Array,
    x: jax.Array,
    *,
    predict_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    metadata: MetadataItems = (),
) -> PredictiveDistribution:
    r"""Map parameter-space posterior draws to a :class:`PredictiveDistribution`.

    The inference backends (BlackJAX / ADVI / SVGD / Pathfinder) all return
    posterior draws in *parameter* space (shape ``(num_samples, d)``). This is
    the single canonical adapter that turns those draws into a predictive
    distribution at the inputs ``x`` (Rule 1 — DRY), used by every backend's
    ``predict_distribution`` / ``posterior_predictive`` hook.

    Two paths:

    * **Model-aware** (``predict_fn`` supplied). This is the *genuine* Bayesian
      posterior predictive: the model forward map is marginalised over the
      posterior parameter draws,
      :math:`p(y \mid x, \mathcal{D}) = \int p(y \mid x, \theta)\,
      p(\theta \mid \mathcal{D})\, d\theta`, here approximated by the Monte-Carlo
      average over the draws
      :math:`\{\theta_s\}_{s=1}^{S} \sim p(\theta \mid \mathcal{D})`
      (Gelman et al. 2013, *Bayesian Data Analysis* 3rd ed., §3.2). The forward
      model is vmapped over the leading sample axis and the per-draw predictions
      are reduced to their empirical mean / variance via
      :func:`sample_based_predictive`.
    * **Lightweight** (``predict_fn is None``). A parameter-moment stand-in for
      the case where the target log-density does not depend on a separate
      predictive model: the posterior parameter mean / variance are broadcast to
      ``x.shape``. This is *not* the true predictive — it is a placeholder that
      keeps the moment shapes consistent and reproduces
      :class:`opifex.uncertainty.inference_backends.blackjax.BlackJAXBackend`'s
      lightweight form byte-for-byte.

    Args:
        parameter_samples: Posterior parameter draws, shape ``(num_samples, d)``;
            axis ``0`` is the sample axis.
        x: Inputs at which to evaluate the predictive, shape ``(batch, ...)``.
        predict_fn: Optional forward model with signature
            ``predict_fn(params_vector, x) -> predictions``. When supplied the
            model-aware path is taken; when ``None`` the lightweight stand-in is
            returned. Defaults to ``None``.
        metadata: Immutable, hashable provenance tuple. Defaults to ``()``.

    Returns:
        A :class:`PredictiveDistribution`. The model-aware path stores the
        per-draw predictions on ``samples`` with the empirical ``mean`` /
        ``variance``; the lightweight path stores the broadcast parameter
        moments plus the broadcast draws.
    """
    if predict_fn is not None:
        predictions = jax.vmap(lambda params: predict_fn(params, x))(parameter_samples)
        return sample_based_predictive(predictions, metadata=metadata)

    mean = jnp.broadcast_to(jnp.mean(parameter_samples, axis=0), x.shape)
    variance = jnp.broadcast_to(jnp.var(parameter_samples, axis=0), x.shape)
    broadcast_samples = (
        jnp.broadcast_to(
            parameter_samples[:, None, :],
            (parameter_samples.shape[0], *x.shape),
        )
        if x.shape and parameter_samples.ndim == 2
        else parameter_samples
    )
    return PredictiveDistribution(
        mean=mean,
        variance=variance,
        samples=broadcast_samples,
        metadata=metadata,
    )


def replace_predictive_metadata(
    predictive: PredictiveDistribution,
    *,
    estimator: str,
    likelihood: str,
    link: str,
    paper: str | None = None,
    source: str = "opifex.uncertainty.markov",
) -> PredictiveDistribution:
    """Return ``predictive`` with refreshed provenance metadata.

    Keeps every array field of ``predictive`` untouched and re-stamps only the
    static metadata tuple, advertising ``estimator`` / ``likelihood`` /
    ``link`` (and an optional ``paper`` citation) for the
    :data:`DefaultStrategy.GAUSSIAN_PROCESS` method.

    Args:
        predictive: The predictive whose metadata is refreshed.
        estimator: Estimator identifier (e.g. ``"bernoulli_markov_vi_gp"``).
        likelihood: Likelihood name (e.g. ``"bernoulli"``).
        link: Response-link name (e.g. ``"logit"``).
        paper: Optional citation string; omitted from metadata when ``None``.
        source: ``source_package`` provenance tag. Defaults to the markov
            inference package.

    Returns:
        A metadata-refreshed copy of ``predictive`` via
        :func:`dataclasses.replace` (the typed equivalent of the flax-struct
        ``.replace`` immutable update).
    """
    extra: list[tuple[str, str]] = [("estimator", estimator)]
    if paper is not None:
        extra.append(("paper", paper))
    extra.append(("likelihood", likelihood))
    extra.append(("link", link))
    # ``dataclasses.replace`` is the typed equivalent of the flax-struct
    # ``predictive.replace`` immutable update (the struct method delegates to
    # it); it copies every array field unchanged and re-stamps only metadata.
    return dataclasses.replace(
        predictive,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=source,
            extra=tuple(extra),
        ),
    )


__all__ = [
    "ensemble_predictive",
    "gaussian_process_predictive",
    "predictive_from_parameter_samples",
    "replace_predictive_metadata",
    "sample_based_predictive",
]

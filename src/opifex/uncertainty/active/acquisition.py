"""Pure JAX single-point acquisition kernels for active learning.

Every kernel takes a :class:`opifex.uncertainty.types.PredictiveDistribution`
and returns a ``jax.Array`` of per-candidate utility scores. The kernels
are JAX-traceable (``jit`` / ``grad`` / ``vmap`` safe) and never call into
``flax.nnx``; that boundary is enforced by the duplicate-code gate at
``basic_trainer.py``.

Reference (read-only): ``trieste``. Each acquisition function cites the
exact trieste source line it ports from. The opifex port substitutes
``jax.scipy.stats.norm`` for ``tensorflow_probability.distributions.Normal``
and drops the trieste search-space / dataset plumbing — opifex's
:class:`PredictiveDistribution` already carries posterior moments and
samples.

Acquisition conventions:

* EI / Log-EI / PI follow the **minimisation** convention used by trieste
  (``best_value`` is the best (smallest) observed value; positive scores
  reflect improvement towards lower values).
* UCB returns ``mean + beta * std`` (the "upper" confidence bound; maximise
  to explore high-mean regions).
* LCB returns ``mean - beta * std`` (the "lower" confidence bound; minimise
  for exploration under a minimisation objective).
* BALD returns the per-candidate mutual information ``H[E[p]] - E[H[p]]``.
  For the regression-ensemble case carried by
  :class:`PredictiveDistribution`, the per-sample distributions are
  Gaussians with shared aleatoric variance, and the predictive marginal is
  approximated by a moment-matched Gaussian (an upper bound on the true
  mixture entropy by Jensen's inequality — same approximation as
  ``trieste/acquisition/function/active_learning.py:418`` for the GP-
  classification BALD).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx, struct
from jax.scipy.stats import norm as jnorm

from opifex.uncertainty.types import MetadataItems, metadata_to_dict, PredictiveDistribution


# Numerical floors used in entropy / log-prob computations.
_VARIANCE_FLOOR: float = 1e-12
_LOG_FLOOR: float = 1e-12


class AcquisitionStrategy(StrEnum):
    """Named acquisition strategies dispatched by :func:`acquire`."""

    BALD = "bald"
    EI = "ei"
    LOG_EI = "log_ei"
    UCB = "ucb"
    LCB = "lcb"
    PI = "pi"
    MAX_VARIANCE = "max_variance"


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveLearningConfig:
    """Configuration container for an active-learning round.

    GUIDE_ALIGNMENT pattern (A): plain ``@dataclass(frozen=True,
    slots=True, kw_only=True)``. Sequence fields use ``tuple[...]``
    (GUIDE_ALIGNMENT item 22a). ``__post_init__`` performs eager
    validation.
    """

    strategy: AcquisitionStrategy
    batch_size: int
    acquisition_family: str = "single_point"
    extra_streams: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {self.batch_size!r}")
        if not isinstance(self.strategy, AcquisitionStrategy):
            raise TypeError(f"strategy must be AcquisitionStrategy; got {type(self.strategy)}")


@struct.dataclass
class AcquiredBatch:
    """Result of one acquisition round.

    GUIDE_ALIGNMENT pattern (B): ``flax.struct.dataclass`` carrying
    ``jax.Array`` payloads. ``strategy`` and ``metadata`` are marked
    ``pytree_node=False`` so they participate in the JIT cache key rather
    than the leaf list. :meth:`validate` is public and is **never** called
    from the unflatten path.
    """

    indices: jax.Array
    scores: jax.Array
    strategy: str = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata tuple."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Eager-validate shapes; not called from ``__post_init__``."""
        if self.indices.ndim != 1:
            raise ValueError(
                f"AcquiredBatch.indices must be 1-D; got shape {self.indices.shape}"
            )
        if not self.strategy:
            raise ValueError("AcquiredBatch.strategy must be a non-empty string.")


# ---------------------------------------------------------------------------
# Single-point acquisition kernels
# ---------------------------------------------------------------------------


def _require_variance(pd: PredictiveDistribution) -> jax.Array:
    if pd.variance is None:
        raise ValueError("acquisition requires PredictiveDistribution.variance")
    return jnp.maximum(pd.variance, _VARIANCE_FLOOR)


def expected_improvement(
    predictive_dist: PredictiveDistribution,
    *,
    best_value: float,
) -> jax.Array:
    r"""Single-point Expected Improvement (minimisation convention).

    Ported from ``trieste/acquisition/function/function.py:226``
    (``expected_improvement.__call__``):

    .. math::
        \mathrm{EI}(x) = (\eta - \mu(x)) \Phi\!\left(\tfrac{\eta - \mu(x)}{\sigma(x)}\right)
        + \sigma(x) \, \phi\!\left(\tfrac{\eta - \mu(x)}{\sigma(x)}\right)

    Substitutions vs. trieste: ``tfp.distributions.Normal`` → ``jax.scipy.stats.norm``.
    """
    variance = _require_variance(predictive_dist)
    std = jnp.sqrt(variance)
    u = (best_value - predictive_dist.mean) / std
    return (best_value - predictive_dist.mean) * jnorm.cdf(u) + std * jnorm.pdf(u)


def _log_ei_helper(u: jax.Array) -> jax.Array:
    r"""Numerically stable ``log(phi(u) + u * Phi(u))``.

    Ported from ``trieste/acquisition/function/function.py:281``
    (``log_ei_helper``). For the safe regime ``u >= -1`` the direct
    expression ``log(phi(u) + u * Phi(u))`` is well-conditioned. For
    ``u << -1`` the trieste implementation uses ``tfp.math.erfcx`` to
    avoid catastrophic cancellation. opifex substitutes the closed form

    .. math::
        \log(\phi(u) + u \Phi(u)) = \log \phi(u) + \log(1 + u \cdot R(u))

    where ``R(u) = Phi(u) / phi(u)`` is the Mill's ratio reciprocal,
    computed via the standard asymptotic expansion
    ``R(u) \approx -1/u + 1/u^3`` for very negative ``u``. The branch
    boundary is the same ``u = -1`` used by trieste.
    """
    bound = jnp.asarray(-1.0, dtype=u.dtype)
    u_safe = jnp.where(u < bound, bound, u)
    log_ei_upper = jnp.log(
        jnp.maximum(jnorm.pdf(u_safe) + u_safe * jnorm.cdf(u_safe), _LOG_FLOOR)
    )

    # Asymptotic for u << -1: phi(u) + u * Phi(u) ≈ phi(u) * (1 + u * R(u))
    # where R(u) = Phi(u)/phi(u) ≈ -1/u * (1 - 1/u^2 + 3/u^4 - ...).
    # So 1 + u * R(u) ≈ 1 + (-1 + 1/u^2 - 3/u^4) = 1/u^2 - 3/u^4 + ...
    # For numerical safety we clip u away from zero in the lower branch.
    u_neg = jnp.where(u >= bound, bound, u)
    inv_u_sq = 1.0 / (u_neg**2)
    correction = inv_u_sq - 3.0 * inv_u_sq**2 + 15.0 * inv_u_sq**3
    log_correction = jnp.log(jnp.maximum(correction, _LOG_FLOOR))
    log_ei_lower = jnorm.logpdf(u_neg) + log_correction

    return jnp.where(u < bound, log_ei_lower, log_ei_upper)


def log_expected_improvement(
    predictive_dist: PredictiveDistribution,
    *,
    best_value: float,
) -> jax.Array:
    r"""Numerically stable log-EI.

    Ported from ``trieste/acquisition/function/function.py:269``
    (``log_expected_improvement.__call__``). Mitigates the
    vanishing-gradient issue of standard EI in regions where EI is tiny
    (Ament et al. 2023, *Unexpected EI*).
    """
    variance = _require_variance(predictive_dist)
    std = jnp.sqrt(variance)
    u = (best_value - predictive_dist.mean) / std
    return _log_ei_helper(u) + jnp.log(std)


def lower_confidence_bound(
    predictive_dist: PredictiveDistribution,
    *,
    beta: float,
) -> jax.Array:
    r"""LCB ``mu - beta * sigma`` (minimise for exploration).

    Ported from ``trieste/acquisition/function/function.py:571``
    (``lower_confidence_bound``).
    """
    if beta < 0:
        raise ValueError(f"beta must be non-negative; got {beta!r}")
    variance = _require_variance(predictive_dist)
    return predictive_dist.mean - beta * jnp.sqrt(variance)


def upper_confidence_bound(
    predictive_dist: PredictiveDistribution,
    *,
    beta: float,
) -> jax.Array:
    r"""UCB ``mu + beta * sigma`` (maximise for exploration).

    Mirror of LCB; same trieste reference at
    ``trieste/acquisition/function/function.py:571``
    (``NegativeLowerConfidenceBound`` simply negates ``lower_confidence_bound``).
    """
    if beta < 0:
        raise ValueError(f"beta must be non-negative; got {beta!r}")
    variance = _require_variance(predictive_dist)
    return predictive_dist.mean + beta * jnp.sqrt(variance)


def probability_of_improvement(
    predictive_dist: PredictiveDistribution,
    *,
    best_value: float,
) -> jax.Array:
    r"""Probability that ``f(x) < best_value`` under the posterior.

    Ported from ``trieste/acquisition/function/function.py:684``
    (``probability_below_threshold.__call__``). For a Gaussian posterior
    ``f(x) ~ N(mu, sigma^2)`` the closed form is
    ``Phi((best_value - mu) / sigma)``.
    """
    variance = _require_variance(predictive_dist)
    std = jnp.sqrt(variance)
    return jnorm.cdf((best_value - predictive_dist.mean) / std)


def _moment_matched_gaussian_entropy(
    samples: jax.Array,
    aleatoric_variance: jax.Array,
) -> jax.Array:
    """Entropy of the moment-matched Gaussian for an ensemble predictive.

    Returns ``H[N(mu_bar, sigma_total^2)]`` where ``sigma_total^2 =
    epistemic + aleatoric`` per candidate. This is the upper-bound
    approximation used by trieste's classification BALD (see
    ``active_learning.py:418``).
    """
    epistemic = jnp.var(samples, axis=0)
    total = epistemic + aleatoric_variance
    return 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * jnp.maximum(total, _VARIANCE_FLOOR))


def bald(
    predictive_dist: PredictiveDistribution,
    *,
    rngs: nnx.Rngs | jax.Array,
) -> jax.Array:
    r"""Bayesian Active Learning by Disagreement (regression-ensemble form).

    Ported from ``trieste/acquisition/function/active_learning.py:418``
    (``bayesian_active_learning_by_disagreement``). The trieste original
    targets binary GP-classification with a Bernoulli likelihood; opifex
    generalises to the regression ensemble carried by
    :class:`PredictiveDistribution.samples` because that's the shape every
    Phase-7 Bayesian backend already produces. The mutual information is

    .. math::
        \mathrm{BALD}(x) = H[\hat p(y \mid x)] - \mathbb{E}_{\theta} [H[p(y \mid x, \theta)]]

    where the predictive marginal entropy uses the moment-matched
    Gaussian approximation (same approximation as the trieste original)
    and the per-sample entropies are exact Gaussians with the carried
    aleatoric variance. The ``rngs`` argument is unused in the
    closed-form regression branch but is retained for API parity with
    sampling-based variants (it is consumed eagerly so callers see no
    spurious key reuse).
    """
    # Eagerly consume one key from the bald stream so downstream callers
    # never reuse a key across batch acquisitions.
    _ = extract_rng_key(rngs, streams=("active_bald", "active_acquire", "default"), context="BALD")

    if predictive_dist.samples is None:
        raise ValueError("BALD requires PredictiveDistribution.samples (ensemble members).")

    samples = predictive_dist.samples  # (num_samples, batch, ...)
    # Per-sample aleatoric variance (broadcasted to per-candidate shape).
    if predictive_dist.aleatoric is not None:
        aleatoric = jnp.maximum(predictive_dist.aleatoric, _VARIANCE_FLOOR)
    else:
        # Match trieste's safe fallback: treat aleatoric as the global
        # min-variance floor (degenerate ensembles still produce a finite
        # entropy and BALD reduces to zero when the samples agree).
        aleatoric = jnp.full(samples.shape[1:], _VARIANCE_FLOOR)

    # H[E[p]]: entropy of the moment-matched predictive Gaussian.
    h_predictive = _moment_matched_gaussian_entropy(samples, aleatoric)
    # E[H[p|theta]]: average entropy of the per-sample Gaussians (which
    # share the aleatoric variance under the ensemble model).
    h_conditional = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * aleatoric)
    return h_predictive - h_conditional


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


_StrategyFn = Callable[..., jax.Array]


def _max_variance_scores(predictive_dist: PredictiveDistribution) -> jax.Array:
    """Fallback "max-variance" acquisition (no trieste analogue; used by L2O)."""
    return _require_variance(predictive_dist)


def _dispatch_scores(
    predictive_dist: PredictiveDistribution,
    strategy: AcquisitionStrategy,
    rngs: nnx.Rngs | jax.Array,
    **kwargs: Any,
) -> jax.Array:
    if strategy is AcquisitionStrategy.BALD:
        return bald(predictive_dist, rngs=rngs)
    if strategy is AcquisitionStrategy.EI:
        return expected_improvement(predictive_dist, best_value=kwargs["best_value"])
    if strategy is AcquisitionStrategy.LOG_EI:
        return log_expected_improvement(predictive_dist, best_value=kwargs["best_value"])
    if strategy is AcquisitionStrategy.UCB:
        return upper_confidence_bound(predictive_dist, beta=kwargs.get("beta", 1.96))
    if strategy is AcquisitionStrategy.LCB:
        # The dispatcher selects the *most informative* points by argsort
        # of the strategy score; for LCB the more-explored direction is
        # lower, so we negate to keep argsort-descending semantics.
        return -lower_confidence_bound(predictive_dist, beta=kwargs.get("beta", 1.96))
    if strategy is AcquisitionStrategy.PI:
        return probability_of_improvement(predictive_dist, best_value=kwargs["best_value"])
    if strategy is AcquisitionStrategy.MAX_VARIANCE:
        return _max_variance_scores(predictive_dist)
    raise ValueError(f"Unknown acquisition strategy: {strategy!r}")  # pragma: no cover


def acquire(
    predictive_dist: PredictiveDistribution,
    *,
    strategy: AcquisitionStrategy | str,
    batch_size: int,
    rngs: nnx.Rngs | jax.Array,
    metadata: MetadataItems = (),
    **kwargs: Any,
) -> AcquiredBatch:
    """Named-strategy acquisition dispatcher.

    The top-``batch_size`` candidates (by descending score) are returned
    inside an :class:`AcquiredBatch`. This is the entry point invoked by
    the rewritten :class:`opifex.training.basic_trainer.ActiveUncertaintyLearner`
    — the trainer wraps the predictive distribution and delegates here.

    Note: this is a "naive top-K" greedy batch over single-point scores;
    for diversity-aware batch acquisition use :func:`batch_bald` or
    :func:`batch_mc_expected_improvement` from
    :mod:`opifex.uncertainty.active.batch_active`.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size!r}")
    if isinstance(strategy, str) and not isinstance(strategy, AcquisitionStrategy):
        try:
            strategy = AcquisitionStrategy(strategy)
        except ValueError as e:
            raise ValueError(f"Unknown acquisition strategy: {strategy!r}") from e

    scores = _dispatch_scores(predictive_dist, strategy, rngs, **kwargs)
    # Top-K descending. ``jnp.argsort`` is ascending; take the tail.
    if scores.ndim != 1:
        # Flatten trailing axes — acquisition is over candidate index axis 0.
        flat = scores.reshape(scores.shape[0], -1).mean(axis=-1)
    else:
        flat = scores
    k = min(batch_size, int(flat.shape[0]))
    top_indices = jnp.argsort(flat)[-k:][::-1]
    return AcquiredBatch(
        indices=top_indices,
        scores=flat,
        strategy=strategy.value,
        metadata=metadata,
    )

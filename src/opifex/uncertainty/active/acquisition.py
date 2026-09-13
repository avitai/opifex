"""Pure JAX single-point acquisition kernels for active learning.

Every kernel takes a :class:`opifex.uncertainty.types.PredictiveDistribution`
and returns a ``jax.Array`` of per-candidate utility scores. The kernels
are JAX-traceable (``jit`` / ``grad`` / ``vmap`` safe) and never call into
``flax.nnx``; that boundary is enforced by the duplicate-code gate at
``uncertainty_trainers.py``.

Each acquisition function cites the paper that defines it. Gaussian
densities come from ``jax.scipy.stats.norm``; :class:`PredictiveDistribution`
already carries posterior moments and samples, so the kernels need no
search-space or dataset plumbing.

Acquisition conventions:

* EI / Log-EI / PI follow the **minimisation** convention
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
  mixture entropy by Jensen's inequality).
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
from tensorflow_probability.substrates.jax.math import erfcx

from opifex.uncertainty.types import metadata_to_dict, MetadataItems, PredictiveDistribution


# Numerical floors used in entropy / log-prob computations.
_VARIANCE_FLOOR: float = 1e-12


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
        """Validate the batch size and acquisition strategy at construction time."""
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
            raise ValueError(f"AcquiredBatch.indices must be 1-D; got shape {self.indices.shape}")
        if not self.strategy:
            raise ValueError("AcquiredBatch.strategy must be a non-empty string.")


# ---------------------------------------------------------------------------
# Single-point acquisition kernels
# ---------------------------------------------------------------------------


def _require_variance(pd: PredictiveDistribution) -> jax.Array:
    """Return the floored predictive variance, raising if it is absent."""
    if pd.variance is None:
        raise ValueError("acquisition requires PredictiveDistribution.variance")
    return jnp.maximum(pd.variance, _VARIANCE_FLOOR)


def expected_improvement(
    predictive_dist: PredictiveDistribution,
    *,
    best_value: float,
) -> jax.Array:
    r"""Single-point Expected Improvement (minimisation convention).

    Jones, Schonlau & Welch (1998), *Efficient Global Optimization of
    Expensive Black-Box Functions*, J. Global Optim. 13(4):455-492:

    .. math::
        \mathrm{EI}(x) = (\eta - \mu(x)) \Phi\!\left(\tfrac{\eta - \mu(x)}{\sigma(x)}\right)
        + \sigma(x) \, \phi\!\left(\tfrac{\eta - \mu(x)}{\sigma(x)}\right)
    """
    variance = _require_variance(predictive_dist)
    std = jnp.sqrt(variance)
    u = (best_value - predictive_dist.mean) / std
    return (best_value - predictive_dist.mean) * jnorm.cdf(u) + std * jnorm.pdf(u)


def _log1mexp(x: jax.Array) -> jax.Array:
    r"""``log(1 - exp(x))`` for ``x < 0``, accurate near zero and far from it.

    Eq. (13) of Ament et al. (2023): ``log(-expm1(x))`` for ``x > -log 2`` and
    ``log1p(-exp(x))`` otherwise. Each branch evaluates on a masked copy of ``x``, so the branch
    that is not selected never produces a non-finite value or gradient.
    """
    log_two = jnp.log(jnp.asarray(2.0, dtype=x.dtype))
    is_near_zero = x > -log_two
    near_zero = jnp.where(is_near_zero, x, -log_two)
    far_from_zero = jnp.where(is_near_zero, -log_two, x)
    return jnp.where(
        is_near_zero, jnp.log(-jnp.expm1(near_zero)), jnp.log1p(-jnp.exp(far_from_zero))
    )


def _log_ei_helper(u: jax.Array) -> jax.Array:
    r"""Numerically stable ``log h(u) = log(phi(u) + u * Phi(u))``.

    Eq. (9) of Ament et al. (2023, *Unexpected Improvements to Expected Improvement for Bayesian
    Optimization*, NeurIPS, arXiv:2310.20708):

    .. math::
        \log h(u) =
        \begin{cases}
            \log(\phi(u) + u\,\Phi(u)) & u > -1, \\
            -u^2/2 - c_1 + \operatorname{log1mexp}(w(u)) & u_a < u \le -1, \\
            -u^2/2 - c_1 - 2 \log|u| & u \le u_a,
        \end{cases}

    with ``w(u) = log(erfcx(-u / sqrt(2)) |u|) + c_2``, ``c_1 = log(2 pi) / 2`` and
    ``c_2 = log(pi / 2) / 2``. The asymptotic branch starts at ``u_a = -1e3`` in float32 and
    ``u_a = -1e6`` in float64, the thresholds of the paper authors' implementation. ``erfcx``
    comes from TensorFlow Probability's JAX substrate. In float32 the gradient of the middle
    branch loses accuracy roughly as ``eps u^2``, because ``1 - exp(w)`` is close to ``1/u^2``.
    Each branch evaluates on a masked copy of ``u``, so values and gradients stay finite.
    """
    dtype = u.dtype
    bound = jnp.asarray(-1.0, dtype=dtype)
    asymptotic_bound = jnp.asarray(-1e6 if dtype == jnp.float64 else -1e3, dtype=dtype)

    is_direct = u > bound
    u_direct = jnp.where(is_direct, u, bound)
    log_h_direct = jnp.log(jnorm.pdf(u_direct) + u_direct * jnorm.cdf(u_direct))

    u_lower = jnp.where(is_direct, bound, u)
    is_erfcx = u_lower > asymptotic_bound
    u_erfcx = jnp.where(is_erfcx, u_lower, asymptotic_bound)
    scaled = -u_erfcx / jnp.sqrt(jnp.asarray(2.0, dtype=dtype))
    log_mills = jnp.log(erfcx(scaled) * jnp.abs(u_erfcx))
    log_mills = log_mills + 0.5 * jnp.log(jnp.asarray(0.5 * jnp.pi, dtype=dtype))
    log_phi = -0.5 * u_lower**2 - 0.5 * jnp.log(2.0 * jnp.pi)
    log_h_lower = log_phi + jnp.where(
        is_erfcx, _log1mexp(log_mills), -2.0 * jnp.log(jnp.abs(u_lower))
    )
    return jnp.where(is_direct, log_h_direct, log_h_lower)


def log_expected_improvement(
    predictive_dist: PredictiveDistribution,
    *,
    best_value: float,
) -> jax.Array:
    r"""Numerically stable log-EI.

    Mitigates the vanishing-gradient issue of standard EI in regions
    where EI is tiny (Ament, Daulton, Eriksson, Balandat & Bakshy 2023,
    *Unexpected Improvements to Expected Improvement for Bayesian
    Optimization*, NeurIPS, arXiv:2310.20708).
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

    Srinivas, Krause, Kakade & Seeger, *Gaussian Process Optimization in
    the Bandit Setting: No Regret and Experimental Design*, arXiv:0912.3995.
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

    The sign-flipped counterpart of :func:`lower_confidence_bound`
    (Srinivas et al., arXiv:0912.3995).
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

    Kushner (1964), *A New Method of Locating the Maximum Point of an
    Arbitrary Multipeak Curve in the Presence of Noise*, J. Basic Eng.
    86(1):97-106. For a Gaussian posterior
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
    epistemic + aleatoric`` per candidate, an upper-bound approximation
    of the mixture entropy.
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

    BALD scores the mutual information between the prediction and the
    model parameters (Houlsby, Hernandez-Lobato & Ghahramani 2014, ICML).
    This implementation targets the regression ensemble carried by
    :class:`PredictiveDistribution.samples` because that's the shape every
    Phase-7 Bayesian backend already produces. The mutual information is

    .. math::
        \mathrm{BALD}(x) = H[\hat p(y \mid x)] - \mathbb{E}_{\theta} [H[p(y \mid x, \theta)]]

    where the predictive marginal entropy uses the moment-matched
    Gaussian approximation and the per-sample entropies are exact
    Gaussians with the carried
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
        # Safe fallback: treat aleatoric as the global
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
    """Fallback "max-variance" acquisition (used by L2O)."""
    return _require_variance(predictive_dist)


def _dispatch_scores(
    predictive_dist: PredictiveDistribution,
    strategy: AcquisitionStrategy,
    rngs: nnx.Rngs | jax.Array,
    **kwargs: Any,
) -> jax.Array:
    """Dispatch table from strategy enum to acquisition kernel.

    LCB is negated so the downstream ``argsort`` selects "more
    informative" points (the lower-bound minimum) consistently with
    every other strategy that produces "higher = more informative".
    """
    dispatch: dict[AcquisitionStrategy, Callable[[], jax.Array]] = {
        AcquisitionStrategy.BALD: lambda: bald(predictive_dist, rngs=rngs),
        AcquisitionStrategy.EI: lambda: expected_improvement(
            predictive_dist, best_value=kwargs["best_value"]
        ),
        AcquisitionStrategy.LOG_EI: lambda: log_expected_improvement(
            predictive_dist, best_value=kwargs["best_value"]
        ),
        AcquisitionStrategy.UCB: lambda: upper_confidence_bound(
            predictive_dist, beta=kwargs.get("beta", 1.96)
        ),
        AcquisitionStrategy.LCB: lambda: (
            -lower_confidence_bound(predictive_dist, beta=kwargs.get("beta", 1.96))
        ),
        AcquisitionStrategy.PI: lambda: probability_of_improvement(
            predictive_dist, best_value=kwargs["best_value"]
        ),
        AcquisitionStrategy.MAX_VARIANCE: lambda: _max_variance_scores(predictive_dist),
    }
    try:
        return dispatch[strategy]()
    except KeyError as e:  # pragma: no cover
        raise ValueError(f"Unknown acquisition strategy: {strategy!r}") from e


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
    the rewritten :class:`opifex.training.uncertainty_trainers.ActiveUncertaintyLearner`
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
    flat = scores.reshape(scores.shape[0], -1).mean(axis=-1) if scores.ndim != 1 else scores
    k = min(batch_size, int(flat.shape[0]))
    top_indices = jnp.argsort(flat)[-k:][::-1]
    return AcquiredBatch(
        indices=top_indices,
        scores=flat,
        strategy=strategy.value,
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# Information-theoretic acquisitions (MES / GIBBON / IVR) — Slice 23
# -----------------------------------------------------------------------------


def min_value_entropy_search(
    *,
    means: jax.Array,
    variances: jax.Array,
    sampled_min_values: jax.Array,
) -> jax.Array:
    r"""Min-Value Entropy Search (Wang & Jegelka 2017, ICML, arXiv:1703.01968).

    Approximates the per-candidate information gain about the global minimum from Monte Carlo
    samples ``y*_s`` of the minimum value. Learning the minimum tells ``f(x) >= y*_s``, so
    eq. (6) of Wang & Jegelka, stated for a maximum, is applied to ``-f``:

    .. math::

        \alpha_{\text{MES}}(x)
            \approx \frac{1}{S}\,\sum_{s=1}^{S}
                \frac{\gamma_s\,\phi(\gamma_s)}{2\,\Phi(\gamma_s)} - \log \Phi(\gamma_s),
        \qquad \gamma_s = \frac{\mu(x) - y^*_s}{\sigma(x)}.

    Each term is the entropy reduction of the Gaussian predictive truncated to ``f >= y*_s``
    and is non-negative. ``phi / Phi`` is evaluated as ``exp(log phi - log Phi)``.

    Args:
        means: ``(N,)`` posterior means.
        variances: ``(N,)`` posterior variances.
        sampled_min_values: ``(S,)`` Monte-Carlo samples of the
            global minimum value.

    Returns:
        ``(N,)`` per-candidate MES scores.
    """
    stds = jnp.sqrt(jnp.maximum(variances, _VARIANCE_FLOOR))
    gamma = (means[:, None] - sampled_min_values[None, :]) / stds[:, None]
    log_cdf = jnorm.logcdf(gamma)
    ratio = jnp.exp(jnorm.logpdf(gamma) - log_cdf)
    contribution = 0.5 * gamma * ratio - log_cdf
    return jnp.mean(contribution, axis=-1)


def gibbon(
    *,
    means: jax.Array,
    variances: jax.Array,
    sampled_min_values: jax.Array,
) -> jax.Array:
    r"""GIBBON acquisition (Moss+ 2021, JMLR, arXiv:2102.03324).

    At batch size 1, GIBBON reduces exactly to MES (Moss+ 2021 §3),
    so this single-point form delegates to
    :func:`min_value_entropy_search`. The batch extension belongs in
    :mod:`opifex.uncertainty.active.batch_active`.
    """
    return min_value_entropy_search(
        means=means, variances=variances, sampled_min_values=sampled_min_values
    )


def integrated_variance_reduction_score(
    *,
    candidate_variances: jax.Array,
    cross_variances: jax.Array,
) -> jax.Array:
    r"""Integrated variance reduction (Cohn, Ghahramani & Jordan 1996, JAIR 4).

    Ranks candidates by how much they reduce the posterior variance
    when conditioned on. For each candidate ``i`` with posterior
    variance ``v_i`` and per-target cross-variance ``c_{i, t}``, the
    reduction at target ``t`` is ``c_{i, t}² / v_i``; the IVR score
    sums over the target set.

    Args:
        candidate_variances: ``(N,)`` posterior variance at each
            candidate.
        cross_variances: ``(N, T)`` cross-covariance between
            candidates and target points.

    Returns:
        ``(N,)`` integrated variance-reduction score, non-negative.
    """
    safe_var = jnp.maximum(candidate_variances, _VARIANCE_FLOOR)
    contributions = (cross_variances**2) / safe_var[:, None]
    return jnp.sum(contributions, axis=-1)

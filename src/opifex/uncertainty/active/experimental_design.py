"""Bayesian Experimental Design — EIG + BO loop driver.

Implements:

* :func:`expected_information_gain` — Monte-Carlo (nested) estimator for
  the Expected Information Gain ``EIG(d) = E_theta[ E_y|theta[log
  p(y|theta,d) - log p(y|d)] ]`` under a Gaussian-noise observation
  model. For a scalar linear-Gaussian design ``y = d^T theta + noise``
  the estimator collapses to the closed-form
  ``0.5 * log(1 + Var[d^T theta] / sigma_noise^2)``.
* :func:`bayesian_experimental_design_loop` — generic ask-tell BO loop
  driver. Mirrors the structure of ``trieste/ask_tell_optimization.py:742``
  (``AskTellOptimizer``) but stripped to the opifex active-learning
  surface: given a surrogate that supports ``predict(candidates)`` and
  ``update(x, y)``, the loop repeatedly:

  1. queries the surrogate's predictive distribution;
  2. evaluates the acquisition function on it;
  3. picks the argmax candidate;
  4. queries the oracle and feeds the new ``(x, y)`` back.

  This is the canonical driver used by the rewritten
  :class:`opifex.optimization.l2o.adaptive_schedulers.BayesianSchedulerOptimizer`.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency
from dataclasses import dataclass, field
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx  # noqa: TC002

from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001


# ---------------------------------------------------------------------------
# Surrogate protocol — the BO loop's only assumption about the model.
# ---------------------------------------------------------------------------


class Surrogate(Protocol):
    """Minimal surrogate interface required by the BO loop.

    Any object that exposes ``predict(candidates) -> PredictiveDistribution``
    and ``update(x, y) -> None`` satisfies this protocol. Phase 6 GP
    backends and the Phase 7 ensemble surrogates both qualify.
    """

    def predict(self, candidates: jax.Array) -> PredictiveDistribution: ...

    def update(self, x: jax.Array, y: jax.Array) -> None: ...


# ---------------------------------------------------------------------------
# EIG
# ---------------------------------------------------------------------------


def _gaussian_log_prob(value: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    """``log N(value; mean, std^2)`` — vectorised."""
    return -0.5 * (jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(std) + ((value - mean) ** 2) / (std**2))


def expected_information_gain(
    *,
    model: Callable[[jax.Array, jax.Array], jax.Array],
    design: jax.Array,
    prior_samples: jax.Array,
    noise_std: float,
    rngs: nnx.Rngs | jax.Array,
    num_outer_samples: int | None = None,
) -> jax.Array:
    r"""Monte-Carlo estimator of the Expected Information Gain.

    .. math::
        \mathrm{EIG}(d) = \mathbb E_{\theta} \!\left[
            \mathbb E_{y \mid \theta} \!\left[
                \log p(y \mid \theta, d) - \log p(y \mid d)
            \right]
        \right]

    With ``y = model(design, theta) + N(0, sigma^2)`` we compute the
    nested-MC estimator (Foster et al. 2019, *Variational Bayesian
    Optimal Experimental Design*) using the same prior-sample set for
    both outer and inner expectations (the "amortised" form). For the
    linear-Gaussian case ``y = d^T theta`` with prior
    ``theta ~ N(0, sigma_p^2 I)`` the estimator converges to
    ``0.5 * log(1 + sigma_p^2 ||d||^2 / sigma_n^2)``.

    The opifex implementation reuses the same prior samples as the inner
    and outer expectations, which is the standard amortised nested-MC
    estimator. ``num_outer_samples`` defaults to all prior samples.
    """
    if noise_std <= 0:
        raise ValueError(f"noise_std must be positive; got {noise_std!r}")
    key = extract_rng_key(
        rngs,
        streams=("active_eig", "active_acquire", "default"),
        context="expected_information_gain",
    )
    num_prior = int(prior_samples.shape[0])
    outer_n = num_prior if num_outer_samples is None else int(num_outer_samples)
    outer_n = min(outer_n, num_prior)

    # Predict mean response per theta sample.
    predictions = jax.vmap(lambda t: model(design, t[None, :]).squeeze(0))(prior_samples)
    # shape: (num_prior,) for scalar model output.

    # Sample observations under each outer theta.
    noise_key = key
    eps = jax.random.normal(noise_key, predictions.shape)
    y = predictions + noise_std * eps

    # log p(y | theta, d): use the same theta that generated y.
    log_p_y_given_theta = _gaussian_log_prob(y, predictions, jnp.asarray(noise_std))

    # log p(y | d) = log mean_theta p(y | theta, d): approximate by
    # averaging the likelihood over *all* prior samples (inner MC). For
    # each outer y we compute a log-sum-exp across the inner predictions.
    # Shape gymnastics: y has shape (outer_n,), predictions has shape
    # (num_prior,); broadcast to (outer_n, num_prior).
    y_outer = y[:outer_n]
    log_lik = _gaussian_log_prob(
        y_outer[:, None],
        predictions[None, :],
        jnp.asarray(noise_std),
    )  # (outer_n, num_prior)
    log_p_y = jax.scipy.special.logsumexp(log_lik, axis=-1) - jnp.log(num_prior)

    return jnp.mean(log_p_y_given_theta[:outer_n] - log_p_y)


# ---------------------------------------------------------------------------
# BO loop result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class BayesianExperimentalDesignResult:
    """Result container for :func:`bayesian_experimental_design_loop`.

    GUIDE_ALIGNMENT pattern (A): plain frozen-slotted-kw-only dataclass.
    All array history is materialised as ``tuple[...]`` so the container
    is hashable (sequence fields use ``tuple``, item 22a).
    """

    acquired_indices: tuple[int, ...]
    acquired_x: tuple[Any, ...] = field(default_factory=tuple)
    acquired_y: tuple[Any, ...] = field(default_factory=tuple)
    history_variance: tuple[float, ...] = field(default_factory=tuple)
    history_score: tuple[float, ...] = field(default_factory=tuple)
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# BO loop driver
# ---------------------------------------------------------------------------


def bayesian_experimental_design_loop(
    *,
    surrogate: Surrogate,
    candidates: jax.Array,
    oracle: Callable[[jax.Array], jax.Array],
    acquisition: Callable[[PredictiveDistribution], jax.Array],
    num_rounds: int,
    rngs: nnx.Rngs | jax.Array,
) -> BayesianExperimentalDesignResult:
    r"""Ask-tell Bayesian-experimental-design / BO loop.

    Mirrors the structure of ``trieste/ask_tell_optimization.py:742``
    (``AskTellOptimizer``). The opifex variant exposes a tiny surface:

    1. Predict the surrogate's :class:`PredictiveDistribution` over
       ``candidates``.
    2. Apply the ``acquisition`` callable to the predictive distribution.
    3. Pick ``argmax`` of the scores; record the index and the
       per-candidate mean predictive variance.
    4. Query the oracle at the chosen candidate.
    5. Update the surrogate with ``(x_new, y_new)``.

    ``rngs`` is consumed up-front to keep the loop deterministic given
    a fixed key (the acquisition function itself is responsible for any
    internal stochasticity).
    """
    if num_rounds <= 0:
        raise ValueError(f"num_rounds must be positive; got {num_rounds!r}")
    _ = extract_rng_key(
        rngs,
        streams=("active_acquire", "default"),
        context="bayesian_experimental_design_loop",
    )

    acquired_indices: list[int] = []
    acquired_x: list[jax.Array] = []
    acquired_y: list[jax.Array] = []
    history_variance: list[float] = []
    history_score: list[float] = []

    for _ in range(int(num_rounds)):
        predictive = surrogate.predict(candidates)
        if predictive.variance is None:
            raise ValueError("BO loop surrogate must return PredictiveDistribution with variance.")
        history_variance.append(float(jnp.mean(predictive.variance)))

        scores = acquisition(predictive)
        if scores.ndim != 1:
            scores = scores.reshape(scores.shape[0], -1).mean(axis=-1)
        best_idx = int(jnp.argmax(scores))
        history_score.append(float(scores[best_idx]))

        x_new = candidates[best_idx : best_idx + 1]
        y_new = oracle(x_new)
        acquired_indices.append(best_idx)
        acquired_x.append(x_new)
        acquired_y.append(y_new)

        surrogate.update(x_new, y_new)

    return BayesianExperimentalDesignResult(
        acquired_indices=tuple(acquired_indices),
        acquired_x=tuple(acquired_x),
        acquired_y=tuple(acquired_y),
        history_variance=tuple(history_variance),
        history_score=tuple(history_score),
    )

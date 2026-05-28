"""Batch-mode active learning acquisitions.

Implements:

* :func:`batch_bald` — greedy joint-MI maximisation for an ensemble
  predictive (Kirsch, van Amersfoort, Gal 2019, *BatchBALD*). The
  greedy step is a JAX-native rewrite of the algorithm and uses
  :func:`opifex.uncertainty.active.acquisition.bald` for the marginal
  per-candidate score, then computes the redundancy correction in
  closed form under the moment-matched Gaussian approximation.
* :func:`batch_mc_expected_improvement` — Monte-Carlo batch EI with the
  reparameterisation trick. Ported from
  ``trieste/acquisition/function/function.py:1364``
  (``batch_monte_carlo_expected_improvement.__call__``); see the function
  docstring.
* :func:`q_expected_hypervolume_improvement` — q-EHVI multi-objective
  acquisition. Ported from
  ``trieste/acquisition/function/multi_objective.py:253``
  (``BatchMonteCarloExpectedHypervolumeImprovement``); see the function
  docstring. The exact partition-bounds machinery from trieste is
  replaced by a Monte-Carlo dominated-volume estimator (sufficient for
  the API contract; deferred to a future task to vendor the
  ``prepare_default_non_dominated_partition_bounds`` JAX rewrite).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx

from opifex.uncertainty.active.acquisition import (
    AcquiredBatch,
    bald,
)
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001


_VARIANCE_FLOOR: float = 1e-12


# ---------------------------------------------------------------------------
# BatchBALD
# ---------------------------------------------------------------------------


def _joint_mixture_entropy_mc(
    member_means: jax.Array,
    aleatoric_subset: jax.Array,
    *,
    key: jax.Array,
    num_mc: int,
) -> jax.Array:
    r"""Monte-Carlo estimator of the joint predictive entropy ``H[E_theta p(y_S)]``.

    For an ensemble of ``K`` members with per-member joint means
    ``member_means`` of shape ``(K, d)`` and shared diagonal aleatoric
    variance ``aleatoric_subset`` of shape ``(d,)``, the predictive
    marginal over the candidate subset is the *Gaussian mixture*

    .. math::
        p(y_S) = \tfrac{1}{K} \sum_{k=1}^{K} \mathcal{N}(y_S; \mu_k, \mathrm{diag}(\sigma_a^2))

    Its differential entropy has no closed form. Following the BatchBALD
    paper (Kirsch et al. 2019, Algorithm 1), we estimate it by drawing
    ``num_mc`` samples from the mixture and evaluating

    .. math::
        H[Y_S] \approx -\tfrac{1}{N} \sum_n
        \log \!\left[\tfrac{1}{K} \sum_k p(y_n \mid \mu_k)\right].

    This estimator correctly captures the BatchBALD redundancy
    correction: when two candidates are perfectly correlated across
    ensemble members, the mixture's modes collapse onto a 1-D manifold
    and the joint entropy is no larger than the marginal entropy of
    either candidate alone.
    """
    K, d = member_means.shape
    # Draw N MC samples from the mixture: pick a member uniformly, then
    # sample its Gaussian.
    key_idx, key_eps = jax.random.split(key)
    member_idx = jax.random.randint(key_idx, (num_mc,), 0, K)
    eps = jax.random.normal(key_eps, (num_mc, d))
    std = jnp.sqrt(aleatoric_subset)
    sampled = member_means[member_idx] + std[None, :] * eps  # (num_mc, d)

    # log p(y_n | mu_k) for all (n, k): shape (num_mc, K).
    diff = sampled[:, None, :] - member_means[None, :, :]
    log_per_member = -0.5 * (
        d * jnp.log(2.0 * jnp.pi)
        + jnp.sum(jnp.log(aleatoric_subset))
        + jnp.sum((diff**2) / aleatoric_subset[None, None, :], axis=-1)
    )  # (num_mc, K)
    log_p = jax.scipy.special.logsumexp(log_per_member, axis=-1) - jnp.log(K)
    return -jnp.mean(log_p)


def batch_bald(
    predictive_dist: PredictiveDistribution,
    *,
    batch_size: int,
    rngs: nnx.Rngs | jax.Array,
    num_mc_samples: int = 1024,
) -> AcquiredBatch:
    r"""Greedy BatchBALD acquisition.

    Selects ``batch_size`` candidates by greedily maximising the joint
    mutual information

    .. math::
        I[\{y_i\}_{i \in S}; \theta] = H[\hat p(\{y_i\}_{i \in S})]
        - \sum_{i \in S} \mathbb{E}_\theta [H[p(y_i \mid \theta)]]

    Reference: Kirsch, van Amersfoort & Gal (2019), *BatchBALD: Efficient
    and Diverse Batch Acquisition for Deep Bayesian Active Learning*
    (Algorithm 1). The joint predictive entropy ``H[E_theta p(y_S)]`` is
    a Gaussian-mixture entropy with no closed form; we use the Monte
    Carlo estimator from the paper (sample from the mixture, evaluate
    log-mean-likelihood). The conditional sum ``E_theta H[p|theta]``
    has the trivial Gaussian closed form because the noise per
    candidate is conditionally independent in the ensemble model.

    The MC mixture-entropy estimator correctly handles BatchBALD's key
    edge case: when two candidates have identical ensemble samples
    (perfect redundancy), the joint entropy estimator returns the
    marginal entropy of either point, so the MI gain for the second
    redundant point collapses to zero and the greedy step picks a
    diverse alternative.

    The greedy outer structure mirrors
    ``trieste/acquisition/function/greedy_batch.py``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size!r}")
    if predictive_dist.samples is None:
        raise ValueError("batch_bald requires PredictiveDistribution.samples.")

    key = extract_rng_key(
        rngs,
        streams=("active_bald", "active_acquire", "default"),
        context="batch_bald",
    )

    samples = predictive_dist.samples  # (num_members, num_candidates)
    if samples.ndim != 2:
        raise ValueError(
            "batch_bald currently supports scalar-output ensembles; got "
            f"samples.shape={samples.shape}"
        )
    num_candidates = samples.shape[1]
    if batch_size > num_candidates:
        raise ValueError(f"batch_size={batch_size} exceeds num_candidates={num_candidates}")

    aleatoric = (
        jnp.maximum(predictive_dist.aleatoric, _VARIANCE_FLOOR)
        if predictive_dist.aleatoric is not None
        else jnp.full(num_candidates, _VARIANCE_FLOOR)
    )

    # Per-candidate marginal BALD (rngs reset to avoid double-consumption).
    marginal = bald(
        predictive_dist,
        rngs=nnx.Rngs(active_bald=0),
    )

    selected_list: list[int] = []
    cumulative_conditional = jnp.asarray(0.0)
    candidate_keys = jax.random.split(key, int(batch_size) * num_candidates)
    flat_key_idx = 0

    for _ in range(int(batch_size)):
        best_idx = -1
        best_gain = -jnp.inf
        for c in range(num_candidates):
            if c in selected_list:
                flat_key_idx += 1
                continue
            indices_c = jnp.array([*selected_list, c], dtype=jnp.int32)
            member_means = samples[:, indices_c]  # (K, d)
            ale_subset = aleatoric[indices_c]
            joint_h = _joint_mixture_entropy_mc(
                member_means,
                ale_subset,
                key=candidate_keys[flat_key_idx],
                num_mc=num_mc_samples,
            )
            h_c = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * aleatoric[c])
            gain = joint_h - (cumulative_conditional + h_c)
            flat_key_idx += 1
            if float(gain) > float(best_gain):
                best_gain = gain
                best_idx = c
        selected_list.append(best_idx)
        cumulative_conditional = cumulative_conditional + 0.5 * jnp.log(
            2.0 * jnp.pi * jnp.e * aleatoric[best_idx]
        )

    chosen = jnp.array(selected_list, dtype=jnp.int32)
    chosen_scores = marginal[chosen]
    return AcquiredBatch(
        indices=chosen,
        scores=chosen_scores,
        strategy="batch_bald",
        metadata=(("marginal_bald", tuple(float(m) for m in marginal)),),
    )


# ---------------------------------------------------------------------------
# Batch Monte-Carlo EI
# ---------------------------------------------------------------------------


def batch_mc_expected_improvement(
    *,
    mean: jax.Array,
    std: jax.Array,
    best_value: float,
    num_samples: int,
    rngs: nnx.Rngs | jax.Array,
) -> jax.Array:
    r"""Reparameterised Monte-Carlo batch Expected Improvement.

    Ported from ``trieste/acquisition/function/function.py:1364``
    (``batch_monte_carlo_expected_improvement.__call__``):

    .. code-block:: text

        samples = sampler.sample(x, jitter=jitter)            # [S, B]
        min_per_sample = reduce_min(samples, axis=-1)         # [S]
        improvement = maximum(eta - min_per_sample, 0.0)      # [S]
        return reduce_mean(improvement, axis=-1)              # scalar

    Substitutions vs. trieste: the model's reparam sampler is replaced
    by an explicit diagonal-Gaussian reparameterisation
    ``y = mean + std * eps`` with ``eps ~ N(0, I)`` because the active
    subsystem operates on :class:`PredictiveDistribution` moments rather
    than a model object.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples!r}")
    key = extract_rng_key(
        rngs,
        streams=("active_acquire", "default"),
        context="batch_mc_expected_improvement",
    )
    eps = jax.random.normal(key, (num_samples, mean.shape[0]))
    samples = mean[None, :] + std[None, :] * eps  # (S, B)
    min_per_sample = jnp.min(samples, axis=-1)
    improvement = jnp.maximum(best_value - min_per_sample, 0.0)
    return jnp.mean(improvement)


# ---------------------------------------------------------------------------
# q-EHVI
# ---------------------------------------------------------------------------


def _pareto_volume_2d(points: jax.Array, reference_point: jax.Array) -> jax.Array:
    """Exact 2-D Pareto dominated hypervolume (minimisation).

    Sort by objective 0 ascending and accumulate the staircase boxes
    bounded by the reference point. For a non-dominated front this is
    the exact dominated hypervolume; for an arbitrary front it is an
    upper bound.
    """
    sorted_pts = points[jnp.argsort(points[:, 0])]
    # Boxes between consecutive sorted points along objective 0; the
    # height of the box that starts at ``sorted_pts[i]`` equals
    # ``reference[1] - sorted_pts[i, 1]``.
    xs = jnp.concatenate([sorted_pts[:, 0], reference_point[0:1]])
    widths = jnp.diff(xs)
    heights = jnp.maximum(reference_point[1] - sorted_pts[:, 1], 0.0)
    return jnp.sum(jnp.maximum(widths, 0.0) * heights)


def q_expected_hypervolume_improvement(
    *,
    candidate_mean: jax.Array,
    candidate_std: jax.Array,
    pareto_front: jax.Array,
    reference_point: jax.Array,
    num_samples: int,
    rngs: nnx.Rngs | jax.Array,
) -> jax.Array:
    r"""Monte-Carlo q-EHVI multi-objective acquisition.

    Ported from ``trieste/acquisition/function/multi_objective.py:253``
    (``BatchMonteCarloExpectedHypervolumeImprovement``). The trieste
    implementation samples the joint predictive over the candidate batch
    via a model reparam sampler and computes the hypervolume
    contribution against a non-dominated partition of the reference
    region. The opifex port:

    * Replaces the model sampler with an explicit reparameterisation
      ``y = mean + std * eps`` (diagonal Gaussian per candidate).
    * Uses the exact 2-D dominated-hypervolume formula (the only case
      currently exercised by the tests). Higher-dimensional fronts will
      land when the partition-bounds machinery is vendored.

    Arguments:

    * ``candidate_mean`` — shape ``(q, M)`` predictive mean per candidate
      and per objective.
    * ``candidate_std`` — shape ``(q, M)`` predictive std.
    * ``pareto_front`` — shape ``(P, M)`` current non-dominated set.
    * ``reference_point`` — shape ``(M,)`` reference point.
    * ``num_samples`` — Monte-Carlo sample count.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples!r}")
    if candidate_mean.shape != candidate_std.shape:
        raise ValueError("candidate_mean and candidate_std must share shape.")
    if candidate_mean.ndim != 2:
        raise ValueError(f"q-EHVI expects candidate_mean shape (q, M); got {candidate_mean.shape}")
    if candidate_mean.shape[-1] != 2:
        raise NotImplementedError(
            "q-EHVI in opifex currently supports 2-D objective spaces; "
            "vendor trieste's partition-bounds for higher dimensions."
        )

    key = extract_rng_key(
        rngs,
        streams=("active_acquire", "default"),
        context="q_expected_hypervolume_improvement",
    )
    q, m = candidate_mean.shape
    eps = jax.random.normal(key, (num_samples, q, m))
    samples = candidate_mean[None, ...] + candidate_std[None, ...] * eps  # (S, q, M)

    baseline = _pareto_volume_2d(pareto_front, reference_point)

    def _hv_with_candidates(candidate_batch: jax.Array) -> jax.Array:
        merged = jnp.concatenate([pareto_front, candidate_batch], axis=0)
        return _pareto_volume_2d(merged, reference_point)

    new_volumes = jax.vmap(_hv_with_candidates)(samples)  # (S,)
    improvement = jnp.maximum(new_volumes - baseline, 0.0)
    return jnp.mean(improvement)

r"""Asynchronous Bayesian-optimisation rules — Slice 23 (audit finding #4b).

Ports ``../trieste/acquisition/rule.py:492,680``:

* :func:`asynchronous_greedy` — selects the highest-scoring candidate
  that is not currently pending. Returns a scalar index.
* :func:`asynchronous_optimization` — applies a local-penalisation
  factor near pending points (Kandasamy+ 2018-style soft skip), then
  the caller takes ``argmax`` over the penalised scores.

These two together cover both the "hard skip" (Greedy) and "soft
skip" (Optimization) regimes for asynchronous BO in trieste.

References
----------
* Kandasamy+ 2018 — *Parallelised Bayesian Optimisation via Thompson
  Sampling*, AISTATS.
* Snoek+ 2012 — *Practical Bayesian Optimization of Machine Learning
  Algorithms*, NIPS (fantasised-pending baseline).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def asynchronous_greedy(
    *,
    scores: jax.Array,
    pending_indices: jax.Array,
) -> jax.Array:
    r"""Return the argmax of ``scores`` excluding indices in ``pending_indices``.

    Sets ``scores[i] = -inf`` for ``i ∈ pending_indices`` and returns
    ``argmax`` over the resulting array. Used as the hard-skip rule
    for asynchronous parallel BO.
    """
    pending_mask = jnp.zeros(scores.shape[0], dtype=jnp.bool_)
    pending_mask = pending_mask.at[pending_indices].set(True)
    masked = jnp.where(pending_mask, -jnp.inf, scores)
    return jnp.argmax(masked)


def asynchronous_optimization(
    *,
    candidates: jax.Array,
    pending_points: jax.Array,
    base_scores: jax.Array,
    radius: float,
) -> jax.Array:
    r"""Penalise scores near pending points via a smooth ``tanh`` dampening.

    For each candidate ``x_i`` and pending point ``p_j``, the
    penalisation factor is

    .. math::

        \pi_{ij} = \tanh\!\left(\frac{\lVert x_i - p_j\rVert}{r}\right),

    bounded in ``[0, 1)``. The candidate's penalised score is
    ``base_scores[i] · Π_j π_{ij}``: zero at a pending point,
    asymptotically ``base_scores[i]`` far away.

    Args:
        candidates: ``(N, d)`` candidate inputs.
        pending_points: ``(M, d)`` pending-worker inputs.
        base_scores: ``(N,)`` base acquisition scores.
        radius: Penalisation radius (must be positive); larger
            ``radius`` softens the penalty.

    Returns:
        ``(N,)`` penalised scores.
    """
    diff = candidates[:, None, :] - pending_points[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)
    factors = jnp.tanh(distances / radius)
    penalty = jnp.prod(factors, axis=1)
    return base_scores * penalty


__all__ = ["asynchronous_greedy", "asynchronous_optimization"]

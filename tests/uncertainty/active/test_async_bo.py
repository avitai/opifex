r"""Tests for ``active/async_bo.py`` — Slice 23 (audit finding #4b).

Phase 8 Task 8.3 (``08-...:586-601``) requires ``active/async_bo.py``
shipping AsynchronousOptimization + AsynchronousGreedy rules (port of
``../trieste/acquisition/rule.py:492,680``).

The asynchronous BO regime maintains a set of "pending" candidates
(workers currently evaluating but not yet returned) and combines them
with completed observations to score the next acquisition target.

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


def test_asynchronous_greedy_skips_indices_marked_pending() -> None:
    """Indices in ``pending_indices`` must not be selected as the next acquisition."""
    from opifex.uncertainty.active.async_bo import asynchronous_greedy

    scores = jnp.array([0.1, 0.9, 0.7, 0.5, 0.8])
    pending = jnp.array([1, 4])  # the top-two scoring indices are pending
    selected = asynchronous_greedy(scores=scores, pending_indices=pending)
    # Best non-pending score is index 2 (0.7).
    assert int(selected) == 2


def test_asynchronous_greedy_returns_global_argmax_when_no_pending() -> None:
    """With no pending workers, the rule reduces to argmax over the scores."""
    from opifex.uncertainty.active.async_bo import asynchronous_greedy

    scores = jnp.array([0.2, 0.6, 0.4, 0.8])
    selected = asynchronous_greedy(scores=scores, pending_indices=jnp.array([], dtype=jnp.int32))
    assert int(selected) == 3


def test_asynchronous_optimization_penalises_pending_via_kernel_distance() -> None:
    """``asynchronous_optimization`` reduces scores near pending workers (local penalisation)."""
    from opifex.uncertainty.active.async_bo import asynchronous_optimization

    candidates = jnp.array([[0.0], [0.5], [1.0]])
    pending_points = jnp.array([[0.5]])  # one worker is busy near candidate 1
    base_scores = jnp.array([1.0, 1.0, 1.0])
    penalised = asynchronous_optimization(
        candidates=candidates,
        pending_points=pending_points,
        base_scores=base_scores,
        radius=0.3,
    )
    # The candidate at distance 0 from pending is penalised the most.
    assert float(penalised[1]) < float(penalised[0])
    assert float(penalised[1]) < float(penalised[2])


def test_asynchronous_optimization_is_jit_compatible() -> None:
    """Async-opt compiles under ``jax.jit``."""
    from opifex.uncertainty.active.async_bo import asynchronous_optimization

    candidates = jnp.array([[0.0], [0.5], [1.0]])
    pending = jnp.array([[0.3]])
    base = jnp.array([1.0, 1.0, 1.0])
    jitted = jax.jit(asynchronous_optimization, static_argnames=("radius",))
    out = jitted(candidates=candidates, pending_points=pending, base_scores=base, radius=0.2)
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))

"""Batch-acquisition tests for Task 8.3.

* ``batch_bald`` returns a non-redundant batch and strictly beats top-k
  BALD on a controlled synthetic example where two pool points have the
  same per-point BALD but identical samples (so picking both is
  redundant). batch-BALD must pick a more diverse alternative.
* ``batch_mc_expected_improvement`` matches the Monte Carlo formula from
  ``trieste.acquisition.function.function:1364``.
* ``q_expected_hypervolume_improvement`` returns non-negative contributions
  whose sum equals the closed-form box-volume contributions for a known
  Pareto front.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.active.acquisition import bald
from opifex.uncertainty.active.batch_active import (
    batch_bald,
    batch_mc_expected_improvement,
    q_expected_hypervolume_improvement,
)
from opifex.uncertainty.types import PredictiveDistribution


def _redundant_predictive() -> PredictiveDistribution:
    """Pool of 4 candidates with one redundant high-BALD pair.

    Uses K=4 ensemble members so the joint mixture-entropy estimator
    actually exercises the redundancy correction. Samples are designed
    so that:

    * Indices 0 and 1 are *perfectly correlated* across members
      (samples[:, 0] == samples[:, 1]). Observing one fully determines
      the other → no extra information about theta from observing both.
    * Index 2 has a DIFFERENT pattern across members so it adds new
      information not extractable from index 0 alone.
    * Index 3 has zero disagreement across members (low marginal BALD).
    """
    samples = jnp.array(
        [
            # member 0
            [-1.0, -1.0, 1.0, 0.0],
            # member 1
            [1.0, 1.0, -1.0, 0.0],
            # member 2: same disagreement on 0,1; OPPOSITE pattern on 2
            [-1.0, -1.0, -1.0, 0.0],
            # member 3
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    mean = jnp.mean(samples, axis=0)
    epistemic = jnp.var(samples, axis=0)
    aleatoric = jnp.full_like(epistemic, 0.05)
    total = epistemic + aleatoric
    return PredictiveDistribution(
        mean=mean,
        samples=samples,
        variance=total,
        epistemic=epistemic,
        aleatoric=aleatoric,
        total_uncertainty=total,
    )


class TestBatchBALD:
    def test_batch_bald_beats_top_k_on_redundant_pool(self) -> None:
        pd = _redundant_predictive()
        rngs = nnx.Rngs(active_bald=0)

        # Naive top-k BALD: indices 0 and 1 have identical marginal BALD
        # (their ensemble samples coincide), and both exceed the BALD of
        # the diverse-but-still-disagreeing point 2. So top-2-by-marginal
        # would pick BOTH redundant points {0, 1}.
        per_point = bald(pd, rngs=nnx.Rngs(active_bald=0))
        # Marginal BALD: indices 0 and 1 are highest (tied); index 2 next.
        assert float(per_point[0]) == pytest.approx(float(per_point[1]), abs=1e-7)
        assert float(per_point[0]) >= float(per_point[2])

        batch = batch_bald(pd, batch_size=2, rngs=rngs)

        # batch-BALD must break the redundancy: the chosen pair must
        # include the diverse point 2 (which adds independent information
        # about theta) rather than both redundant points {0, 1}.
        chosen = {int(i) for i in batch.indices}
        assert chosen != {0, 1}, "batch-BALD failed to break redundancy"
        assert 2 in chosen, "batch-BALD missed the diverse point 2"

    def test_batch_bald_deterministic_for_fixed_key(self) -> None:
        pd = _redundant_predictive()
        a = batch_bald(pd, batch_size=2, rngs=nnx.Rngs(active_bald=7))
        b = batch_bald(pd, batch_size=2, rngs=nnx.Rngs(active_bald=7))
        assert jnp.array_equal(a.indices, b.indices)


class TestBatchMCExpectedImprovement:
    def test_matches_mc_formula(self) -> None:
        """Trieste batch_monte_carlo_expected_improvement:1364 ported."""
        rngs = nnx.Rngs(active_acquire=0)
        # candidate batch: shape (q, d) — here q=3, d=1 trivial domain.
        # The predictive over the joint batch is N(mean, cov).
        mean = jnp.array([0.5, 0.7, 0.9])
        # diagonal covariance for simplicity (trieste path handles
        # arbitrary cov via reparam sampler).
        std = jnp.array([0.3, 0.2, 0.4])
        eta = 1.0
        num_samples = 8192
        # Manual Monte-Carlo ground truth.
        key = jax.random.PRNGKey(0)
        z = jax.random.normal(key, (num_samples, 3))
        samples = mean[None, :] + std[None, :] * z
        # per-batch best = min over batch dim.
        min_per_sample = jnp.min(samples, axis=-1)
        expected_ei = jnp.mean(jnp.maximum(eta - min_per_sample, 0.0))

        out = batch_mc_expected_improvement(
            mean=mean,
            std=std,
            best_value=eta,
            num_samples=num_samples,
            rngs=rngs,
        )
        assert jnp.allclose(out, expected_ei, rtol=5e-2)


class TestQEHVI:
    def test_qehvi_nonnegative_and_matches_known_pareto_volume(self) -> None:
        """Two-objective minimisation; non-dominated points (1,3),(2,2),(3,1).

        Reference point (4, 4). qEHVI under a deterministic predictive
        (zero variance) at point (1.5, 2.5) adds box volume
        ``(2 - 1.5) * (3 - 2.5) = 0.25`` to the hypervolume.
        """
        rngs = nnx.Rngs(active_acquire=0)
        pareto_front = jnp.array(
            [
                [1.0, 3.0],
                [2.0, 2.0],
                [3.0, 1.0],
            ]
        )
        reference_point = jnp.array([4.0, 4.0])
        # Single candidate batch (q=1), 2 objectives, deterministic.
        candidate_mean = jnp.array([[1.5, 2.5]])
        candidate_std = jnp.array([[1e-6, 1e-6]])

        result = q_expected_hypervolume_improvement(
            candidate_mean=candidate_mean,
            candidate_std=candidate_std,
            pareto_front=pareto_front,
            reference_point=reference_point,
            num_samples=1024,
            rngs=rngs,
        )

        assert float(result) >= 0.0
        # With near-zero noise the MC estimate should be close to the
        # exact incremental hypervolume contribution of (1.5, 2.5).
        assert float(result) == pytest.approx(0.25, abs=1e-2)

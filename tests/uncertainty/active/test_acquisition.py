"""Single-point acquisition-function tests for Task 8.3.

The tests pin the published acquisition-function formulas exactly:

* BALD = ``H[E[p]] - E[H[p]]`` (mutual information between the predictive
  marginal and the per-sample predictives). For a regression ensemble of
  Gaussian samples ``(num_samples, batch)`` we evaluate the entropy of the
  per-sample Gaussians (each variance fixed by an aleatoric scale) and the
  entropy of the predictive Gaussian mixture under a moment-matching
  Gaussian approximation. The closed-form ground truth is computed inside
  the test so the formula isn't trusted to the implementation under test.
* EI / Log-EI / UCB / LCB / PI follow the trieste reference (citations in
  :mod:`opifex.uncertainty.active.acquisition`).
* ``acquire(...)`` is the named-strategy dispatcher referenced by the
  rewritten :class:`ActiveUncertaintyLearner`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.active.acquisition import (
    acquire,
    AcquiredBatch,
    AcquisitionStrategy,
    bald,
    expected_improvement,
    log_expected_improvement,
    lower_confidence_bound,
    probability_of_improvement,
    upper_confidence_bound,
)
from opifex.uncertainty.types import PredictiveDistribution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_member_predictive(
    *,
    member_means: tuple[float, float],
    aleatoric_var: float,
) -> tuple[PredictiveDistribution, float]:
    """Construct a 2-member ensemble + return the analytic BALD value.

    For two equally weighted Gaussians ``N(mu_k, sigma^2)`` with shared
    aleatoric variance, the predictive marginal is a Gaussian mixture; its
    entropy is approximated by the moment-matched Gaussian. The analytic
    BALD value (under that moment-matching) is

    ``BALD = 0.5 * log(predictive_var / aleatoric_var)``.

    Each ``samples`` row is ``(num_samples, batch)`` with batch=1.
    """
    mu1, mu2 = member_means
    samples = jnp.array([[mu1], [mu2]])
    mean = jnp.array([0.5 * (mu1 + mu2)])
    epistemic = jnp.array([0.5 * ((mu1 - mean[0]) ** 2 + (mu2 - mean[0]) ** 2)])
    aleatoric = jnp.array([aleatoric_var])
    total = epistemic + aleatoric
    pd = PredictiveDistribution(
        mean=mean,
        samples=samples,
        variance=total,
        epistemic=epistemic,
        aleatoric=aleatoric,
        total_uncertainty=total,
    )
    expected_bald = float(0.5 * jnp.log(total[0] / aleatoric[0]))
    return pd, expected_bald


# ---------------------------------------------------------------------------
# BALD
# ---------------------------------------------------------------------------


class TestBALD:
    """Mutual information acquisition (regression-ensemble form)."""

    def test_bald_two_member_matches_analytic_value(self) -> None:
        pd, expected = _two_member_predictive(
            member_means=(-1.0, 1.0),
            aleatoric_var=0.25,
        )
        rngs = nnx.Rngs(active_bald=0)

        scores = bald(pd, rngs=rngs)

        assert scores.shape == (1,)
        assert jnp.allclose(scores[0], expected, atol=1e-6)

    def test_bald_zero_when_members_identical(self) -> None:
        """No disagreement => zero mutual information."""
        pd, expected = _two_member_predictive(
            member_means=(0.5, 0.5),
            aleatoric_var=0.1,
        )
        rngs = nnx.Rngs(active_bald=0)

        scores = bald(pd, rngs=rngs)

        assert expected == pytest.approx(0.0, abs=1e-12)
        assert jnp.allclose(scores, 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# EI, log-EI
# ---------------------------------------------------------------------------


class TestExpectedImprovement:
    """Analytic EI (trieste expected_improvement port)."""

    def test_ei_matches_closed_form(self) -> None:
        mean = jnp.array([0.5, 1.2])
        variance = jnp.array([0.04, 0.09])
        eta = 1.0
        std = jnp.sqrt(variance)
        u = (eta - mean) / std
        # trieste form: (eta - mean) * Phi((eta-mean)/sigma) + sigma * phi(...)
        from jax.scipy.stats import norm as jnorm

        expected = (eta - mean) * jnorm.cdf(u) + std * jnorm.pdf(u)

        pd = PredictiveDistribution(mean=mean, variance=variance)
        out = expected_improvement(pd, best_value=eta)

        assert jnp.allclose(out, expected, atol=1e-6)

    def test_ei_nonnegative(self) -> None:
        mean = jnp.array([-2.0, 0.0, 5.0])
        variance = jnp.array([0.1, 0.5, 1.0])
        pd = PredictiveDistribution(mean=mean, variance=variance)
        out = expected_improvement(pd, best_value=0.0)
        assert jnp.all(out >= -1e-7)

    def test_logei_matches_log_of_ei_in_safe_regime(self) -> None:
        mean = jnp.array([-0.5, 0.5])  # well-separated from eta
        variance = jnp.array([1.0, 1.0])
        pd = PredictiveDistribution(mean=mean, variance=variance)
        ei = expected_improvement(pd, best_value=1.0)
        log_ei = log_expected_improvement(pd, best_value=1.0)
        assert jnp.allclose(log_ei, jnp.log(ei), atol=1e-5)


# ---------------------------------------------------------------------------
# UCB / LCB
# ---------------------------------------------------------------------------


class TestConfidenceBound:
    def test_ucb_formula(self) -> None:
        mean = jnp.array([0.0, 1.0])
        variance = jnp.array([0.25, 0.04])
        beta = 1.5
        pd = PredictiveDistribution(mean=mean, variance=variance)
        out = upper_confidence_bound(pd, beta=beta)
        assert jnp.allclose(out, mean + beta * jnp.sqrt(variance), atol=1e-7)

    def test_lcb_formula(self) -> None:
        mean = jnp.array([0.0, 1.0])
        variance = jnp.array([0.25, 0.04])
        beta = 1.5
        pd = PredictiveDistribution(mean=mean, variance=variance)
        out = lower_confidence_bound(pd, beta=beta)
        assert jnp.allclose(out, mean - beta * jnp.sqrt(variance), atol=1e-7)

    def test_ucb_beta_must_be_nonnegative(self) -> None:
        pd = PredictiveDistribution(mean=jnp.zeros(2), variance=jnp.ones(2))
        with pytest.raises(ValueError, match="beta"):
            upper_confidence_bound(pd, beta=-0.1)


# ---------------------------------------------------------------------------
# PI
# ---------------------------------------------------------------------------


class TestProbabilityOfImprovement:
    def test_pi_matches_normal_cdf(self) -> None:
        from jax.scipy.stats import norm as jnorm

        mean = jnp.array([0.0, 0.5])
        variance = jnp.array([0.04, 0.25])
        threshold = 0.3
        expected = jnorm.cdf((threshold - mean) / jnp.sqrt(variance))
        pd = PredictiveDistribution(mean=mean, variance=variance)
        out = probability_of_improvement(pd, best_value=threshold)
        assert jnp.allclose(out, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# acquire(...) dispatcher
# ---------------------------------------------------------------------------


class TestAcquireDispatcher:
    """`acquire(...)` must route to the correct kernel by name."""

    def _pd(self) -> PredictiveDistribution:
        mean = jnp.array([0.0, 0.5, 1.0, 1.5])
        variance = jnp.array([0.1, 0.2, 0.3, 0.4])
        samples = jnp.stack([mean, mean + 0.1, mean - 0.1])
        return PredictiveDistribution(
            mean=mean,
            variance=variance,
            samples=samples,
            epistemic=jnp.var(samples, axis=0),
            aleatoric=variance - jnp.var(samples, axis=0),
            total_uncertainty=variance,
        )

    def test_dispatches_to_ucb_by_name(self) -> None:
        pd = self._pd()
        rngs = nnx.Rngs(active_acquire=0)

        result = acquire(
            pd,
            strategy=AcquisitionStrategy.UCB,
            batch_size=2,
            rngs=rngs,
            beta=2.0,
        )

        assert isinstance(result, AcquiredBatch)
        assert result.indices.shape == (2,)
        # UCB picks the highest mean + beta*std combination.
        assert pd.variance is not None
        manual = pd.mean + 2.0 * jnp.sqrt(pd.variance)
        top_two = jnp.argsort(manual)[-2:]
        assert {int(i) for i in result.indices} == {int(i) for i in top_two}
        assert result.strategy == AcquisitionStrategy.UCB.value

    def test_dispatches_to_bald(self) -> None:
        pd = self._pd()
        rngs = nnx.Rngs(active_acquire=0)
        result = acquire(
            pd,
            strategy=AcquisitionStrategy.BALD,
            batch_size=2,
            rngs=rngs,
        )
        assert result.scores.shape == (4,)
        assert result.indices.shape == (2,)
        assert result.strategy == AcquisitionStrategy.BALD.value

    def test_unknown_strategy_raises(self) -> None:
        pd = self._pd()
        rngs = nnx.Rngs(active_acquire=0)
        with pytest.raises(ValueError, match="strategy"):
            acquire(pd, strategy="banana", batch_size=2, rngs=rngs)


# ---------------------------------------------------------------------------
# Container patterns
# ---------------------------------------------------------------------------


class TestAcquiredBatchContainer:
    """`AcquiredBatch` must be a flax struct (pattern B)."""

    def test_acquired_batch_metadata_dict(self) -> None:
        batch = AcquiredBatch(
            indices=jnp.array([0, 2]),
            scores=jnp.array([0.1, 0.5, 0.4]),
            strategy="ucb",
            metadata=(("beta", 1.5),),
        )
        assert batch.metadata_dict() == {"beta": 1.5}

    def test_acquired_batch_is_pytree(self) -> None:
        batch = AcquiredBatch(
            indices=jnp.array([0, 2]),
            scores=jnp.array([0.1, 0.5, 0.4]),
            strategy="ucb",
        )
        leaves, _ = jax.tree.flatten(batch)
        # indices + scores are leaves; strategy / metadata are static.
        assert len(leaves) == 2

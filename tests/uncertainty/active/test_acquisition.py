"""Single-point acquisition-function tests for Task 8.3.

The tests pin the published acquisition-function formulas exactly:

* BALD = ``H[E[p]] - E[H[p]]`` (mutual information between the predictive
  marginal and the per-sample predictives). For a regression ensemble of
  Gaussian samples ``(num_samples, batch)`` we evaluate the entropy of the
  per-sample Gaussians (each variance fixed by an aleatoric scale) and the
  entropy of the predictive Gaussian mixture under a moment-matching
  Gaussian approximation. The closed-form ground truth is computed inside
  the test so the formula isn't trusted to the implementation under test.
* EI / Log-EI / UCB / LCB / PI follow the published formulas (citations in
  :mod:`opifex.uncertainty.active.acquisition`).
* ``acquire(...)`` is the named-strategy dispatcher referenced by the
  rewritten :class:`ActiveUncertaintyLearner`.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from scipy import special
from scipy.stats import norm

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
    """Analytic EI (Jones, Schonlau & Welch 1998)."""

    def test_ei_matches_closed_form(self) -> None:
        mean = jnp.array([0.5, 1.2])
        variance = jnp.array([0.04, 0.09])
        eta = 1.0
        std = jnp.sqrt(variance)
        u = (eta - mean) / std
        # closed form: (eta - mean) * Phi((eta-mean)/sigma) + sigma * phi(...)
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


_EPS32 = float(np.finfo(np.float32).eps)


def _reference_log_h(z: float) -> float:
    """Float64 ``log(phi(z) + z Phi(z))`` by Ament et al. (2023), eq. 9, with scipy's ``erfcx``."""
    if z > -1.0:
        return math.log(norm.pdf(z) + z * norm.cdf(z))
    w = math.log(special.erfcx(-z / math.sqrt(2.0)) * abs(z)) + 0.5 * math.log(0.5 * math.pi)
    return -0.5 * z * z - 0.5 * math.log(2.0 * math.pi) + math.log(-math.expm1(w))


class TestLogExpectedImprovementBranches:
    """Log-EI across the three branches of Ament et al. (2023), eq. 9.

    With unit standard deviation and ``best_value = 0``, the log-EI at mean ``-z`` is
    ``log h(z) = log(phi(z) + z Phi(z))``. Float32 values are compared with a float64 evaluation of
    eq. 9 on ``z`` from 3 down to ``-1e6``: the direct branch (``z > -1``), the ``erfcx`` branch,
    and the asymptotic branch below ``-1e3``. The grid includes ``z`` in ``[-13.4, -12.9]``, where
    ``jax.scipy.special.erfcx`` returns 0 in float32 for arguments in ``[9.195, 9.419]``.

    Values are compared within 1e-6 relative (measured 3.6e-7). In the ``erfcx`` branch the
    gradient loses accuracy as float32 cancels ``1 - exp(w)``, which is close to ``1/z^2``, so the
    gradient is compared within ``2 eps32 z^2 + 1e-5`` relative (measured at most 0.72 of that
    bound, with a largest error of 9.9e-2 at ``z = -928``). Outside that branch the gradient is
    compared within 1e-5 relative (measured 2.5e-7 above ``z = -1``). Every gradient tolerance also
    carries ``2 eps64 z^2``, the error of the float64 reference itself, which subtracts two numbers
    near ``-z^2/2``; below ``z = -1e3`` that term dominates (measured 1.65e-4 near ``|z| = 1e6``,
    0.65 of the tolerance).
    """

    _GRID = np.concatenate(
        [
            np.linspace(3.0, -3.0, 61),
            -np.logspace(0.5, 6.0, 400),
            -np.linspace(12.9, 13.4, 51),
        ]
    ).astype(np.float32)

    @staticmethod
    def _log_h(z: jax.Array) -> jax.Array:
        """Return log-EI at mean ``-z`` with unit variance and ``best_value = 0``."""
        predictive = PredictiveDistribution(mean=-z, variance=jnp.ones_like(z))
        return log_expected_improvement(predictive, best_value=0.0)

    def test_matches_the_eq9_reference(self) -> None:
        values = np.asarray(self._log_h(jnp.asarray(self._GRID)), dtype=np.float64)
        reference = np.asarray([_reference_log_h(float(z)) for z in self._GRID])
        relative = np.abs(values - reference) / np.maximum(1.0, np.abs(reference))
        assert np.all(np.isfinite(values))
        assert float(np.max(relative)) <= 1e-6, float(np.max(relative))

    def test_gradient_matches_the_mills_ratio(self) -> None:
        """``d/dz log h(z) = Phi(z) / h(z)``, compared in log space for very negative ``z``."""
        gradients = np.asarray(
            jax.vmap(jax.grad(lambda z: self._log_h(z[None])[0]))(jnp.asarray(self._GRID)),
            dtype=np.float64,
        )
        reference = np.asarray(
            [math.exp(norm.logcdf(float(z)) - _reference_log_h(float(z))) for z in self._GRID]
        )
        assert np.all(np.isfinite(gradients))
        relative = np.abs(gradients - reference) / np.abs(reference)
        z = self._GRID.astype(np.float64)
        in_erfcx_branch = (z <= -1.0) & (z > -1e3)
        # The float64 reference subtracts log h from log Phi, two numbers near -z^2/2, so its own
        # relative error grows as eps64 z^2 (2e-4 at |z| = 1e6, measured 1.6e-4 there).
        reference_error = 2.0 * float(np.finfo(np.float64).eps) * z * z
        tolerance = np.where(in_erfcx_branch, 2.0 * _EPS32 * z * z, 0.0) + 1e-5 + reference_error
        assert np.all(relative <= tolerance), float(np.max(relative / tolerance))

    @pytest.mark.parametrize("boundary", [-1.0, -1e3])
    def test_is_continuous_across_branch_boundaries(self, boundary: float) -> None:
        """Adjacent float32 inputs on either side of a branch boundary give adjacent values."""
        below = np.nextafter(np.float32(boundary), np.float32(-np.inf))
        above = np.nextafter(np.float32(boundary), np.float32(np.inf))
        values = np.asarray(self._log_h(jnp.asarray([below, above])), dtype=np.float64)
        slope_bound = abs(boundary) + 2.0
        tolerance = slope_bound * float(above - below) + 8.0 * _EPS32 * max(1.0, abs(values[0]))
        assert abs(values[1] - values[0]) <= tolerance, (values, tolerance)

    def test_jit_and_vmap_match_eager(self) -> None:
        grid = jnp.asarray(self._GRID)
        eager = self._log_h(grid)
        compiled = jax.jit(jax.vmap(lambda z: self._log_h(z[None])[0]))(grid)
        scale = jnp.maximum(1.0, jnp.abs(eager))
        assert bool(jnp.all(jnp.abs(compiled - eager) <= 4.0 * _EPS32 * scale))


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

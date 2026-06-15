"""Tests for the reliability utilities (Task 6.5).

References:
* Wilson, E. B. (1927), "Probable inference, the law of succession,
  and statistical inference", JASA 22(158), 209-212 — the canonical
  binomial-proportion confidence interval used here.
* Hasofer, A. M. & Lind, N. C. (1974), "Exact and invariant
  second-moment code format", J. Eng. Mech. 100(1), 111-121 — the
  Cornell / Hasofer-Lind reliability index ``beta = -Phi^{-1}(p_f)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.reliability import (
    failure_probability,
    reliability_index,
    ReliabilityResult,
)


def test_failure_probability_recovers_empirical_rate() -> None:
    """Plan exit criterion 1: failure probability from boolean indicators."""
    indicators = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    result = failure_probability(indicators)
    assert isinstance(result, ReliabilityResult)
    assert jnp.allclose(result.probability, 0.5)
    assert result.num_samples == 10
    assert result.num_failures == 5


def test_wilson_ci_is_finite_and_contains_empirical_probability() -> None:
    """Plan exit criterion 2: binomial CI is finite + contains the estimate."""
    rng_key = jax.random.PRNGKey(0)
    indicators = (jax.random.uniform(rng_key, (256,)) < 0.3).astype(jnp.float32)
    result = failure_probability(indicators, confidence_level=0.95)
    lower, upper = result.confidence_interval
    assert jnp.isfinite(lower) and jnp.isfinite(upper)
    assert lower <= result.probability <= upper
    assert 0.0 <= lower <= upper <= 1.0


def test_reliability_index_decreases_with_failure_probability() -> None:
    """Plan exit criterion 3: ``beta`` is monotone decreasing in ``p_f``."""
    p_low = jnp.array(0.01)
    p_mid = jnp.array(0.10)
    p_high = jnp.array(0.40)
    beta_low = reliability_index(p_low)
    beta_mid = reliability_index(p_mid)
    beta_high = reliability_index(p_high)
    assert beta_low > beta_mid > beta_high


def test_reliability_index_matches_standard_normal_inverse() -> None:
    """``beta = -Phi^{-1}(p_f)`` for the canonical reference values."""
    # Standard reference: p_f = 0.5 → beta = 0; p_f ≈ 0.0228 → beta ≈ 2.
    assert jnp.allclose(reliability_index(jnp.array(0.5)), 0.0, atol=1e-6)
    assert jnp.allclose(reliability_index(jnp.array(0.022750132)), 2.0, atol=1e-4)


def test_failure_probability_empty_sample_raises() -> None:
    """Plan exit criterion 4: empty samples raise ``ValueError``."""
    with pytest.raises(ValueError, match="empty"):
        failure_probability(jnp.array([]))


def test_failure_probability_invalid_confidence_raises() -> None:
    """Plan exit criterion 4: invalid confidence level raises ``ValueError``."""
    indicators = jnp.array([1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="confidence_level"):
        failure_probability(indicators, confidence_level=1.5)


def test_failure_probability_jit_compatible() -> None:
    """JAX-transform compatibility — Task 6.5 exit criterion."""
    indicators = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])

    def prob_only(ind: jax.Array) -> jax.Array:
        return failure_probability(ind).probability

    jit_result = jax.jit(prob_only)(indicators)
    eager_result = prob_only(indicators)
    assert jnp.allclose(jit_result, eager_result)


def test_reliability_index_grad_compatible() -> None:
    """``reliability_index`` flows gradients (used in inverse-design)."""
    grad_beta = jax.grad(reliability_index)(jnp.array(0.05))
    # d/d p_f (-Phi^{-1}(p_f)) = -1 / phi(Phi^{-1}(p_f)) < 0 always.
    assert grad_beta < 0
    assert jnp.isfinite(grad_beta)

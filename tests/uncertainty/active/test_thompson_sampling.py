r"""Tests for ``active/thompson_sampling.py`` — Slice 22.

Continuous Thompson sampling for Bayesian optimisation (Russo+ 2018
*Tutorial on Thompson Sampling*, FnT). For each acquisition round:

1. Draw one posterior-function realisation ``f̃ ~ q(f)``.
2. Return ``argmin_x f̃(x)`` over the candidate set (or, equivalently,
   ``argmax`` under a maximisation convention).

The opifex port consumes the existing
:class:`opifex.uncertainty.types.PredictiveDistribution`'s
``samples`` field (one Monte-Carlo draw per candidate) and returns
the argmin index.

References
----------
* Russo+ 2018 — *A Tutorial on Thompson Sampling*, FnT.
* trieste ``acquisition/function/continuous_thompson_sampling.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.types import PredictiveDistribution


def _predictive_with_samples(seed: int = 0, num_candidates: int = 8) -> PredictiveDistribution:
    """Build a small PredictiveDistribution with a single sample per candidate."""
    key = jax.random.PRNGKey(seed)
    mean = jnp.linspace(-1.0, 1.0, num_candidates)
    variance = jnp.full_like(mean, 0.2**2)
    samples = mean[None, :] + 0.2 * jax.random.normal(key, (1, num_candidates))
    return PredictiveDistribution(mean=mean, variance=variance, samples=samples)


def test_continuous_thompson_sampling_returns_argmin_of_posterior_draw() -> None:
    """Single sample / single round → argmin index over the candidate set."""
    from opifex.uncertainty.active.thompson_sampling import continuous_thompson_sampling

    predictive = _predictive_with_samples(seed=0, num_candidates=6)
    selected = continuous_thompson_sampling(predictive=predictive)
    # Sample[0] is shape (6,); argmin returns a scalar int in [0, 6).
    assert selected.dtype in (jnp.int32, jnp.int64)
    assert int(selected) >= 0
    assert int(selected) < 6
    # The returned index must coincide with the global argmin of sample 0.
    expected = int(jnp.argmin(predictive.samples[0]))  # type: ignore[index]
    assert int(selected) == expected


def test_continuous_thompson_sampling_supports_maximisation_via_negation() -> None:
    """Negating ``mean``+``samples`` flips Thompson sampling to argmax."""
    from opifex.uncertainty.active.thompson_sampling import continuous_thompson_sampling

    predictive = _predictive_with_samples(seed=1, num_candidates=5)
    negated_samples = -predictive.samples  # type: ignore[operator]
    negated = PredictiveDistribution(
        mean=-predictive.mean,
        variance=predictive.variance,
        samples=negated_samples,
    )
    argmin_of_negated = int(continuous_thompson_sampling(predictive=negated))
    argmax_of_original = int(jnp.argmax(predictive.samples[0]))  # type: ignore[index]
    assert argmin_of_negated == argmax_of_original


def test_continuous_thompson_sampling_raises_when_samples_missing() -> None:
    """Without a posterior sample the routine has nothing to argmin."""
    from opifex.uncertainty.active.thompson_sampling import continuous_thompson_sampling

    predictive = PredictiveDistribution(mean=jnp.zeros(4), variance=jnp.ones(4), samples=None)
    import pytest

    with pytest.raises(ValueError, match="samples"):
        continuous_thompson_sampling(predictive=predictive)


def test_continuous_thompson_sampling_is_jit_compatible() -> None:
    """The selection routine compiles under ``jax.jit``."""
    from opifex.uncertainty.active.thompson_sampling import continuous_thompson_sampling

    predictive = _predictive_with_samples(seed=2)

    @jax.jit
    def select(samples: jax.Array) -> jax.Array:
        pd = PredictiveDistribution(
            mean=predictive.mean, variance=predictive.variance, samples=samples
        )
        return continuous_thompson_sampling(predictive=pd)

    selected = select(predictive.samples)
    assert int(selected) >= 0

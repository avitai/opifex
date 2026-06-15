"""Tests for the classical-UQ sampling helpers (Task 6.6)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.scientific.sampling import (
    halton_sequence,
    latin_hypercube_sample,
    random_sample,
)


def test_random_sample_respects_bounds_and_shape() -> None:
    """Plan exit criterion: validates bounds, dim, sample count."""
    lower = jnp.array([0.0, -1.0])
    upper = jnp.array([1.0, 2.0])
    samples = random_sample(
        num_samples=128, lower=lower, upper=upper, rng_key=jax.random.PRNGKey(0)
    )
    assert samples.shape == (128, 2)
    assert jnp.all(samples >= lower)
    assert jnp.all(samples <= upper)


def test_random_sample_requires_caller_owned_key() -> None:
    """Plan exit criterion: no hidden fixed seeds — key is a required kwarg."""
    with pytest.raises(TypeError):
        # rng_key is keyword-only and required; omitting it must fail.
        random_sample(num_samples=8, lower=jnp.zeros(2), upper=jnp.ones(2))  # type: ignore[call-arg]


def test_latin_hypercube_stratifies_each_dimension() -> None:
    """LHS strata are exactly equally populated by construction."""
    samples = latin_hypercube_sample(
        num_samples=10,
        lower=jnp.zeros(3),
        upper=jnp.ones(3),
        rng_key=jax.random.PRNGKey(1),
    )
    assert samples.shape == (10, 3)
    # Each dimension's stratum index must contain every value 0..9 once.
    for j in range(3):
        strata = jnp.floor(samples[:, j] * 10).astype(jnp.int32)
        assert jnp.array_equal(jnp.sort(strata), jnp.arange(10))


def test_halton_sequence_is_deterministic_and_in_bounds() -> None:
    """Halton points reproduce exactly across calls and respect bounds."""
    lower = jnp.zeros(2)
    upper = jnp.ones(2)
    seq_a = halton_sequence(num_samples=64, lower=lower, upper=upper, skip=10)
    seq_b = halton_sequence(num_samples=64, lower=lower, upper=upper, skip=10)
    assert jnp.array_equal(seq_a, seq_b)
    assert jnp.all(seq_a >= lower)
    assert jnp.all(seq_a <= upper)


def test_sampling_helpers_raise_on_invalid_shapes() -> None:
    """Plan exit criterion: invalid sample shapes raise ``ValueError``."""
    with pytest.raises(ValueError, match="num_samples"):
        random_sample(
            num_samples=0,
            lower=jnp.zeros(2),
            upper=jnp.ones(2),
            rng_key=jax.random.PRNGKey(0),
        )
    with pytest.raises(ValueError, match=r"lower.*upper"):
        random_sample(
            num_samples=8,
            lower=jnp.zeros(2),
            upper=jnp.ones(3),
            rng_key=jax.random.PRNGKey(0),
        )
    with pytest.raises(ValueError, match="num_samples"):
        latin_hypercube_sample(
            num_samples=-1,
            lower=jnp.zeros(2),
            upper=jnp.ones(2),
            rng_key=jax.random.PRNGKey(0),
        )
    with pytest.raises(ValueError, match="skip"):
        halton_sequence(
            num_samples=8,
            lower=jnp.zeros(2),
            upper=jnp.ones(2),
            skip=-1,
        )

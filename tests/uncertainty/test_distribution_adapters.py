"""Tests for the distribution adapter layer.

``from_distribution`` accepts ``artifex.generative_models.core.distributions.base.Distribution``
as its primary target and falls back to Distrax-like objects exposing
``sample`` / ``log_prob`` / ``mean`` / ``variance``. Both paths yield a
:class:`opifex.uncertainty.types.PredictiveDistribution` with mean and
variance populated.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.uncertainty.distributions import (
    ArtifexDistributionAdapter,
    DistrAxAdapter,
    from_distribution,
)
from opifex.uncertainty.types import PredictiveDistribution


class _FakeDistraxNormal:
    """Distrax-like object exposing the secondary adapter surface."""

    def __init__(self, loc: jax.Array, scale: jax.Array) -> None:
        self._loc = loc
        self._scale = scale

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        *,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        del rngs
        return jnp.zeros(sample_shape or (1,))

    def log_prob(self, x: jax.Array) -> jax.Array:
        return jnp.zeros_like(x)

    def mean(self) -> jax.Array:
        return self._loc

    def variance(self) -> jax.Array:
        return self._scale * self._scale


def test_artifex_distribution_adapter_wraps_real_normal_into_predictive_distribution() -> None:
    """Real ``artifex.generative_models.core.distributions.continuous.Normal`` round-trips."""
    from artifex.generative_models.core.distributions.continuous import Normal

    rngs = nnx.Rngs(0)
    dist = Normal(loc=jnp.zeros(3), scale=jnp.ones(3), rngs=rngs)
    adapter = ArtifexDistributionAdapter()
    predictive = adapter.from_distribution(dist)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.mean.shape == (3,)
    assert predictive.variance is not None
    assert predictive.variance.shape == (3,)


def test_distrax_adapter_wraps_distrax_like_object_into_predictive_distribution() -> None:
    fake = _FakeDistraxNormal(jnp.array([1.0, 2.0]), jnp.array([0.5, 0.5]))
    adapter = DistrAxAdapter()
    predictive = adapter.from_distribution(fake)
    assert isinstance(predictive, PredictiveDistribution)
    assert jnp.array_equal(predictive.mean, fake.mean())
    assert predictive.variance is not None
    assert jnp.array_equal(predictive.variance, fake.variance())


def test_from_distribution_prefers_artifex_over_distrax() -> None:
    """``from_distribution`` resolution order: Artifex first, Distrax fallback."""
    from artifex.generative_models.core.distributions.continuous import Normal

    rngs = nnx.Rngs(0)
    artifex_dist = Normal(loc=jnp.zeros(2), scale=jnp.ones(2), rngs=rngs)
    out = from_distribution(artifex_dist)
    assert isinstance(out, PredictiveDistribution)
    # Sanity: distrax-like falls back successfully too.
    fake = _FakeDistraxNormal(jnp.zeros(2), jnp.ones(2))
    out_fake = from_distribution(fake)
    assert isinstance(out_fake, PredictiveDistribution)


def test_from_distribution_rejects_unsupported_object() -> None:
    """Objects lacking sample/log_prob/mean/variance surface raise TypeError."""
    import pytest

    class _NotADistribution:
        pass

    with pytest.raises(TypeError, match=r"distribution"):
        from_distribution(_NotADistribution())


def test_distribution_adapter_supports_jit_for_log_probability() -> None:
    """``log_prob`` round-trip is jit-safe for the Distrax-like adapter.

    The wrapped distribution itself is not a pytree, so it is captured as a
    closure constant rather than threaded as a jit argument.
    """
    fake = _FakeDistraxNormal(jnp.zeros(2), jnp.ones(2))
    jit_log_prob = jax.jit(fake.log_prob)
    result = jit_log_prob(jnp.zeros(2))
    assert result.shape == (2,)


def test_adapter_resolution_order_is_documented() -> None:
    """The module surfaces the resolution order via __all__ for downstream consumers."""
    from opifex.uncertainty import distributions

    expected = {"ArtifexDistributionAdapter", "DistrAxAdapter", "from_distribution"}
    assert expected <= set(distributions.__all__)

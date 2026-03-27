"""Tests for variational inference framework.

Tests the core variational components that don't require distrax.
The MeanFieldGaussian.log_prob and kl_divergence methods require
distrax and are tested separately in the Bayesian FNO example.
"""

import dataclasses

import jax.numpy as jnp
from flax import nnx

from opifex.neural.bayesian.variational_framework import (
    MeanFieldGaussian,
    PriorConfig,
    UncertaintyEncoder,
    VariationalConfig,
)


class TestPriorConfig:
    """Tests for PriorConfig dataclass."""

    def test_defaults(self):
        """Default config has empty constraints."""
        config = PriorConfig()
        assert config.conservation_laws == ()
        assert config.prior_scale == 1.0

    def test_custom_laws(self):
        """Custom conservation laws can be set."""
        config = PriorConfig(conservation_laws=("energy", "momentum"))
        assert len(config.conservation_laws) == 2

    def test_is_dataclass(self):
        """PriorConfig is a dataclass."""
        assert dataclasses.is_dataclass(PriorConfig)


class TestVariationalConfig:
    """Tests for VariationalConfig dataclass."""

    def test_required_input_dim(self):
        """input_dim is required."""
        config = VariationalConfig(input_dim=16)
        assert config.input_dim == 16

    def test_defaults(self):
        """Default hidden dims and sampling config."""
        config = VariationalConfig(input_dim=8)
        assert config.hidden_dims == (64, 32)
        assert config.num_samples == 10
        assert config.kl_weight == 1.0
        assert config.temperature == 1.0


class TestMeanFieldGaussian:
    """Tests for MeanFieldGaussian variational posterior."""

    def test_init_creates_params(self):
        """Initializes mean and log_std variational parameters."""
        mfg = MeanFieldGaussian(num_params=8, rngs=nnx.Rngs(0))
        assert mfg.mean.value.shape == (8,)
        assert mfg.log_std.value.shape == (8,)

    def test_mean_starts_at_zero(self):
        """Mean is initialized to zeros."""
        mfg = MeanFieldGaussian(num_params=4, rngs=nnx.Rngs(0))
        assert jnp.allclose(mfg.mean.value, jnp.zeros(4))

    def test_log_std_starts_negative(self):
        """Log std is initialized to -2.0 (small variance)."""
        mfg = MeanFieldGaussian(num_params=4, rngs=nnx.Rngs(0))
        assert jnp.allclose(mfg.log_std.value, jnp.full(4, -2.0))

    def test_sample_shape(self):
        """Samples have correct shape (num_samples, num_params)."""
        mfg = MeanFieldGaussian(num_params=8, rngs=nnx.Rngs(0))
        samples = mfg.sample(5, rngs=nnx.Rngs(1))
        assert samples.shape == (5, 8)

    def test_samples_near_mean(self):
        """With small log_std (-2), samples should be close to mean."""
        mfg = MeanFieldGaussian(num_params=4, rngs=nnx.Rngs(0))
        samples = mfg.sample(100, rngs=nnx.Rngs(1))
        sample_mean = jnp.mean(samples, axis=0)
        # With log_std = -2.0, std = exp(-2) ≈ 0.135
        assert jnp.allclose(sample_mean, mfg.mean.value, atol=0.1)

    def test_different_rngs_give_different_samples(self):
        """Different RNG keys produce different samples."""
        mfg = MeanFieldGaussian(num_params=4, rngs=nnx.Rngs(0))
        s1 = mfg.sample(3, rngs=nnx.Rngs(1))
        s2 = mfg.sample(3, rngs=nnx.Rngs(2))
        assert not jnp.allclose(s1, s2)

    def test_is_nnx_module(self):
        """MeanFieldGaussian is an nnx.Module."""
        mfg = MeanFieldGaussian(num_params=4, rngs=nnx.Rngs(0))
        assert isinstance(mfg, nnx.Module)


class TestUncertaintyEncoder:
    """Tests for UncertaintyEncoder."""

    def test_init(self):
        """Encoder initializes with correct dimensions."""
        encoder = UncertaintyEncoder(
            input_dim=16,
            hidden_dims=(32, 16),
            output_dim=8,
            rngs=nnx.Rngs(0),
        )
        assert isinstance(encoder, nnx.Module)

    def test_forward_shape(self):
        """Forward pass produces correct output shape."""
        encoder = UncertaintyEncoder(
            input_dim=16,
            hidden_dims=(32, 16),
            output_dim=8,
            rngs=nnx.Rngs(0),
        )
        x = jnp.ones((4, 16))
        out = encoder(x)
        # Output should have mean and log_std for each output dim
        assert out.shape[0] == 4

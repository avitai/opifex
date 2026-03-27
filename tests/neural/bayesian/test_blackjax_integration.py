"""Tests for BlackJAX MCMC integration.

Tests the integration layer between Flax NNX models and BlackJAX
samplers for Bayesian inference.
"""

import pytest
from flax import nnx

from opifex.neural.bayesian.blackjax_integration import BlackJAXIntegration


class SimpleModel(nnx.Module):
    """Minimal model for testing BlackJAX integration."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.linear = nnx.Linear(4, 2, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


class TestBlackJAXIntegrationInit:
    """Tests for BlackJAXIntegration initialization."""

    def test_init_with_nuts(self):
        """Initializes with NUTS sampler."""
        model = SimpleModel(rngs=nnx.Rngs(0))
        bjx = BlackJAXIntegration(model, sampler_type="nuts", rngs=nnx.Rngs(1))
        assert bjx.sampler_type == "nuts"
        assert bjx.num_params > 0

    def test_init_with_hmc(self):
        """Initializes with HMC sampler."""
        model = SimpleModel(rngs=nnx.Rngs(0))
        bjx = BlackJAXIntegration(model, sampler_type="hmc", rngs=nnx.Rngs(1))
        assert bjx.sampler_type == "hmc"

    def test_counts_parameters(self):
        """Correctly counts model parameters."""
        model = SimpleModel(rngs=nnx.Rngs(0))
        bjx = BlackJAXIntegration(model, rngs=nnx.Rngs(1))
        # Linear(4, 2) has 4*2 weights + 2 bias = 10 params
        assert bjx.num_params == 10

    def test_is_nnx_module(self):
        """Integration wrapper is an nnx.Module."""
        model = SimpleModel(rngs=nnx.Rngs(0))
        bjx = BlackJAXIntegration(model, rngs=nnx.Rngs(1))
        assert isinstance(bjx, nnx.Module)

    def test_default_sampling_config(self):
        """Default config has standard warmup and sample counts."""
        model = SimpleModel(rngs=nnx.Rngs(0))
        bjx = BlackJAXIntegration(model, rngs=nnx.Rngs(1))
        assert bjx.num_warmup == 1000
        assert bjx.num_samples == 1000
        assert bjx.step_size == pytest.approx(1e-3)

    def test_custom_sampling_config(self):
        """Custom warmup, samples, and step size are respected."""
        model = SimpleModel(rngs=nnx.Rngs(0))
        bjx = BlackJAXIntegration(
            model,
            num_warmup=100,
            num_samples=500,
            step_size=0.01,
            rngs=nnx.Rngs(1),
        )
        assert bjx.num_warmup == 100
        assert bjx.num_samples == 500
        assert bjx.step_size == pytest.approx(0.01)

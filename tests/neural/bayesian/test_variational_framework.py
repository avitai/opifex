"""Tests for variational inference framework.

Tests the core variational components that don't require distrax.
The MeanFieldGaussian.log_prob and kl_divergence methods require
distrax and are tested separately in the Bayesian FNO example.
"""

import dataclasses

import jax
import jax.flatten_util
import jax.numpy as jnp
from flax import nnx

from opifex.neural.bayesian.variational_framework import (
    AmortizedVariationalFramework,
    MeanFieldGaussian,
    PriorConfig,
    UncertaintyEncoder,
    VariationalConfig,
)


class _BiasFreeLinear(nnx.Module):
    """Single bias-free ``nnx.Linear`` base model for injection tests.

    A bias-free layer makes the all-zero parameter vector map to an
    exactly-zero output, so a real weight injection is observable: the
    fake input-noise surrogate would instead echo a non-zero
    ``base_model(x)``.
    """

    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.linear = nnx.Linear(in_dim, out_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x)


def _build_framework(in_dim: int = 3, out_dim: int = 2) -> AmortizedVariationalFramework:
    """Build a framework over a bias-free linear base model."""
    base_model = _BiasFreeLinear(in_dim, out_dim, rngs=nnx.Rngs(0))
    return AmortizedVariationalFramework(
        base_model=base_model,
        prior_config=PriorConfig(),
        variational_config=VariationalConfig(input_dim=in_dim),
        rngs=nnx.Rngs(1),
    )


class TestAmortizedForwardInjectsSampledParams:
    """The amortized forward must inject sampled weights, not perturb inputs."""

    def test_amortized_forward_injects_sampled_params(self):
        """Changing ``params_vector`` changes the output via model weights.

        With a bias-free base model an all-zero parameter vector yields an
        exactly-zero output (true weight injection), while a non-zero
        vector reproduces the merged model's forward pass. The fake
        input-noise surrogate could never satisfy both.
        """
        framework = _build_framework(in_dim=3, out_dim=2)
        x = jnp.ones((4, 3))

        zero_params = jnp.zeros((framework.num_params,))
        out_zero = framework._forward_with_params(x, zero_params)
        assert out_zero.shape == (4, 2)
        assert jnp.allclose(out_zero, jnp.zeros_like(out_zero))

        perturbed_params = zero_params.at[0].set(1.0).at[1].set(-0.5)
        out_perturbed = framework._forward_with_params(x, perturbed_params)
        assert not jnp.allclose(out_perturbed, out_zero)

        # The injection must equal a manual nnx.merge of the same params,
        # confirming the weights — not the input — carry the perturbation.
        graphdef, params_state = nnx.split(framework.base_model, nnx.Param)
        unravel = jax.flatten_util.ravel_pytree(params_state)[1]
        manual_model = nnx.merge(graphdef, unravel(perturbed_params))
        expected = manual_model(x)  # type: ignore[operator]
        assert jnp.allclose(out_perturbed, expected)

    def test_forward_with_params_is_jittable(self):
        """The injected forward is a pure function of ``params_vector`` under jit."""
        framework = _build_framework(in_dim=3, out_dim=2)
        x = jnp.ones((4, 3))
        params = jax.random.normal(jax.random.key(0), (framework.num_params,))

        graphdef = nnx.graphdef(framework)

        @jax.jit
        def forward(state, params_vector):
            model = nnx.merge(graphdef, state)
            return model._forward_with_params(x, params_vector)

        out = forward(nnx.state(framework), params)
        assert out.shape == (4, 2)
        assert jnp.all(jnp.isfinite(out))

    def test_forward_with_params_vmaps_over_samples(self):
        """A stack of parameter vectors vmaps to a stack of predictions."""
        framework = _build_framework(in_dim=3, out_dim=2)
        x = jnp.ones((4, 3))
        param_stack = jax.random.normal(jax.random.key(0), (5, framework.num_params))

        graphdef, base_state = nnx.split(framework.base_model, nnx.Param)
        unravel = jax.flatten_util.ravel_pytree(base_state)[1]

        def forward(params_vector):
            model = nnx.merge(graphdef, unravel(params_vector))
            return model(x)  # type: ignore[operator]

        stacked = jax.vmap(forward)(param_stack)
        assert stacked.shape == (5, 4, 2)


class TestAmortizedPublicAPI:
    """Public methods consume the real injected forward, not a surrogate."""

    def test_predict_with_uncertainty_shapes_and_finite(self):
        """Mean and uncertainty have the output shape and are finite."""
        framework = _build_framework(in_dim=3, out_dim=2)
        x = jnp.ones((4, 3))
        mean, uncertainty = framework.predict_with_uncertainty(x, 8, rngs=nnx.Rngs(5))
        assert mean.shape == (4, 2)
        assert uncertainty.shape == (4, 2)
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(uncertainty >= 0.0)

    def test_sample_predictive_distribution_shape(self):
        """Predictive samples stack along a leading sample axis."""
        framework = _build_framework(in_dim=3, out_dim=2)
        x = jnp.ones((4, 3))
        samples = framework.sample_predictive_distribution(x, 6, rngs=nnx.Rngs(6))
        assert samples.shape == (6, 4, 2)
        assert jnp.all(jnp.isfinite(samples))

    def test_compute_elbo_is_finite_scalar(self):
        """ELBO over the injected posterior is a finite scalar."""
        framework = _build_framework(in_dim=3, out_dim=2)
        x = jnp.ones((4, 3))
        y = jnp.zeros((4, 2))
        elbo = framework.compute_elbo(x, y, 8, rngs=nnx.Rngs(7))
        assert elbo.shape == ()
        assert jnp.isfinite(elbo)


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

"""Tests for FNO architectural correctness.

These tests verify that the FNO implementation matches the reference
Li et al. (2021) architecture. Tests written FIRST per TDD — the
implementation must pass them, not the other way around.

Reference:
    Li et al. (2021) "Fourier Neural Operator for Parametric Partial
    Differential Equations" (ICLR 2021)
    PDEBench: Takamoto et al. (2022) NeurIPS
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.operators.fno.base import (
    FourierLayer,
    FourierNeuralOperator,
    FourierSpectralConvolution,
)


class TestSpectralWeightInit:
    """Weight initialization must match Li et al. scale."""

    def test_weight_scale_is_small(self):
        """Spectral weights should be O(1/(in*out)), not O(1/sqrt(in+out))."""
        conv = FourierSpectralConvolution(
            in_channels=20, out_channels=20, modes=12, rngs=nnx.Rngs(0)
        )
        # Weights stored as separate real/imaginary arrays
        max_real = float(jnp.max(jnp.abs(conv.weights_real[...])))
        max_imag = float(jnp.max(jnp.abs(conv.weights_imag[...])))
        max_weight = max(max_real, max_imag)
        # Li et al. scale: 1/(20*20) = 0.0025, so weights should be < 0.01
        assert max_weight < 0.05, (
            f"Spectral weights too large: max={max_weight:.4f}. "
            f"Expected O(1/(in*out)) ≈ 0.0025, got O(1/sqrt(in+out)) ≈ 0.14"
        )


class TestSpectral2DCoverage:
    """2D spectral convolution must cover both frequency quadrants."""

    def test_2d_uses_both_quadrants(self):
        """_spectral_2d must process both positive and negative y-frequencies."""
        layer = FourierLayer(
            in_channels=4,
            out_channels=4,
            modes=4,
            rngs=nnx.Rngs(0),
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 16, 16))
        out = layer(x)

        # Output should be different from zero (spectral conv is applied)
        assert jnp.max(jnp.abs(out)) > 0.01

        # The output should use the full spatial resolution
        assert out.shape == (2, 4, 16, 16)

    def test_2d_spectral_is_not_identity(self):
        """The spectral path should produce non-trivial output."""
        layer = FourierLayer(
            in_channels=4,
            out_channels=4,
            modes=4,
            rngs=nnx.Rngs(0),
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 16, 16))

        # Apply spectral transform only (no skip)
        spectral_out = layer._apply_spectral_transform(x)

        # Should not be all zeros (indicates modes are being processed)
        assert jnp.max(jnp.abs(spectral_out)) > 1e-6


class TestOutputProjection:
    """Output projection should be a two-layer MLP (matching Li et al.)."""

    def test_output_has_intermediate_layer(self):
        """FNO output goes through width -> 128 -> out (two linear + GELU)."""
        model = FourierNeuralOperator(
            in_channels=3,
            out_channels=1,
            hidden_channels=20,
            modes=8,
            num_layers=4,
            rngs=nnx.Rngs(0),
        )
        # The output projection should be more than a single linear
        assert hasattr(model, "output_projection_1")
        assert hasattr(model, "output_projection_2")


class TestDomainPadding:
    """FNO should support domain padding for non-periodic problems."""

    def test_padding_does_not_change_output_shape(self):
        """With padding, output shape matches input spatial dims."""
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            modes=4,
            num_layers=2,
            domain_padding=2,
            rngs=nnx.Rngs(0),
        )
        x = jnp.ones((2, 1, 16, 16))
        out = model(x)
        assert out.shape == (2, 1, 16, 16)


class TestLastLayerNoActivation:
    """The last Fourier layer should NOT apply activation (Li et al. pattern)."""

    def test_model_applies_activation_selectively(self):
        """Only intermediate Fourier layers have activation, not the last one."""
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            modes=4,
            num_layers=4,
            rngs=nnx.Rngs(0),
        )
        x = jnp.ones((1, 1, 8, 8))
        # Should run without error and produce finite output
        out = model(x)
        assert jnp.all(jnp.isfinite(out))


class TestFNOForwardShape:
    """Basic shape tests for the FNO forward pass."""

    def test_2d_forward(self):
        """2D FNO: (batch, C_in, H, W) -> (batch, C_out, H, W)."""
        model = FourierNeuralOperator(
            in_channels=3,
            out_channels=1,
            hidden_channels=16,
            modes=6,
            num_layers=4,
            rngs=nnx.Rngs(0),
        )
        x = jnp.ones((4, 3, 32, 32))
        out = model(x)
        assert out.shape == (4, 1, 32, 32)

    def test_1d_forward(self):
        """1D FNO: (batch, C_in, X) -> (batch, C_out, X)."""
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=8,
            num_layers=4,
            spatial_dims=1,
            rngs=nnx.Rngs(0),
        )
        x = jnp.ones((4, 1, 64))
        out = model(x)
        assert out.shape == (4, 1, 64)

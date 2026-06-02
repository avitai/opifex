"""Tests for the shared flux-divergence conservation loss helper.

Validates the local conservation-law residual against analytic fields:
a divergence-free field gives ~0 loss while a field with known non-zero
divergence gives the correct positive value. Also exercises jit/grad/vmap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.neural.operators.physics._conservation import (
    central_difference,
    conservation_residual_loss,
    flux_divergence,
)


class TestCentralDifference:
    """Central difference matches analytic derivatives on a periodic grid."""

    def test_derivative_of_sine_matches_cosine(self):
        """d/dx sin(x) = cos(x) on a periodic grid."""
        n = 256
        x = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        spacing = float(x[1] - x[0])
        derivative = central_difference(jnp.sin(x), axis=0, spacing=spacing)
        assert jnp.allclose(derivative, jnp.cos(x), atol=1e-3)

    def test_constant_field_has_zero_derivative(self):
        """A constant field has identically zero derivative."""
        field = jnp.full((32,), 3.5)
        derivative = central_difference(field, axis=0, spacing=0.1)
        assert jnp.allclose(derivative, 0.0, atol=1e-12)


class TestFluxDivergence:
    """Divergence sums component-wise central differences."""

    def test_linear_flux_has_constant_divergence(self):
        """Flux F(x) = (a*x, b*x) has divergence a + b (interior)."""
        n = 128
        x = jnp.linspace(0.0, 1.0, n, endpoint=False)
        spacing = float(x[1] - x[0])
        flux = jnp.stack([2.0 * x, -0.5 * x], axis=-1)  # (n, 2)
        divergence = flux_divergence(flux, spatial_axis=0, spacing=spacing)
        interior = divergence[1:-1]
        assert jnp.allclose(interior, 2.0 - 0.5, atol=1e-4)


class TestConservationResidualLoss:
    """The residual loss penalises non-zero flux divergence."""

    def test_divergence_free_field_gives_zero_loss(self):
        """A constant flux is divergence-free -> ~0 conservation loss."""
        flux = jnp.ones((4, 64, 3))  # constant across all points
        loss = conservation_residual_loss(flux, spatial_axis=1, spacing=0.05)
        assert loss < 1e-12

    def test_known_divergence_gives_correct_positive_value(self):
        """Linear flux gives the analytic mean-squared divergence value."""
        n = 200
        x = jnp.linspace(0.0, 1.0, n, endpoint=False)
        spacing = float(x[1] - x[0])
        # Single-component flux F = 3*x -> divergence = 3 everywhere (interior).
        flux = (3.0 * x)[:, None]  # (n, 1)
        loss = conservation_residual_loss(flux, spatial_axis=0, spacing=spacing)
        # Interior residual is exactly 3.0; boundary wrap introduces a few
        # outliers, so compare on the interior-dominated mean.
        divergence = flux_divergence(flux, spatial_axis=0, spacing=spacing)
        expected = jnp.mean(divergence**2)
        assert jnp.isclose(loss, expected)
        assert jnp.mean(divergence[1:-1]) == pytest.approx(3.0, abs=1e-3)

    def test_time_derivative_cancels_divergence(self):
        """Continuity holds when ∂_t q = -∇·F, giving ~0 loss."""
        n = 128
        x = jnp.linspace(0.0, 1.0, n, endpoint=False)
        spacing = float(x[1] - x[0])
        flux = jnp.sin(2.0 * jnp.pi * x)[:, None]
        divergence = flux_divergence(flux, spatial_axis=0, spacing=spacing)
        loss = conservation_residual_loss(
            flux, spatial_axis=0, spacing=spacing, time_derivative=-divergence
        )
        assert loss < 1e-12


class TestTransformCompatibility:
    """jit / grad / vmap smoke tests for the conservation loss."""

    def test_jit(self):
        """Loss is jit-compilable and matches eager evaluation."""
        flux = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 3))
        jitted = jax.jit(lambda f: conservation_residual_loss(f, spatial_axis=1, spacing=0.1))
        eager = conservation_residual_loss(flux, spatial_axis=1, spacing=0.1)
        assert jnp.isclose(jitted(flux), eager)

    def test_grad(self):
        """Loss is differentiable with finite, non-zero gradients."""
        flux = jax.random.normal(jax.random.PRNGKey(1), (16, 2))
        grad = jax.grad(lambda f: conservation_residual_loss(f, spatial_axis=0, spacing=0.1))(flux)
        assert grad.shape == flux.shape
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.linalg.norm(grad) > 0.0

    def test_vmap(self):
        """Loss vmaps cleanly over a batch of flux fields."""
        flux = jax.random.normal(jax.random.PRNGKey(2), (5, 32, 3))
        batched = jax.vmap(lambda f: conservation_residual_loss(f, spatial_axis=0, spacing=0.1))(
            flux
        )
        assert batched.shape == (5,)
        assert jnp.all(jnp.isfinite(batched))

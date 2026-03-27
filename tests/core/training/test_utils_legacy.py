"""Tests for legacy training utility functions."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.training.utils_legacy import (
    safe_compute_energy,
    safe_compute_forces,
    safe_model_call,
)


class SimpleModel(nnx.Module):
    """Minimal model for testing safe call utilities."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


class EnergyModel(nnx.Module):
    """Model with compute_energy for testing."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.linear = nnx.Linear(6, 1, rngs=rngs)

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        return self.linear(x)

    def compute_energy(self, positions: jax.Array, **kwargs) -> jax.Array:
        """Compute scalar energy from positions."""
        flat = positions.reshape(positions.shape[0], -1) if positions.ndim > 2 else positions
        return self.linear(flat).squeeze(-1)


class TestSafeModelCall:
    """Tests for safe_model_call."""

    def test_calls_standard_model(self):
        """Standard nnx.Module is called successfully."""
        model = SimpleModel(4, 2, rngs=nnx.Rngs(0))
        x = jnp.ones((1, 4))
        result = safe_model_call(model, x)
        assert result.shape == (1, 2)

    def test_raises_for_non_callable(self):
        """Non-callable object raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            safe_model_call("not_a_model", jnp.ones((1, 4)))


class TestSafeComputeEnergy:
    """Tests for safe_compute_energy."""

    def test_uses_compute_energy_method(self):
        """Calls compute_energy when available."""
        model = EnergyModel(rngs=nnx.Rngs(0))
        positions = jnp.ones((2, 3, 2))  # batch of 2, 3 atoms, 2D
        energy = safe_compute_energy(model, positions)
        assert energy.shape == (2,)

    def test_falls_back_to_call(self):
        """Falls back to __call__ when no compute_energy."""
        model = SimpleModel(6, 1, rngs=nnx.Rngs(0))
        positions = jnp.ones((2, 3, 2))  # will be flattened to (2, 6)
        energy = safe_compute_energy(model, positions)
        assert energy.ndim >= 1


class TestSafeComputeForces:
    """Tests for safe_compute_forces."""

    def test_uses_compute_forces_method(self):
        """Uses compute_forces when available on model."""

        class ForcesModel:
            def compute_forces(self, positions, **kwargs):
                return -positions  # Simple repulsive force

        model = ForcesModel()
        positions = jnp.ones((2, 3))
        forces = safe_compute_forces(model, positions)
        assert forces.shape == (2, 3)
        assert jnp.allclose(forces, -jnp.ones((2, 3)))

    def test_gradient_fallback_2d(self):
        """Computes forces via gradient for 2D positions."""
        model = EnergyModel(rngs=nnx.Rngs(0))
        positions = jnp.ones((1, 6))
        forces = safe_compute_forces(model, positions)
        assert forces.shape == (1, 6)

    def test_rejects_1d_positions(self):
        """Raises for unsupported 1D positions."""
        model = EnergyModel(rngs=nnx.Rngs(0))
        with pytest.raises(ValueError, match="Unsupported positions shape"):
            safe_compute_forces(model, jnp.ones((6,)))

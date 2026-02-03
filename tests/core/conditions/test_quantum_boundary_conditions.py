"""
Tests for Quantum Boundary Conditions

Tests for WavefunctionBC and QuantumInitialCondition.
"""

import jax.numpy as jnp
import pytest

from opifex.core.conditions import (
    QuantumInitialCondition,
    WavefunctionBC,
)


class TestWavefunctionBC:
    """Test quantum mechanical wavefunction boundary conditions."""

    def test_initialization_vanishing(self):
        """Test wavefunction BC with vanishing condition."""
        bc = WavefunctionBC(condition_type="vanishing", boundary="all")

        assert bc.condition_type == "vanishing"
        assert bc.boundary == "all"
        assert bc.value is None

    def test_initialization_normalization(self):
        """Test wavefunction BC with normalization condition."""
        bc = WavefunctionBC(condition_type="normalization", norm_value=1.0)

        assert bc.condition_type == "normalization"
        assert bc.norm_value == 1.0

    def test_initialization_periodic(self):
        """Test wavefunction BC with periodic condition."""
        bc = WavefunctionBC(condition_type="periodic", boundary="all")

        assert bc.condition_type == "periodic"
        assert bc.boundary == "all"

    def test_invalid_condition_type(self):
        """Test initialization with invalid condition type."""
        with pytest.raises(ValueError, match="Invalid condition_type"):
            WavefunctionBC(condition_type="invalid")

    def test_validate_vanishing(self):
        """Test validation for vanishing boundary condition."""
        bc = WavefunctionBC(condition_type="vanishing")
        assert bc.validate() is True

    def test_validate_normalization_valid(self):
        """Test validation for valid normalization condition."""
        bc = WavefunctionBC(condition_type="normalization", norm_value=1.0)
        assert bc.validate() is True

    def test_validate_normalization_invalid(self):
        """Test validation for invalid normalization condition."""
        bc = WavefunctionBC(condition_type="normalization", norm_value=-1.0)
        assert bc.validate() is False

    def test_validate_periodic(self):
        """Test validation for periodic boundary condition."""
        bc = WavefunctionBC(condition_type="periodic", value=1.0)
        assert bc.validate() is True

    def test_evaluate_vanishing(self):
        """Test evaluation for vanishing boundary condition."""
        bc = WavefunctionBC(condition_type="vanishing")
        x = jnp.array([1.0, 2.0, 3.0])

        result = bc.evaluate(x)
        expected = jnp.zeros_like(x)
        assert jnp.allclose(result, expected)

    def test_evaluate_normalization(self):
        """Test evaluation for normalization condition."""
        bc = WavefunctionBC(condition_type="normalization", norm_value=2.0)
        x = jnp.array([1.0, 2.0, 3.0])

        result = bc.evaluate(x)
        expected = jnp.full_like(x, 2.0)
        assert jnp.allclose(result, expected)

    def test_evaluate_periodic(self):
        """Test evaluation for periodic boundary condition."""
        bc = WavefunctionBC(condition_type="periodic", value=1.0 + 0.5j)
        x = jnp.array([1.0, 2.0, 3.0])

        result = bc.evaluate(x)
        expected = jnp.full_like(x, 1.0 + 0.5j) + 0j
        assert jnp.allclose(result, expected)


class TestQuantumInitialCondition:
    """Test quantum initial conditions."""

    def test_initialization_ground_state(self):
        """Test quantum initial condition for ground state."""

        def ground_state(x):
            return jnp.exp(-(x**2))

        qic = QuantumInitialCondition(
            condition_type="ground_state",
            value=ground_state,
            normalization=1.0,
            n_electrons=2,
        )

        assert qic.condition_type == "ground_state"
        assert qic.value == ground_state
        assert qic.normalization == 1.0
        assert qic.n_electrons == 2

    def test_initialization_excited_state(self):
        """Test quantum initial condition for excited state."""

        def excited_state(x):
            return x * jnp.exp(-(x**2) / 2)

        qic = QuantumInitialCondition(
            condition_type="excited_state", value=excited_state
        )

        assert qic.condition_type == "excited_state"
        assert qic.value == excited_state

    def test_invalid_condition_type(self):
        """Test initialization with invalid condition type."""

        def wf(x):
            return jnp.exp(-(x**2))

        with pytest.raises(ValueError, match="Invalid condition_type"):
            QuantumInitialCondition(condition_type="invalid", value=wf)

    def test_validate_ground_state(self):
        """Test validation for ground state condition."""

        def ground_state(x):
            return jnp.exp(-(x**2))

        qic = QuantumInitialCondition(condition_type="ground_state", value=ground_state)
        assert qic.validate() is True

    def test_validate_custom_wavefunction(self):
        """Test validation for custom wavefunction."""

        def custom_wf(x):
            return jnp.sin(x)

        qic = QuantumInitialCondition(condition_type="custom", value=custom_wf)
        assert qic.validate() is True

    def test_validate_negative_normalization(self):
        """Test validation with negative normalization."""

        def wf(x):
            return jnp.exp(-(x**2))

        qic = QuantumInitialCondition(
            condition_type="ground_state", value=wf, normalization=-1.0
        )
        assert qic.validate() is False

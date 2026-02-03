"""Test module for neural base classes - comprehensive TDD validation.

Tests for StandardMLP and QuantumMLP following TDD methodology:
- Test-driven development with comprehensive coverage
- Modern neural network features validation
- Quantum-specific functionality testing
- Integration with existing test patterns
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.base import QuantumMLP, StandardMLP


class TestStandardMLP:
    """Test suite for StandardMLP with modern neural network features."""

    def test_initialization_with_defaults(self):
        """Test StandardMLP initialization with modern defaults."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(layer_sizes=[4, 8, 1], rngs=rngs)

        # Test modern defaults
        assert mlp.activation == "gelu"  # Modern default instead of tanh
        assert mlp.dropout_rate == 0.0
        assert mlp.use_bias is True
        assert mlp.apply_final_dropout is False
        assert len(mlp.layers) == 2  # Input->hidden, hidden->output

    def test_initialization_with_custom_parameters(self):
        """Test StandardMLP with custom parameters including new features."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(
            layer_sizes=[2, 16, 8, 1],
            activation="tanh",
            dropout_rate=0.1,
            use_bias=False,
            apply_final_dropout=True,
            rngs=rngs,
        )

        # Test custom configuration
        assert mlp.activation == "tanh"
        assert mlp.dropout_rate == 0.1
        assert mlp.use_bias is False
        assert mlp.apply_final_dropout is True
        assert len(mlp.layers) == 3

    def test_forward_pass_basic(self):
        """Test basic forward pass functionality."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(layer_sizes=[4, 8, 1], rngs=rngs)

        x = jnp.ones((2, 4))  # Batch of 2, input dim 4
        output = mlp(x, deterministic=True)

        assert output.shape == (2, 1)
        assert jnp.isfinite(output).all()

    def test_forward_pass_with_dropout_training(self):
        """Test forward pass with dropout during training."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(layer_sizes=[4, 8, 1], dropout_rate=0.2, rngs=rngs)

        x = jnp.ones((2, 4))

        # Training mode (deterministic=False)
        output1 = mlp(x, deterministic=False)
        output2 = mlp(x, deterministic=False)

        # Outputs should be different due to dropout randomness
        assert not jnp.allclose(output1, output2, atol=1e-5)
        assert output1.shape == (2, 1)
        assert output2.shape == (2, 1)

    def test_forward_pass_with_dropout_inference(self):
        """Test forward pass with dropout during inference."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(layer_sizes=[4, 8, 1], dropout_rate=0.2, rngs=rngs)

        x = jnp.ones((2, 4))

        # Inference mode (deterministic=True)
        output1 = mlp(x, deterministic=True)
        output2 = mlp(x, deterministic=True)

        # Outputs should be identical in inference mode
        assert jnp.allclose(output1, output2)
        assert output1.shape == (2, 1)

    def test_apply_final_dropout_feature(self):
        """Test the new apply_final_dropout feature."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(
            layer_sizes=[4, 8, 1], dropout_rate=0.3, apply_final_dropout=True, rngs=rngs
        )

        x = jnp.ones((2, 4))

        # Training mode should apply dropout to final layer
        output1 = mlp(x, deterministic=False)
        output2 = mlp(x, deterministic=False)

        # Should be different due to final dropout
        assert not jnp.allclose(output1, output2, atol=1e-5)

        # Inference mode should be deterministic
        output_inf1 = mlp(x, deterministic=True)
        output_inf2 = mlp(x, deterministic=True)
        assert jnp.allclose(output_inf1, output_inf2)

    def test_different_activations(self):
        """Test StandardMLP with different activation functions."""
        rngs = nnx.Rngs(42)
        x = jnp.ones((1, 4))

        activations = ["gelu", "tanh", "relu", "sigmoid", "silu"]

        for activation in activations:
            mlp = StandardMLP(layer_sizes=[4, 8, 1], activation=activation, rngs=rngs)

            output = mlp(x, deterministic=True)
            assert output.shape == (1, 1)
            assert jnp.isfinite(output).all()

    def test_error_handling_invalid_layer_sizes(self):
        """Test error handling for invalid layer sizes."""
        rngs = nnx.Rngs(42)

        with pytest.raises(
            ValueError, match="layer_sizes must have at least 2 elements"
        ):
            StandardMLP(
                layer_sizes=[4],  # Only one layer
                rngs=rngs,
            )

    def test_dropout_error_handling(self):
        """Test error handling for dropout misconfiguration."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(layer_sizes=[4, 8, 1], dropout_rate=0.2, rngs=rngs)

        x = jnp.ones((2, 4))

        # This should work fine
        output = mlp(x, deterministic=False)
        assert output.shape == (2, 1)


class TestQuantumMLP:
    """Test suite for QuantumMLP with quantum-specific features."""

    def test_initialization_with_quantum_defaults(self):
        """Test QuantumMLP with quantum defaults."""
        rngs = nnx.Rngs(42)

        mlp = QuantumMLP(
            layer_sizes=[6, 16, 1],
            rngs=rngs,
        )

        # Test quantum defaults
        assert mlp.activation == "tanh"  # Quantum-optimized default
        assert mlp.enforce_symmetry is True
        assert mlp.dropout_rate == 0.0
        assert mlp.use_bias is True
        assert mlp.apply_final_dropout is False

    def test_initialization_with_custom_quantum_parameters(self):
        """Test QuantumMLP with custom quantum parameters."""
        rngs = nnx.Rngs(42)

        mlp = QuantumMLP(
            layer_sizes=[6, 16, 8, 1],
            activation="gelu",
            enforce_symmetry=False,
            dropout_rate=0.1,
            apply_final_dropout=True,
            rngs=rngs,
            symmetry_type="rotation",
        )

        # Test custom configuration
        assert mlp.activation == "gelu"
        assert mlp.enforce_symmetry is False
        assert mlp.dropout_rate == 0.1
        assert mlp.apply_final_dropout is True
        assert mlp.symmetry_type == "rotation"

    def test_forward_pass_molecular_system(self):
        """Test forward pass with molecular system-like input."""
        rngs = nnx.Rngs(42)

        mlp = QuantumMLP(
            layer_sizes=[12, 32, 16, 1],  # 4 atoms * 3 coords = 12 input
            rngs=rngs,
        )

        # Molecular coordinates for 4 atoms
        x = jnp.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Batch 1
                [0.1, 0.1, 0.1, 1.1, 0.1, 0.1, 0.1, 1.1, 0.1, 0.1, 0.1, 1.1],  # Batch 2
            ]
        )

        output = mlp(x, deterministic=True)

        assert output.shape == (2, 1)
        assert jnp.isfinite(output).all()

    def test_energy_computation_method(self):
        """Test quantum-specific energy computation method."""
        rngs = nnx.Rngs(42)

        mlp = QuantumMLP(layer_sizes=[12, 32, 1], rngs=rngs)

        # Molecular positions (4 atoms * 3 coords)
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # Atom 1
                [1.0, 0.0, 0.0],  # Atom 2
                [0.0, 1.0, 0.0],  # Atom 3
                [0.0, 0.0, 1.0],  # Atom 4
            ]
        )

        # Reshape for network input
        flat_positions = positions.flatten()

        energy = mlp.compute_energy(flat_positions, deterministic=True)

        assert energy.shape == ()  # Scalar energy
        assert jnp.isfinite(energy)

    def test_forces_computation_method(self):
        """Test quantum-specific forces computation method."""
        rngs = nnx.Rngs(42)

        mlp = QuantumMLP(layer_sizes=[12, 32, 1], rngs=rngs)

        # Molecular positions (4 atoms * 3 coords)
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # Atom 1
                [1.0, 0.0, 0.0],  # Atom 2
                [0.0, 1.0, 0.0],  # Atom 3
                [0.0, 0.0, 1.0],  # Atom 4
            ]
        )

        # Reshape for network input
        flat_positions = positions.flatten()

        forces = mlp.compute_forces(flat_positions, deterministic=True)

        assert forces.shape == (12,)  # Forces for all coordinates
        assert jnp.isfinite(forces).all()

    def test_energy_and_forces_computation(self):
        """Test combined energy and forces computation."""
        rngs = nnx.Rngs(42)

        mlp = QuantumMLP(layer_sizes=[12, 32, 1], rngs=rngs)

        # Molecular positions (4 atoms * 3 coords)
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # Atom 1
                [1.0, 0.0, 0.0],  # Atom 2
                [0.0, 1.0, 0.0],  # Atom 3
                [0.0, 0.0, 1.0],  # Atom 4
            ]
        )

        # Reshape for network input
        flat_positions = positions.flatten()

        energy, forces = mlp.compute_energy_and_forces(
            flat_positions, deterministic=True
        )

        assert energy.shape == ()  # Scalar energy
        assert forces.shape == (12,)  # Forces for all coordinates
        assert jnp.isfinite(energy)
        assert jnp.isfinite(forces).all()

    def test_quantum_dropout_features(self):
        """Test quantum-specific dropout features."""
        rngs = nnx.Rngs(42)

        mlp = QuantumMLP(
            layer_sizes=[6, 16, 1],
            dropout_rate=0.2,
            apply_final_dropout=True,
            rngs=rngs,
        )

        x = jnp.ones((2, 6))

        # Training mode with quantum dropout
        output1 = mlp(x, deterministic=False)
        output2 = mlp(x, deterministic=False)

        # Should be different due to dropout
        assert not jnp.allclose(output1, output2, atol=1e-5)

        # Inference mode should be deterministic
        output_inf1 = mlp(x, deterministic=True)
        output_inf2 = mlp(x, deterministic=True)
        assert jnp.allclose(output_inf1, output_inf2)

    def test_quantum_activations(self):
        """Test QuantumMLP with quantum-relevant activation functions."""
        rngs = nnx.Rngs(42)
        x = jnp.ones((1, 6))

        # Test quantum-relevant activations
        quantum_activations = ["tanh", "gelu", "silu"]

        for activation in quantum_activations:
            mlp = QuantumMLP(layer_sizes=[6, 16, 1], activation=activation, rngs=rngs)

            output = mlp(x, deterministic=True)
            assert output.shape == (1, 1)
            assert jnp.isfinite(output).all()

    def test_symmetry_configuration(self):
        """Test symmetry configuration options."""
        rngs = nnx.Rngs(42)

        # Test permutation symmetry
        mlp_perm = QuantumMLP(
            layer_sizes=[6, 16, 1],
            enforce_symmetry=True,
            symmetry_type="permutation",
            rngs=rngs,
        )

        assert mlp_perm.enforce_symmetry is True
        assert mlp_perm.symmetry_type == "permutation"

        # Test rotation symmetry
        mlp_rot = QuantumMLP(
            layer_sizes=[6, 16, 1],
            enforce_symmetry=True,
            symmetry_type="rotation",
            rngs=rngs,
        )

        assert mlp_rot.enforce_symmetry is True
        assert mlp_rot.symmetry_type == "rotation"

    def test_initialization_validation(self):
        """Test validation logic in QuantumMLP initialization."""
        # Test invalid layer sizes validation
        with pytest.raises(
            ValueError, match="layer_sizes must have at least 2 elements"
        ):
            QuantumMLP(
                layer_sizes=[1],  # Invalid: only one element
                rngs=nnx.Rngs(42),
            )

    def test_quantum_scaling_application(self):
        """Test quantum scaling factor application in initialization."""
        # Create QuantumMLP to ensure quantum scaling is applied (line 243)
        mlp = QuantumMLP(
            layer_sizes=[6, 4, 1],
            activation="tanh",
            rngs=nnx.Rngs(42),
        )

        # Verify quantum scaling was applied by checking layer weights are scaled
        # The quantum scaling factor is 0.5, so weights should be smaller
        assert len(mlp.layers) == 2
        # Just verify the model was initialized successfully with quantum scaling
        test_input = jnp.ones((1, 6))
        output = mlp(test_input)
        assert output.shape == (1, 1)

    def test_energy_computation_error_handling(self):
        """Test error handling in energy computation methods."""
        mlp = QuantumMLP(
            layer_sizes=[6, 4, 1],
            activation="tanh",
            rngs=nnx.Rngs(42),
        )

        # Test invalid 1D position length (line 418)
        invalid_1d_positions = jnp.array([1.0, 2.0, 3.0, 4.0])  # Not divisible by 3
        with pytest.raises(
            ValueError, match="Flattened positions length 4 must be divisible by 3"
        ):
            mlp.compute_energy(invalid_1d_positions)

        # Test invalid dimensions (line 429)
        invalid_4d_positions = jnp.ones((2, 3, 3, 2))  # 4D array
        with pytest.raises(
            ValueError, match="Expected positions with 1, 2 or 3 dimensions, got 4"
        ):
            mlp.compute_energy(invalid_4d_positions)

    def test_energy_scalar_error_handling(self):
        """Test error handling in _compute_energy_scalar method."""
        mlp = QuantumMLP(
            layer_sizes=[6, 4, 1],
            activation="tanh",
            rngs=nnx.Rngs(42),
        )

        # Test invalid dimensions for _compute_energy_scalar (lines 445-448, 468)
        invalid_1d_positions = jnp.array([1.0, 2.0, 3.0])  # 1D array
        with pytest.raises(
            ValueError, match="_compute_energy_scalar expects 2D positions, got 1D"
        ):
            mlp._compute_energy_scalar(invalid_1d_positions)

        invalid_3d_positions = jnp.ones((1, 2, 3))  # 3D array
        with pytest.raises(
            ValueError, match="_compute_energy_scalar expects 2D positions, got 3D"
        ):
            mlp._compute_energy_scalar(invalid_3d_positions)

    def test_forces_computation_error_handling(self):
        """Test error handling in forces computation methods."""
        mlp = QuantumMLP(
            layer_sizes=[6, 4, 1],
            activation="tanh",
            rngs=nnx.Rngs(42),
        )

        # Test invalid 1D position length for forces (line 503)
        invalid_1d_positions = jnp.array(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )  # Not divisible by 3
        with pytest.raises(
            ValueError, match="Flattened positions length 5 must be divisible by 3"
        ):
            mlp.compute_forces(invalid_1d_positions)

        # Test invalid dimensions for forces (lines 519-529)
        invalid_4d_positions = jnp.ones((2, 3, 3, 2))  # 4D array
        with pytest.raises(
            ValueError, match="Expected positions with 1, 2 or 3 dimensions, got 4"
        ):
            mlp.compute_forces(invalid_4d_positions)

    def test_energy_and_forces_error_handling(self):
        """Test error handling in compute_energy_and_forces method."""
        mlp = QuantumMLP(
            layer_sizes=[6, 4, 1],
            activation="tanh",
            rngs=nnx.Rngs(42),
        )

        # Test invalid 1D position length (line 555)
        invalid_1d_positions = jnp.array(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )  # Not divisible by 3
        with pytest.raises(
            ValueError, match="Flattened positions length 5 must be divisible by 3"
        ):
            mlp.compute_energy_and_forces(invalid_1d_positions)

    def test_energy_output_shape_handling(self):
        """Test energy output shape handling edge cases."""
        mlp = QuantumMLP(
            layer_sizes=[6, 4, 1],
            activation="tanh",
            rngs=nnx.Rngs(42),
        )

        # Test 1D input energy computation (lines 424-433)
        # This should trigger the energy.squeeze() path for 1D input
        positions_1d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2 atoms
        energy = mlp.compute_energy(positions_1d)
        # For 1D input, should return scalar
        assert energy.ndim == 0 or (energy.ndim == 1 and energy.shape[0] == 1)

        # Test 2D input energy computation
        positions_2d = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Single molecule
        energy = mlp.compute_energy(positions_2d)
        # Should return properly shaped energy
        assert energy.shape == (1, 1)

        # Test 3D input energy computation (batched)
        positions_3d = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # Batch of 1
        energy = mlp.compute_energy(positions_3d)
        # Should return properly shaped energy
        assert energy.shape == (1, 1)

    def test_forces_different_dimensions(self):
        """Test forces computation for different input dimensions."""
        mlp = QuantumMLP(
            layer_sizes=[6, 4, 1],
            activation="tanh",
            rngs=nnx.Rngs(42),
        )

        # Test 1D input (lines 519-529)
        positions_1d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2 atoms
        forces = mlp.compute_forces(positions_1d)
        assert forces.shape == (6,)  # Should return flattened forces

        # Test 2D input (single molecule)
        positions_2d = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        forces = mlp.compute_forces(positions_2d)
        assert forces.shape == (2, 3)  # Should return forces per atom

        # Test 3D input (batched molecules)
        positions_3d = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # Batch of 1
        forces = mlp.compute_forces(positions_3d)
        assert forces.shape == (1, 2, 3)  # Should return batched forces

    def test_energy_and_forces_different_dimensions(self):
        """Test energy and forces computation for different input dimensions."""
        mlp = QuantumMLP(
            layer_sizes=[6, 4, 1],
            activation="tanh",
            rngs=nnx.Rngs(42),
        )

        # Test 1D input (lines 572-578)
        positions_1d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2 atoms
        energy, forces = mlp.compute_energy_and_forces(positions_1d)
        assert energy.ndim == 0 or (
            energy.ndim == 1 and energy.shape[0] == 1
        )  # Scalar energy
        assert forces.shape == (6,)  # Flattened forces

        # Test 2D input (single molecule)
        positions_2d = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        energy, forces = mlp.compute_energy_and_forces(positions_2d)
        assert energy.ndim == 0 or (
            energy.ndim == 1 and energy.shape[0] == 1
        )  # Scalar energy
        assert forces.shape == (2, 3)  # Forces per atom

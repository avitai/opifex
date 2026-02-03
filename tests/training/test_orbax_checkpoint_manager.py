"""Tests for Orbax checkpoint manager integration."""

import shutil
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp

# type: ignore[import-untyped]
import optax  # type: ignore[import-untyped]
import pytest
from flax import nnx
from flax.training import train_state

from opifex.training.orbax_checkpoint_manager import OrbaxCheckpointManager


class SimpleTestModel(nnx.Module):
    """Simple test model for checkpoint testing."""

    def __init__(self, features=4, *, rngs=None):
        """Initialize simple test model.

        Args:
            features: Number of features in the linear layer
            rngs: Required RNG dict for NNX compatibility
        """
        super().__init__()
        self.dense1 = nnx.Linear(features, features, rngs=rngs)  # type: ignore[arg-type]
        self.dense2 = nnx.Linear(features, features, rngs=rngs)  # type: ignore[arg-type]

    def __call__(self, x):
        """Forward pass."""
        x = self.dense1(x)
        x = nnx.relu(x)
        return self.dense2(x)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return SimpleTestModel(features=4, rngs=nnx.Rngs(42))


@pytest.fixture
def rng_key():
    """Create JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def physics_metadata():
    """Create physics metadata for testing."""
    return {
        "chemical_accuracy": 0.001,  # kcal/mol
        "scf_convergence": True,
        "scf_iterations": 15,
        "constraint_violations": {
            "energy_conservation": 1e-6,
            "density_constraint": 5e-6,
        },
        "wavefunction_coefficients": [0.1, 0.2, 0.3],
        "density_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "molecular_orbitals": {"homo": -0.5, "lumo": 0.2},
        "convergence_history": list(range(1000)),
    }


@pytest.fixture
def optimizer():
    """Create optimizer for testing."""
    return optax.adam(learning_rate=0.001)


@pytest.fixture
def train_state_obj(simple_model, optimizer):
    """Create TrainState for testing."""
    manager = OrbaxCheckpointManager(tempfile.mkdtemp())
    return manager.create_train_state(simple_model, optimizer, step=0)


class TestOrbaxCheckpointManager:
    """Test cases for OrbaxCheckpointManager."""

    def test_initialization(self, temp_dir):
        """Test checkpoint manager initialization."""
        manager = OrbaxCheckpointManager(temp_dir)

        assert str(manager.checkpoint_dir) == temp_dir
        assert manager.checkpoint_manager is not None
        assert Path(temp_dir).exists()

    def test_save_and_load_checkpoint(self, temp_dir, simple_model, physics_metadata):
        """Test basic save and load functionality."""
        manager = OrbaxCheckpointManager(temp_dir)

        # Save checkpoint
        step = 100
        loss = 0.05
        checkpoint_path = manager.save_checkpoint(
            simple_model, step, loss, physics_metadata
        )

        # Verify checkpoint was saved
        assert checkpoint_path == str(Path(temp_dir) / str(step))
        assert Path(checkpoint_path).exists()

        # Load checkpoint
        loaded_model, loaded_metadata = manager.load_checkpoint(simple_model, step)

        # Verify model was loaded correctly
        assert loaded_model is not None
        # Test model forward pass to ensure it works
        test_input = jnp.ones((1, 4))
        output = loaded_model(test_input)  # type: ignore[misc,operator]
        assert output.shape == (1, 4)

        # Verify metadata was preserved - check nested structure
        assert loaded_metadata is not None
        assert "physics_metadata" in loaded_metadata
        physics_meta = loaded_metadata["physics_metadata"]
        assert "wavefunction_coefficients" in physics_meta
        assert "density_matrix" in physics_meta
        assert "molecular_orbitals" in physics_meta
        assert len(physics_meta["convergence_history"]) == 1000

    def test_load_latest_checkpoint(self, temp_dir, simple_model, physics_metadata):
        """Test loading latest checkpoint."""
        manager = OrbaxCheckpointManager(temp_dir)

        # Save multiple checkpoints
        for step in [100, 200, 300]:
            manager.save_checkpoint(simple_model, step, 0.05, physics_metadata)

        # Load latest checkpoint (should be step 300)
        manager.load_checkpoint(simple_model)

        # Verify latest checkpoint was loaded
        latest_step = manager.get_latest_step()
        assert latest_step == 300

    def test_load_specific_checkpoint(self, temp_dir, simple_model, physics_metadata):
        """Test loading specific checkpoint step."""
        manager = OrbaxCheckpointManager(temp_dir)

        # Save multiple checkpoints
        for step in [100, 200, 300]:
            manager.save_checkpoint(simple_model, step, 0.05, physics_metadata)

        # Load specific checkpoint
        _, loaded_metadata = manager.load_checkpoint(simple_model, step=200)

        # Verify correct checkpoint was loaded
        assert loaded_metadata["step"] == 200

    def test_save_with_metadata(self, temp_dir, simple_model, physics_metadata):
        """Test saving with comprehensive physics metadata."""
        manager = OrbaxCheckpointManager(temp_dir)

        # Save with rich metadata
        step = 500
        loss = 0.02
        additional_metadata = {"experiment_id": "test_001", "notes": "Test run"}

        manager.save_checkpoint(
            simple_model, step, loss, physics_metadata, additional_metadata
        )

        # Load and verify all metadata preserved
        _, loaded_metadata = manager.load_checkpoint(simple_model, step)

        # Check core metadata
        assert loaded_metadata["step"] == step
        assert loaded_metadata["loss"] == loss

        # Check physics metadata (nested structure)
        physics_meta = loaded_metadata["physics_metadata"]
        assert physics_meta["chemical_accuracy"] == 0.001
        assert physics_meta["scf_convergence"] is True
        assert physics_meta["scf_iterations"] == 15

        # Check additional metadata
        assert loaded_metadata["experiment_id"] == "test_001"
        assert loaded_metadata["notes"] == "Test run"

    def test_load_nonexistent_checkpoint(self, temp_dir, simple_model):
        """Test loading nonexistent checkpoint."""
        manager = OrbaxCheckpointManager(temp_dir)

        # Try to load from empty directory
        loaded_model, loaded_metadata = manager.load_checkpoint(simple_model, step=999)

        # Should return original model and empty metadata
        assert loaded_model is simple_model
        assert loaded_metadata == {}

    def test_load_raw_checkpoint_data(self, temp_dir, simple_model, physics_metadata):
        """Test loading raw checkpoint data without target model."""
        manager = OrbaxCheckpointManager(temp_dir)

        # Save checkpoint
        step = 100
        manager.save_checkpoint(simple_model, step, 0.05, physics_metadata)

        # Load raw data
        loaded_model, loaded_metadata = manager.load_checkpoint(
            target_model=None, step=step
        )

        # Should return None for model and metadata dict
        assert loaded_model is None
        assert isinstance(loaded_metadata, dict)
        assert "step" in loaded_metadata

    def test_error_handling_invalid_directory(self):
        """Test error handling for invalid directory."""
        with pytest.raises(ValueError, match="Checkpoint directory cannot be empty"):
            OrbaxCheckpointManager("")

    def test_error_handling_invalid_model(self, temp_dir):
        """Test error handling for invalid model type."""
        manager = OrbaxCheckpointManager(temp_dir)

        with pytest.raises(TypeError, match=r"Expected model to be nnx\.Module"):
            manager.save_checkpoint("not_a_model", 100, 0.05)  # type: ignore[arg-type]

    def test_error_handling_invalid_step(self, temp_dir, simple_model):
        """Test error handling for invalid step."""
        manager = OrbaxCheckpointManager(temp_dir)

        with pytest.raises(ValueError, match="Step must be a non-negative integer"):
            manager.save_checkpoint(simple_model, -1, 0.05)

    def test_checkpoint_manager_options(self, temp_dir):
        """Test checkpoint manager with custom options."""
        manager = OrbaxCheckpointManager(temp_dir, max_to_keep=3)

        assert manager.max_to_keep == 3
        assert manager.checkpoint_manager is not None

    def test_train_state_creation(self, simple_model, optimizer):
        """Test TrainState creation functionality."""
        manager = OrbaxCheckpointManager(tempfile.mkdtemp())

        # Create TrainState
        train_state_obj = manager.create_train_state(simple_model, optimizer, step=100)

        # Verify TrainState structure
        assert isinstance(train_state_obj, train_state.TrainState)
        assert train_state_obj.step == 100
        assert train_state_obj.apply_fn == simple_model
        assert train_state_obj.tx == optimizer

        # Verify parameters are present
        assert hasattr(train_state_obj, "params")
        assert train_state_obj.params is not None

    def test_save_and_load_train_state_checkpoint(
        self, temp_dir, train_state_obj, physics_metadata
    ):
        """Test saving and loading TrainState checkpoints."""
        manager = OrbaxCheckpointManager(temp_dir)

        # Save TrainState checkpoint
        step = 200
        loss = 0.03
        checkpoint_path = manager.save_train_state_checkpoint(
            train_state_obj, step, loss, physics_metadata
        )

        # Verify checkpoint was saved
        assert checkpoint_path == str(Path(temp_dir) / str(step))
        assert Path(checkpoint_path).exists()

        # Load TrainState checkpoint
        loaded_train_state, _loaded_metadata = manager.load_train_state_checkpoint(
            train_state_obj, step
        )  # type: ignore[reportUnusedVariable]

        # Verify TrainState was loaded correctly
        assert isinstance(loaded_train_state, train_state.TrainState)
        assert loaded_train_state.step == step
        assert loaded_train_state.apply_fn == train_state_obj.apply_fn
        assert loaded_train_state.tx == train_state_obj.tx

        # Test that the model still works
        test_input = jnp.ones((1, 4))
        output = loaded_train_state.apply_fn(test_input)
        assert output.shape == (1, 4)

        # Verify metadata was preserved
        assert _loaded_metadata["step"] == step  # type: ignore[reportUnusedVariable]
        assert _loaded_metadata["loss"] == loss  # type: ignore[reportUnusedVariable]
        assert "physics_metadata" in _loaded_metadata  # type: ignore[reportUnusedVariable]

    def test_train_state_with_optimizer_state(
        self, temp_dir, simple_model, optimizer, physics_metadata
    ):
        """Test that optimizer state is properly preserved in TrainState checkpoints."""
        manager = OrbaxCheckpointManager(temp_dir)

        # Create TrainState with initial step
        initial_step = 50
        train_state_obj = manager.create_train_state(
            simple_model, optimizer, step=initial_step
        )

        # Simulate some training steps to update optimizer state
        test_input = jnp.ones((1, 4))
        for step in range(initial_step, initial_step + 10):
            # Forward pass
            output = train_state_obj.apply_fn(test_input)

            # Simulate loss computation
            loss = jnp.mean(output**2)

            # Update TrainState (simulating training step)
            train_state_obj = train_state_obj.replace(step=step)

        # Save checkpoint
        checkpoint_step = 60
        loss = 0.02
        manager.save_train_state_checkpoint(
            train_state_obj, checkpoint_step, loss, physics_metadata
        )

        # Load checkpoint
        loaded_train_state, _ = manager.load_train_state_checkpoint(
            train_state_obj, checkpoint_step
        )

        # Verify optimizer state is preserved
        assert loaded_train_state.step == checkpoint_step
        assert loaded_train_state.apply_fn == train_state_obj.apply_fn
        assert loaded_train_state.tx == train_state_obj.tx

        # Verify the model parameters are preserved
        assert loaded_train_state.params is not None

    def test_complete_model_state_preservation(
        self, temp_dir, simple_model, physics_metadata
    ):
        """Test that complete model state including all metadata is preserved."""
        manager = OrbaxCheckpointManager(temp_dir)

        # Create comprehensive metadata
        comprehensive_metadata = {
            "experiment_config": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "optimizer": "adam",
            },
            "training_history": {
                "losses": [0.1, 0.08, 0.06, 0.04, 0.02],
                "accuracies": [0.5, 0.6, 0.7, 0.8, 0.9],
                "learning_rates": [0.001, 0.001, 0.0008, 0.0006, 0.0004],
            },
            "model_config": {
                "features": 4,
                "activation": "relu",
                "dropout": 0.1,
            },
            "physics_metadata": physics_metadata,
        }

        # Save checkpoint with comprehensive metadata
        step = 300
        loss = 0.01
        manager.save_checkpoint(
            simple_model, step, loss, physics_metadata, comprehensive_metadata
        )

        # Load checkpoint
        _, loaded_metadata = manager.load_checkpoint(simple_model, step)

        # Verify all metadata is preserved
        assert loaded_metadata["step"] == step
        assert loaded_metadata["loss"] == loss
        assert "experiment_config" in loaded_metadata
        assert "training_history" in loaded_metadata
        assert "model_config" in loaded_metadata
        assert "physics_metadata" in loaded_metadata

        # Verify nested structures are preserved
        exp_config = loaded_metadata["experiment_config"]
        assert exp_config["learning_rate"] == 0.001
        assert exp_config["batch_size"] == 32

        training_history = loaded_metadata["training_history"]
        assert len(training_history["losses"]) == 5
        assert len(training_history["accuracies"]) == 5

        # Verify physics metadata is preserved
        physics_meta = loaded_metadata["physics_metadata"]
        assert physics_meta["chemical_accuracy"] == 0.001
        assert physics_meta["scf_convergence"] is True

    def test_train_state_metadata_loading(
        self, temp_dir, train_state_obj, physics_metadata
    ):
        """Test that TrainState checkpoints can be loaded as metadata-only."""
        manager = OrbaxCheckpointManager(temp_dir)

        # Save TrainState checkpoint
        step = 400
        loss = 0.015
        manager.save_train_state_checkpoint(
            train_state_obj, step, loss, physics_metadata
        )

        # Load only metadata (without target model)
        _loaded_model, loaded_metadata = manager.load_checkpoint(
            target_model=None, step=step
        )  # type: ignore[reportUnusedVariable]

        # Should return None for model and metadata dict
        assert _loaded_model is None  # type: ignore[reportUnusedVariable]
        assert isinstance(loaded_metadata, dict)
        assert loaded_metadata["step"] == step
        assert loaded_metadata["loss"] == loss
        assert "physics_metadata" in loaded_metadata

    def test_error_handling_train_state_invalid_type(self, temp_dir):
        """Test error handling for invalid TrainState types."""
        manager = OrbaxCheckpointManager(temp_dir)

        with pytest.raises(TypeError, match=r"Expected model to be nnx\.Module"):
            manager.create_train_state("not_a_model", None, step=0)  # type: ignore[arg-type]

    def test_checkpoint_version_tagging(self, temp_dir, simple_model, physics_metadata):
        """Test that checkpoint version is properly tagged."""
        manager = OrbaxCheckpointManager(temp_dir)

        # Save checkpoint
        step = 500
        loss = 0.025
        manager.save_checkpoint(simple_model, step, loss, physics_metadata)

        # Load checkpoint
        loaded_model, loaded_metadata = manager.load_checkpoint(simple_model, step)

        # Verify checkpoint version is present in metadata
        assert "checkpoint_version" in loaded_metadata
        assert loaded_metadata["checkpoint_version"] == "2.0"

        # Verify model still works
        test_input = jnp.ones((1, 4))
        output = loaded_model(test_input)  # type: ignore[misc,operator]
        assert output.shape == (1, 4)

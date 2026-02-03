"""
Test suite for CheckpointManager class

This module contains comprehensive tests for the CheckpointManager class,
covering initialization, saving, loading, and management functionality.
"""

import tempfile
from pathlib import Path

import pytest
from flax import nnx

from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.training.checkpoint_manager import CheckpointManager


class TestCheckpointManagerInitialization:
    """Test CheckpointManager initialization functionality."""

    def test_checkpoint_manager_initialization(self):
        """Test basic CheckpointManager initialization."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test
            manager = CheckpointManager(temp_dir)

            # Verify
            assert manager.checkpoint_dir == Path(temp_dir)
            assert manager.max_checkpoints == 5
            assert manager.auto_cleanup is True

    def test_checkpoint_manager_creates_directory(self):
        """Test that CheckpointManager creates directory if it doesn't exist."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = Path(temp_dir) / "new_checkpoints"

            # Test
            manager = CheckpointManager(nonexistent_dir)

            # Verify
            assert manager.checkpoint_dir.exists()
            assert manager.checkpoint_dir.is_dir()

    def test_checkpoint_manager_invalid_directory(self):
        """Test CheckpointManager with invalid directory path."""
        # Test with a path that cannot be created (e.g., nested under a file)
        with tempfile.NamedTemporaryFile() as temp_file:
            invalid_path = Path(temp_file.name) / "subdir"

            # Test & Verify
            with pytest.raises(OSError, match="Not a directory"):
                CheckpointManager(invalid_path)


class TestCheckpointManagerSave:
    """Test checkpoint saving functionality."""

    def test_save_checkpoint_basic(self):
        """Test basic checkpoint saving functionality."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )

            # Test
            checkpoint_path = manager.save_checkpoint(model, step=100, loss=0.5)

            # Verify
            assert Path(checkpoint_path).exists()
            assert "checkpoint_step_100" in checkpoint_path
            assert checkpoint_path.endswith(".pkl")

    def test_save_checkpoint_with_metadata(self):
        """Test checkpoint saving with metadata."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )
            metadata = {"epoch": 10, "learning_rate": 0.001}

            # Test
            checkpoint_path = manager.save_checkpoint(
                model, step=100, loss=0.5, metadata=metadata
            )

            # Verify
            assert Path(checkpoint_path).exists()
            checkpoint_data = manager.load_checkpoint(checkpoint_path)
            assert checkpoint_data["metadata"]["epoch"] == 10
            assert checkpoint_data["metadata"]["learning_rate"] == 0.001

    def test_save_checkpoint_multiple_steps(self):
        """Test saving multiple checkpoints."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )

            # Test
            checkpoint_path_1 = manager.save_checkpoint(model, step=100, loss=0.5)
            checkpoint_path_2 = manager.save_checkpoint(model, step=200, loss=0.3)

            # Verify
            assert Path(checkpoint_path_1).exists()
            assert Path(checkpoint_path_2).exists()
            assert checkpoint_path_1 != checkpoint_path_2
            assert "checkpoint_step_100" in checkpoint_path_1
            assert "checkpoint_step_200" in checkpoint_path_2

    def test_save_checkpoint_overwrite_same_step(self):
        """Test saving checkpoint with same step number."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )

            # Test
            checkpoint_path_1 = manager.save_checkpoint(model, step=100, loss=0.5)
            import time

            time.sleep(1)  # Ensure different timestamps
            checkpoint_path_2 = manager.save_checkpoint(model, step=100, loss=0.3)

            # Verify
            assert Path(checkpoint_path_1).exists()
            assert Path(checkpoint_path_2).exists()
            assert checkpoint_path_1 != checkpoint_path_2  # Different timestamps

    def test_save_checkpoint_invalid_model(self):
        """Test saving checkpoint with invalid model."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)

            # Test & Verify
            with pytest.raises(TypeError):
                manager.save_checkpoint("not a model", step=100, loss=0.5)  # type: ignore[arg-type]

    def test_save_checkpoint_invalid_step(self):
        """Test saving checkpoint with invalid step values."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )

            # Test & Verify
            with pytest.raises((TypeError, ValueError)):
                manager.save_checkpoint(model, step="invalid", loss=0.5)  # type: ignore[arg-type]

            with pytest.raises(ValueError, match="Step cannot be negative"):
                manager.save_checkpoint(model, step=-1, loss=0.5)


class TestCheckpointManagerLoad:
    """Test checkpoint loading functionality."""

    def test_load_checkpoint_basic(self):
        """Test basic checkpoint loading functionality."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            original_model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )

            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(
                original_model, step=100, loss=0.5
            )

            # Test
            checkpoint_data = manager.load_checkpoint(checkpoint_path)

            # Verify
            assert isinstance(checkpoint_data, dict)
            assert "model_state" in checkpoint_data
            assert "step" in checkpoint_data
            assert "loss" in checkpoint_data
            assert "timestamp" in checkpoint_data
            assert "metadata" in checkpoint_data
            assert checkpoint_data["step"] == 100
            assert checkpoint_data["loss"] == 0.5

    def test_load_checkpoint_with_metadata(self):
        """Test loading checkpoint with metadata."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )
            original_metadata = {"epoch": 50, "loss": 0.123, "learning_rate": 0.001}

            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(
                model, step=100, metadata=original_metadata, loss=0.5
            )

            # Test
            checkpoint_data = manager.load_checkpoint(checkpoint_path)

            # Verify
            loaded_metadata = checkpoint_data["metadata"]
            assert loaded_metadata["epoch"] == 50
            assert loaded_metadata["loss"] == 0.123
            assert loaded_metadata["learning_rate"] == 0.001

    def test_load_checkpoint_preserves_parameters(self):
        """Test that loading checkpoint preserves model parameters."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            original_model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )

            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(
                original_model, step=100, loss=0.5
            )

            # Test
            checkpoint_data = manager.load_checkpoint(checkpoint_path)

            # Verify
            assert "model_state" in checkpoint_data
            assert isinstance(checkpoint_data["model_state"], nnx.State)

    def test_load_checkpoint_nonexistent_file(self):
        """Test loading checkpoint from nonexistent file."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            nonexistent_path = temp_dir + "/nonexistent.pkl"

            # Test & Verify
            with pytest.raises(FileNotFoundError):
                manager.load_checkpoint(nonexistent_path)

    def test_load_checkpoint_corrupted_file(self):
        """Test loading checkpoint from corrupted file."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            corrupted_path = Path(temp_dir) / "corrupted.pkl"

            # Create corrupted file
            with open(corrupted_path, "w") as f:
                f.write("not pickle data")

            # Test & Verify
            with pytest.raises((ValueError, OSError), match="Failed to load"):
                manager.load_checkpoint(str(corrupted_path))


class TestCheckpointManagerListing:
    """Test checkpoint listing functionality."""

    def test_list_checkpoints_empty_directory(self):
        """Test listing checkpoints in empty directory."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)

            # Test
            checkpoints = manager.list_checkpoints()

            # Verify
            assert isinstance(checkpoints, list)
            assert len(checkpoints) == 0

    def test_list_checkpoints_with_checkpoints(self):
        """Test listing checkpoints with saved checkpoints."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )

            # Save checkpoints
            manager.save_checkpoint(model, step=100, loss=0.5)
            manager.save_checkpoint(model, step=200, loss=0.5)
            manager.save_checkpoint(model, step=300, loss=0.5)

            # Test
            checkpoints = manager.list_checkpoints()

            # Verify
            assert len(checkpoints) == 3
            assert all(isinstance(cp, dict) for cp in checkpoints)
            assert any(cp["step"] == 100 for cp in checkpoints)
            assert any(cp["step"] == 200 for cp in checkpoints)
            assert any(cp["step"] == 300 for cp in checkpoints)

    def test_list_checkpoints_sorted_by_step(self):
        """Test that list_checkpoints returns sorted list."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )

            # Save checkpoints in random order
            manager.save_checkpoint(model, step=300, loss=0.5)
            manager.save_checkpoint(model, step=100, loss=0.5)
            manager.save_checkpoint(model, step=200, loss=0.5)

            # Test
            checkpoints = manager.list_checkpoints()

            # Verify
            steps = [cp["step"] for cp in checkpoints]
            assert steps == [100, 200, 300]

    def test_list_checkpoints_ignores_non_checkpoint_files(self):
        """Test that list_checkpoints ignores non-checkpoint files."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )

            # Save checkpoint
            manager.save_checkpoint(model, step=100, loss=0.5)

            # Create non-checkpoint file
            other_file = manager.checkpoint_dir / "other_file.txt"
            with open(other_file, "w") as f:
                f.write("not a checkpoint")

            # Test
            checkpoints = manager.list_checkpoints()

            # Verify
            assert len(checkpoints) == 1
            assert checkpoints[0]["step"] == 100


class TestCheckpointManagerValidation:
    """Test checkpoint validation functionality."""

    def test_validate_checkpoint_valid_file(self):
        """Test validation of valid checkpoint file."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )

            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(model, step=100, loss=0.5)

            # Test
            is_valid = manager.validate_checkpoint(checkpoint_path)

            # Verify
            assert is_valid is True

    def test_validate_checkpoint_nonexistent_file(self):
        """Test validation of nonexistent checkpoint file."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            nonexistent_path = temp_dir + "/nonexistent.pkl"

            # Test
            is_valid = manager.validate_checkpoint(nonexistent_path)

            # Verify
            assert is_valid is False

    def test_validate_checkpoint_corrupted_file(self):
        """Test validation of corrupted checkpoint file."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            corrupted_path = Path(temp_dir) / "corrupted.pkl"

            # Create corrupted file
            with open(corrupted_path, "w") as f:
                f.write("not pickle data")

            # Test
            is_valid = manager.validate_checkpoint(str(corrupted_path))

            # Verify
            assert is_valid is False

    def test_validate_checkpoint_empty_file(self):
        """Test validation of empty checkpoint file."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            empty_path = Path(temp_dir) / "empty.pkl"

            # Create empty file
            empty_path.touch()

            # Test
            is_valid = manager.validate_checkpoint(str(empty_path))

            # Verify
            assert is_valid is False


class TestCheckpointManagerIntegration:
    """Test checkpoint manager integration functionality."""

    def test_save_load_cycle_preserves_functionality(self):
        """Test that save/load cycle preserves model functionality."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            original_model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )

            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(
                original_model, step=100, loss=0.5
            )

            # Test
            restored_model = manager.restore_model(original_model, checkpoint_path)

            # Verify
            assert isinstance(restored_model, nnx.Module)

    def test_multiple_save_load_cycles(self):
        """Test multiple save/load cycles."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )

            # Test
            checkpoint_path = manager.save_checkpoint(
                model, step=100, loss=0.5, metadata={"epoch": 1}
            )
            checkpoint_data = manager.load_checkpoint(checkpoint_path)

            # Verify
            assert checkpoint_data["metadata"]["epoch"] == 1

            # Second cycle
            checkpoint_path_2 = manager.save_checkpoint(
                model, step=200, loss=0.3, metadata={"epoch": 2}
            )
            checkpoint_data_2 = manager.load_checkpoint(checkpoint_path_2)

            # Verify
            assert checkpoint_data_2["metadata"]["epoch"] == 2

    def test_checkpoint_manager_with_different_models(self):
        """Test checkpoint manager with different model types."""
        # Setup
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            rngs = nnx.Rngs(0)

            model1 = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes=4,
                num_layers=2,
                rngs=rngs,
            )
            model2 = FourierNeuralOperator(
                in_channels=2,
                out_channels=2,
                hidden_channels=64,
                modes=8,
                num_layers=3,
                rngs=rngs,
            )

            # Test
            checkpoint_path_1 = manager.save_checkpoint(model1, step=100, loss=0.5)
            checkpoint_path_2 = manager.save_checkpoint(model2, step=200, loss=0.3)

            # Verify
            checkpoint_data_1 = manager.load_checkpoint(checkpoint_path_1)
            checkpoint_data_2 = manager.load_checkpoint(checkpoint_path_2)

            assert checkpoint_data_1["step"] == 100
            assert checkpoint_data_2["step"] == 200

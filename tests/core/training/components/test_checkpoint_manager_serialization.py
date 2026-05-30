"""Serialization-safety tests for :class:`CheckpointManager`.

These tests pin the security fix for the insecure-deserialization
(pickle remote-code-execution) finding: ``CheckpointManager`` must
serialize JAX/Flax NNX state through a safe mechanism (Orbax for arrays,
JSON for plain-Python metadata) and must not import or use ``pickle``.
"""

import inspect
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

import opifex.core.training.components.checkpoint_manager as checkpoint_manager_module
from opifex.core.training.components.checkpoint_manager import CheckpointManager
from opifex.neural.operators.fno.base import FourierNeuralOperator


def _build_model() -> FourierNeuralOperator:
    """Construct a small representative neural-operator model."""
    return FourierNeuralOperator(
        in_channels=1,
        out_channels=1,
        hidden_channels=16,
        modes=4,
        num_layers=2,
        rngs=nnx.Rngs(0),
    )


class TestCheckpointSerializationSafety:
    """Verify pickle-free, round-tripping checkpoint serialization."""

    def test_checkpoint_roundtrip_without_pickle(self) -> None:
        """Save/load round-trips arrays + metadata without using pickle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            model = _build_model()
            original_state = nnx.state(model)
            metadata = {"epoch": 7, "learning_rate": 0.001, "note": "safe-serializer"}

            checkpoint_path = manager.save_checkpoint(
                model, step=100, loss=0.25, metadata=metadata
            )

            loaded = manager.load_checkpoint(checkpoint_path)

            # Restored model state must be an nnx.State whose arrays match.
            restored_state = loaded["model_state"]
            assert isinstance(restored_state, nnx.State)
            original_paths = [path for path, _ in original_state.flat_state()]
            restored_paths = [path for path, _ in restored_state.flat_state()]
            assert restored_paths == original_paths
            original_leaves = jax.tree_util.tree_leaves(original_state)
            restored_leaves = jax.tree_util.tree_leaves(restored_state)
            assert len(restored_leaves) == len(original_leaves)
            for original_leaf, restored_leaf in zip(
                original_leaves, restored_leaves, strict=True
            ):
                assert jnp.allclose(
                    jnp.asarray(restored_leaf), jnp.asarray(original_leaf)
                ), "restored array does not match original"

            # Plain-Python metadata must survive the round-trip.
            assert loaded["step"] == 100
            assert loaded["loss"] == 0.25
            assert loaded["metadata"]["epoch"] == 7
            assert loaded["metadata"]["learning_rate"] == 0.001
            assert loaded["metadata"]["note"] == "safe-serializer"

            # The restored state must be usable to update a fresh model.
            fresh = _build_model()
            nnx.update(fresh, restored_state)

    def test_module_does_not_import_pickle(self) -> None:
        """The checkpoint module must not import or reference pickle."""
        source = inspect.getsource(checkpoint_manager_module)
        assert "import pickle" not in source
        assert "pickle." not in source
        assert "S301" not in source
        assert "B301" not in source
        # No leftover pickle module object on the namespace.
        assert not hasattr(checkpoint_manager_module, "pickle")

    def test_saved_checkpoint_contains_no_pickle_bytes(self) -> None:
        """Persisted checkpoint artifacts must not be pickle streams."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            model = _build_model()

            checkpoint_path = manager.save_checkpoint(model, step=5, loss=0.5)

            artifact = Path(checkpoint_path)
            files = [artifact] if artifact.is_file() else list(artifact.rglob("*"))
            for file in files:
                if not file.is_file():
                    continue
                head = file.read_bytes()[:2]
                # Pickle protocol-2+ streams start with b"\x80"; reject them.
                assert not head.startswith(b"\x80"), f"pickle stream found in {file}"

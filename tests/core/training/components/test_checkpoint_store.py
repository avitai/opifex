"""Tests for the unified :mod:`checkpoint_store` abstraction.

These tests pin the single, orbax-backed :class:`CheckpointStore` that
replaces the previously-duplicated ``OrbaxCheckpointManager`` (step-int
addressed) and ``CheckpointManager`` (path-string addressed) classes.

The store uses Orbax's step-int addressing as the canonical disk contract.
Behaviour preserved from the deleted managers:

* lossless save/restore round-trip of an ``nnx.Module`` parameter pytree,
* rich JSON metadata (physics + additional) round-trip,
* ``max_to_keep`` retention handled natively by Orbax,
* best-metric checkpoint selection (carried over from the path-string
  manager's ``get_best_checkpoint``),
* pickle-free serialization (Orbax ``StandardSave`` + ``JsonSave``).
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from flax.training import train_state

import opifex.core.training.components.checkpoint_store as checkpoint_store_module
from opifex.core.training.components.checkpoint_store import (
    CheckpointStore,
    OrbaxCheckpointStore,
)


class _SimpleModel(nnx.Module):
    """Minimal two-layer NNX model providing real array state."""

    def __init__(self, features: int = 4, *, rngs: nnx.Rngs) -> None:
        self.dense1 = nnx.Linear(features, features, rngs=rngs)
        self.dense2 = nnx.Linear(features, features, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.dense2(nnx.relu(self.dense1(x)))


@pytest.fixture
def model() -> _SimpleModel:
    """Construct a small deterministic model."""
    return _SimpleModel(features=4, rngs=nnx.Rngs(0))


class TestOrbaxCheckpointStoreInit:
    """Initialization and validation."""

    def test_creates_directory(self, tmp_path) -> None:
        """Creates the checkpoint directory on construction."""
        ckpt_dir = tmp_path / "ckpt"
        store = OrbaxCheckpointStore(ckpt_dir)
        assert ckpt_dir.exists()
        assert store.max_to_keep == 5

    def test_custom_max_to_keep(self, tmp_path) -> None:
        """Respects a custom ``max_to_keep``."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt", max_to_keep=3)
        assert store.max_to_keep == 3

    def test_empty_dir_raises(self) -> None:
        """An empty directory string is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            OrbaxCheckpointStore("")

    def test_whitespace_dir_raises(self) -> None:
        """A whitespace-only directory string is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            OrbaxCheckpointStore("   ")

    def test_directory_is_resolved(self, tmp_path) -> None:
        """The checkpoint directory is resolved to an absolute path."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        assert store.checkpoint_dir.is_absolute()


class TestProtocolConformance:
    """The concrete store satisfies the abstract protocol."""

    def test_store_is_checkpoint_store(self, tmp_path) -> None:
        """``OrbaxCheckpointStore`` is a structural ``CheckpointStore``."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        assert isinstance(store, CheckpointStore)


class TestSaveRestoreRoundTrip:
    """Lossless save/restore of model state and metadata."""

    def test_roundtrip_nnx_module_lossless(self, tmp_path, model) -> None:
        """Restored parameters match the originals exactly."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        original_state = nnx.state(model)

        path = store.save(model, step=100, loss=0.05)
        assert path.endswith("100")

        restored_model, metadata = store.restore(model, step=100)
        assert restored_model is not None
        restored_state = nnx.state(restored_model)

        original_leaves = nnx.to_flat_state(original_state)
        restored_leaves = nnx.to_flat_state(restored_state)
        assert [p for p, _ in restored_leaves] == [p for p, _ in original_leaves]
        for (_, original), (_, restored) in zip(original_leaves, restored_leaves, strict=True):
            assert jnp.allclose(original.value, restored.value)

        assert metadata["step"] == 100
        assert metadata["loss"] == 0.05

    def test_metadata_roundtrip(self, tmp_path, model) -> None:
        """Physics and additional metadata survive the round-trip."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        physics = {"chemical_accuracy": 0.001, "scf_iterations": 15}
        additional = {"experiment_id": "exp_001"}

        store.save(
            model,
            step=10,
            loss=0.2,
            physics_metadata=physics,
            additional_metadata=additional,
        )
        _, metadata = store.restore(model, step=10)

        assert metadata["physics_metadata"]["chemical_accuracy"] == 0.001
        assert metadata["physics_metadata"]["scf_iterations"] == 15
        assert metadata["experiment_id"] == "exp_001"
        assert metadata["checkpoint_version"] == "2.0"

    def test_restore_missing_returns_original(self, tmp_path, model) -> None:
        """Restoring a never-saved step returns the original model."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        restored, metadata = store.restore(model, step=999)
        assert restored is model
        assert metadata == {}

    def test_restore_metadata_only(self, tmp_path, model) -> None:
        """Restoring with no target model yields metadata-only."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(model, step=5, loss=0.1)
        restored, metadata = store.restore(target_model=None, step=5)
        assert restored is None
        assert metadata["step"] == 5

    def test_invalid_model_type_raises(self, tmp_path) -> None:
        """Saving an unsupported model type raises ``TypeError``."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        with pytest.raises(TypeError, match=r"nnx\.Module"):
            store.save("not-a-model", step=1, loss=0.1)  # type: ignore[arg-type]

    def test_negative_step_raises(self, tmp_path, model) -> None:
        """Saving a negative step raises ``ValueError``."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        with pytest.raises(ValueError, match="non-negative"):
            store.save(model, step=-1, loss=0.1)


class TestTrainStateRoundTrip:
    """TrainState helpers preserve optimizer state and parameters."""

    def test_create_train_state_structure(self, tmp_path, model) -> None:
        """``create_train_state`` wires apply_fn/tx/params/step correctly."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        optimizer = optax.adam(1e-3)
        state = store.create_train_state(model, optimizer, step=100)

        assert isinstance(state, train_state.TrainState)
        assert state.step == 100
        assert state.apply_fn is model
        assert state.tx is optimizer
        assert state.params is not None

    def test_create_train_state_invalid_model_raises(self, tmp_path) -> None:
        """A non-NNX model raises ``TypeError``."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        with pytest.raises(TypeError, match=r"nnx\.Module"):
            store.create_train_state("not-a-model", None, step=0)  # type: ignore[arg-type]

    def test_train_state_roundtrip(self, tmp_path, model) -> None:
        """Save/restore a ``TrainState`` losslessly."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        optimizer = optax.adam(1e-3)
        state = store.create_train_state(model, optimizer, step=0)

        store.save_train_state(state, step=50, loss=0.03)
        restored, metadata = store.restore_train_state(state, step=50)

        assert isinstance(restored, train_state.TrainState)
        assert restored.step == 50
        assert restored.apply_fn is state.apply_fn
        assert restored.tx is state.tx
        assert metadata["loss"] == 0.03


class TestListingAndRetention:
    """Step listing, latest-step, retention and deletion."""

    def test_list_and_latest_step(self, tmp_path, model) -> None:
        """List returns saved steps and latest reports the newest."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        for step in (100, 200, 300):
            store.save(model, step=step, loss=0.1)
        assert sorted(store.list_steps()) == [100, 200, 300]
        assert store.latest_step() == 300

    def test_restore_specific_step_metadata(self, tmp_path, model) -> None:
        """Restoring a specific step returns that step's metadata."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        for step in (100, 200, 300):
            store.save(model, step=step, loss=0.1)
        _, metadata = store.restore(model, step=200)
        assert metadata["step"] == 200

    def test_max_to_keep_retention(self, tmp_path, model) -> None:
        """Only the most recent ``max_to_keep`` checkpoints are retained."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt", max_to_keep=2)
        for step in (1, 2, 3, 4):
            store.save(model, step=step, loss=0.1)
        kept = sorted(store.list_steps())
        assert len(kept) == 2
        assert kept == [3, 4]

    def test_delete_step(self, tmp_path, model) -> None:
        """Deleting a step removes it from the listing."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(model, step=10, loss=0.1)
        store.save(model, step=20, loss=0.1)
        assert store.delete(10) is True
        assert 10 not in store.list_steps()


class TestBestMetricSelection:
    """Best-metric checkpoint selection carried over from the old manager."""

    def test_best_step_minimize_loss(self, tmp_path, model) -> None:
        """The lowest-loss step is selected when minimizing."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(model, step=1, loss=0.9)
        store.save(model, step=2, loss=0.2)
        store.save(model, step=3, loss=0.5)
        assert store.best_step(metric="loss", minimize=True) == 2

    def test_best_step_maximize_metric(self, tmp_path, model) -> None:
        """A custom metadata metric is honoured when maximizing."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        store.save(model, step=1, loss=0.5, additional_metadata={"accuracy": 0.7})
        store.save(model, step=2, loss=0.5, additional_metadata={"accuracy": 0.9})
        assert store.best_step(metric="accuracy", minimize=False) == 2

    def test_best_step_empty_returns_none(self, tmp_path) -> None:
        """No checkpoints yields ``None``."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        assert store.best_step() is None


class TestResourceManagement:
    """Context-manager support and explicit close."""

    def test_context_manager(self, tmp_path, model) -> None:
        """The store works as a context manager and closes cleanly."""
        with OrbaxCheckpointStore(tmp_path / "ckpt") as store:
            store.save(model, step=1, loss=0.1)
            assert store.latest_step() == 1


class TestSerializationSafety:
    """No pickle anywhere in the persisted artifacts or module source."""

    def test_module_does_not_use_pickle(self) -> None:
        """The store module never imports or references pickle."""
        source = inspect.getsource(checkpoint_store_module)
        assert "import pickle" not in source
        assert "pickle." not in source

    def test_saved_artifacts_are_not_pickle(self, tmp_path, model) -> None:
        """Persisted files are not pickle streams."""
        store = OrbaxCheckpointStore(tmp_path / "ckpt")
        path = store.save(model, step=1, loss=0.1)
        from pathlib import Path

        for file in Path(path).rglob("*"):
            if file.is_file():
                head = file.read_bytes()[:2]
                assert not head.startswith(b"\x80"), f"pickle stream in {file}"

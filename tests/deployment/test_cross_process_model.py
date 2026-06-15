"""Cross-process reconstruction of registered ``nnx.Module`` models.

These tests pin the contract that a model registered by one process can be
rebuilt by another process that shares only the on-disk storage directory:
the reconstruction recipe (registered class name + init kwargs + structure
hash) and the weight checkpoint are sufficient, with no shared in-memory
``GraphDef``. A fresh process is simulated by clearing the in-memory template
cache or by constructing a brand-new :class:`ModelRegistry`.
"""

import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.deployment.core_serving import ModelMetadata, ModelRegistry
from opifex.deployment.servable_registry import (
    register_servable_model,
    ServableModelRegistry,
)


@register_servable_model("cross_process_mlp")
class CrossProcessMLP(nnx.Module):
    """Two-layer MLP whose structure depends on a configurable width."""

    def __init__(self, *, hidden: int = 16, rngs: nnx.Rngs) -> None:
        """Build an ``8 -> hidden -> 8`` MLP."""
        self.up = nnx.Linear(8, hidden, rngs=rngs)
        self.down = nnx.Linear(hidden, 8, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the MLP with a ReLU between the two linear layers."""
        return self.down(nnx.relu(self.up(x)))


def _shifted_model(hidden: int = 16) -> CrossProcessMLP:
    """Return a model whose weights are shifted off the default init."""
    model = CrossProcessMLP(hidden=hidden, rngs=nnx.Rngs(0))
    shifted = jax.tree_util.tree_map(lambda leaf: leaf + 4.0, nnx.state(model))
    nnx.update(model, shifted)
    return model


def _metadata(name: str) -> ModelMetadata:
    """Return metadata matching :class:`CrossProcessMLP`'s I/O shape."""
    return ModelMetadata(
        name=name,
        version="1.0.0",
        model_type="mlp",
        input_shape=(8,),
        output_shape=(8,),
    )


def _probe() -> jax.Array:
    """Return a deterministic ``(1, 8)`` probe input."""
    return jnp.linspace(-1.0, 1.0, 8).reshape(1, 8)


def test_servable_registry_round_trips_class() -> None:
    """The decorator registers the class for name-based resolution."""
    assert ServableModelRegistry().require("cross_process_mlp") is CrossProcessMLP


def test_brand_new_registry_reconstructs_identical_outputs() -> None:
    """A fresh ModelRegistry rebuilds a saved model with identical outputs.

    The key cross-process check: a second :class:`ModelRegistry` is built
    against the same directory (empty in-memory templates), so reconstruction
    relies entirely on the persisted recipe + weight checkpoint. The rebuilt
    module must reproduce the registered model's output exactly.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ModelRegistry(storage_path=temp_dir)
        original = _shifted_model()
        model_id = writer.register_model(
            original, _metadata("fresh_reader"), init_kwargs={"hidden": 16}
        )

        reader = ModelRegistry(storage_path=temp_dir)
        assert reader._templates == {}, "a fresh registry starts with no in-memory templates"

        loaded, _ = reader.get_model(model_id)
        probe = _probe()
        # nnx.Module is callable; pyright cannot see the dynamic __call__.
        actual = loaded(probe)  # type: ignore[operator]
        assert jnp.allclose(actual, original(probe), atol=1e-6), (
            "cross-process reconstruction did not reproduce the registered model's outputs"
        )


def test_cleared_templates_force_disk_reconstruction() -> None:
    """Clearing the in-memory cache forces a recipe-only reconstruction."""
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(storage_path=temp_dir)
        original = _shifted_model()
        model_id = registry.register_model(
            original, _metadata("cleared"), init_kwargs={"hidden": 16}
        )

        registry._templates.clear()

        loaded, _ = registry.get_model(model_id)
        probe = _probe()
        actual = loaded(probe)  # type: ignore[operator]  # nnx.Module is callable
        assert jnp.allclose(actual, original(probe), atol=1e-6)


def test_metadata_round_trips_cross_process() -> None:
    """Metadata persisted at registration is restored by a fresh registry."""
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ModelRegistry(storage_path=temp_dir)
        model_id = writer.register_model(
            _shifted_model(), _metadata("meta_round_trip"), init_kwargs={"hidden": 16}
        )

        reader = ModelRegistry(storage_path=temp_dir)
        _, metadata = reader.get_model(model_id)

        assert metadata.name == "meta_round_trip"
        assert metadata.version == "1.0.0"
        assert metadata.model_type == "mlp"
        # Shapes survive the JSON metadata round-trip (as a JSON array).
        assert tuple(metadata.input_shape) == (8,)
        assert tuple(metadata.output_shape) == (8,)


def test_recipe_persists_structure_hash_and_registered_name() -> None:
    """The recipe records the registered name and a structure version hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(storage_path=temp_dir)
        model_id = registry.register_model(
            _shifted_model(), _metadata("recipe"), init_kwargs={"hidden": 16}
        )

        recipe = json.loads((Path(temp_dir) / model_id / "model_info.json").read_text())
        assert recipe["registered_name"] == "cross_process_mlp"
        assert recipe["init_kwargs"] == {"hidden": 16}
        assert isinstance(recipe["structure_hash"], str) and recipe["structure_hash"]


def test_drift_in_config_fails_fast() -> None:
    """A recipe whose config does not match the hash fails fast on load.

    Tampering the persisted ``init_kwargs`` so the rebuilt module has a
    different structure must raise a clear, actionable error rather than
    silently restoring weights into a mismatched skeleton.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(storage_path=temp_dir)
        model_id = registry.register_model(
            _shifted_model(hidden=16), _metadata("drift"), init_kwargs={"hidden": 16}
        )
        registry._templates.clear()

        info_path = Path(temp_dir) / model_id / "model_info.json"
        recipe = json.loads(info_path.read_text())
        recipe["init_kwargs"] = {"hidden": 32}  # structural drift vs the saved hash
        info_path.write_text(json.dumps(recipe))

        with pytest.raises(ValueError, match="Structure drift"):
            registry.get_model(model_id)


def test_unregistered_class_name_fails_fast() -> None:
    """An unknown registered-name in the recipe raises an actionable error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(storage_path=temp_dir)
        model_id = registry.register_model(
            _shifted_model(), _metadata("ghost"), init_kwargs={"hidden": 16}
        )
        registry._templates.clear()

        info_path = Path(temp_dir) / model_id / "model_info.json"
        recipe = json.loads(info_path.read_text())
        recipe["registered_name"] = "no_such_servable_model"
        info_path.write_text(json.dumps(recipe))

        with pytest.raises(KeyError, match="not registered"):
            registry.get_model(model_id)


def test_registering_unregistered_class_fails_fast() -> None:
    """Registering a model whose class was never decorated fails fast."""

    class UnregisteredModel(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs) -> None:
            self.linear = nnx.Linear(8, 8, rngs=rngs)

        def __call__(self, x: jax.Array) -> jax.Array:
            return self.linear(x)

    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(storage_path=temp_dir)
        with pytest.raises(KeyError, match="not registered for serving"):
            registry.register_model(UnregisteredModel(rngs=nnx.Rngs(0)), _metadata("nope"))


def test_reconstructed_model_is_jit_grad_vmap_compatible() -> None:
    """The reconstructed module's forward survives jit/grad/vmap.

    Cross-process reconstruction must yield a live NNX module, not a frozen
    snapshot: jitting, differentiating, and vmapping its forward pass all work
    and agree with the eager reference.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ModelRegistry(storage_path=temp_dir)
        original = _shifted_model()
        model_id = writer.register_model(
            original, _metadata("transformable"), init_kwargs={"hidden": 16}
        )

        reader = ModelRegistry(storage_path=temp_dir)
        loaded, _ = reader.get_model(model_id)
        assert isinstance(loaded, nnx.Module)

        probe = _probe()
        eager = loaded(probe)  # type: ignore[operator]  # nnx.Module is callable

        graphdef, state = nnx.split(loaded)

        def forward(state: nnx.State, x: jax.Array) -> jax.Array:
            return nnx.merge(graphdef, state)(x)  # type: ignore[operator]

        jitted = jax.jit(forward)(state, probe)
        assert jnp.allclose(jitted, eager, atol=1e-6)

        def scalar_loss(state: nnx.State, x: jax.Array) -> jax.Array:
            return jnp.sum(forward(state, x) ** 2)

        grads = jax.grad(scalar_loss)(state, probe)
        leaves = jax.tree_util.tree_leaves(grads)
        assert leaves and all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)

        row = probe[0]
        per_row = lambda single: forward(state, single[None, :])[0]
        batch = jnp.broadcast_to(row, (5, row.shape[0]))
        vmapped = jax.vmap(per_row)(batch)
        assert vmapped.shape == (5, 8)
        assert jnp.all(jnp.isfinite(vmapped))
        # Identical inputs map to identical rows, and the batched forward
        # matches the eager reference up to float32 matmul tolerance.
        assert jnp.allclose(vmapped, vmapped[0], atol=1e-6)
        assert jnp.allclose(vmapped[0], eager[0], rtol=1e-3, atol=1e-3)

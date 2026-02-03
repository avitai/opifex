"""Tests for opifex.distributed.training."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import Mesh


pytestmark = pytest.mark.distributed

from opifex.distributed.training import (
    create_distributed_train_step,
    create_sharded_model,
    shard_batch,
)


# -- Simple test model --


class _LinearModel(nnx.Module):
    """Minimal NNX model for testing."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


def _mse_loss(model: nnx.Module, batch: dict[str, Any]) -> jax.Array:
    """Simple MSE loss for testing."""
    pred = model(batch["x"])  # type: ignore[reportCallIssue]
    return jnp.mean((pred - batch["y"]) ** 2)


@pytest.fixture
def single_device_mesh() -> Mesh:
    """Create a 1-device mesh for CPU testing."""
    devices = jax.devices()
    return Mesh(np.array(devices[:1]), axis_names=("data",))


class TestCreateDistributedTrainStep:
    """Test create_distributed_train_step."""

    def test_returns_callable(self) -> None:
        step = create_distributed_train_step(_mse_loss)
        assert callable(step)

    def test_train_step_decreases_loss(self, single_device_mesh: Mesh) -> None:
        model = _LinearModel(4, 2, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

        batch = {
            "x": jnp.ones((2, 4)),
            "y": jnp.zeros((2, 2)),
        }

        train_step = create_distributed_train_step(_mse_loss)

        with jax.set_mesh(single_device_mesh):
            loss_before = _mse_loss(model, batch)
            for _ in range(5):
                _loss = train_step(model, optimizer, batch)
            loss_after = _mse_loss(model, batch)

        assert float(loss_after) < float(loss_before)


class TestShardBatch:
    """Test shard_batch utility."""

    def test_shard_dict_batch(self, single_device_mesh: Mesh) -> None:
        batch = {
            "x": jnp.ones((4, 8)),
            "y": jnp.zeros((4, 2)),
        }
        sharded = shard_batch(batch, single_device_mesh)
        assert sharded["x"].shape == (4, 8)
        assert sharded["y"].shape == (4, 2)

    def test_shard_preserves_values(self, single_device_mesh: Mesh) -> None:
        x = jnp.arange(8).reshape(2, 4)
        batch = {"x": x}
        sharded = shard_batch(batch, single_device_mesh)
        np.testing.assert_array_equal(
            np.asarray(sharded["x"]),
            np.asarray(x),
        )


class TestCreateShardedModel:
    """Test create_sharded_model."""

    def test_creates_model_under_mesh(self, single_device_mesh: Mesh) -> None:
        model = create_sharded_model(
            _LinearModel,
            single_device_mesh,
            4,
            2,
            rngs=nnx.Rngs(0),
        )
        assert isinstance(model, nnx.Module)

    def test_model_is_functional(self, single_device_mesh: Mesh) -> None:
        model = create_sharded_model(
            _LinearModel,
            single_device_mesh,
            4,
            2,
            rngs=nnx.Rngs(0),
        )
        x = jnp.ones((1, 4))
        assert isinstance(model, _LinearModel)
        out = model(x)
        assert out.shape == (1, 2)

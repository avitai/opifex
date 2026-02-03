"""Tests for opifex.distributed.manager."""

from __future__ import annotations

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, PartitionSpec


pytestmark = pytest.mark.distributed

from opifex.distributed.config import DistributedConfig
from opifex.distributed.manager import DistributedManager


@pytest.fixture
def default_config() -> DistributedConfig:
    """Default single-axis data-parallel config."""
    return DistributedConfig()


@pytest.fixture
def mock_mesh_manager() -> MagicMock:
    """Mock DeviceMeshManager for unit testing without real devices."""
    import numpy as np

    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=("data",))

    manager = MagicMock()
    manager.create_device_mesh.return_value = mesh
    manager.get_mesh_info.return_value = {
        "total_devices": len(devices),
        "axes": {"data": len(devices)},
    }
    return manager


class TestDistributedManagerCreation:
    """Test DistributedManager mesh creation."""

    def test_create_mesh_with_mock(
        self,
        default_config: DistributedConfig,
        mock_mesh_manager: MagicMock,
    ) -> None:
        mgr = DistributedManager(default_config, mesh_manager=mock_mesh_manager)
        mesh = mgr.create_mesh()
        assert isinstance(mesh, Mesh)
        mock_mesh_manager.create_device_mesh.assert_called_once()

    def test_mesh_lazy_creation(
        self,
        default_config: DistributedConfig,
        mock_mesh_manager: MagicMock,
    ) -> None:
        mgr = DistributedManager(default_config, mesh_manager=mock_mesh_manager)
        # Access mesh property twice — should only create once
        _ = mgr.mesh
        _ = mgr.mesh
        mock_mesh_manager.create_device_mesh.assert_called_once()

    def test_resolve_minus_one(
        self,
        default_config: DistributedConfig,
        mock_mesh_manager: MagicMock,
    ) -> None:
        mgr = DistributedManager(default_config, mesh_manager=mock_mesh_manager)
        mgr.create_mesh()
        call_args = mock_mesh_manager.create_device_mesh.call_args
        mesh_spec = call_args[0][0]
        # -1 should be resolved to actual device count
        _, size = mesh_spec[0]
        assert size == jax.device_count()

    def test_config_property(self, default_config: DistributedConfig) -> None:
        mgr = DistributedManager(default_config, mesh_manager=MagicMock())
        assert mgr.config is default_config


class TestDistributedManagerSharding:
    """Test array sharding and replication."""

    def test_shard_array(
        self,
        default_config: DistributedConfig,
        mock_mesh_manager: MagicMock,
    ) -> None:
        mgr = DistributedManager(default_config, mesh_manager=mock_mesh_manager)
        arr = jnp.ones((4, 8))
        result = mgr.shard_array(arr, PartitionSpec("data", None))
        assert result.shape == (4, 8)

    def test_replicate_array(
        self,
        default_config: DistributedConfig,
        mock_mesh_manager: MagicMock,
    ) -> None:
        mgr = DistributedManager(default_config, mesh_manager=mock_mesh_manager)
        arr = jnp.ones((4, 8))
        result = mgr.replicate_array(arr)
        assert result.shape == (4, 8)


class TestDistributedManagerInfo:
    """Test mesh info retrieval."""

    def test_get_mesh_info(
        self,
        default_config: DistributedConfig,
        mock_mesh_manager: MagicMock,
    ) -> None:
        mgr = DistributedManager(default_config, mesh_manager=mock_mesh_manager)
        info = mgr.get_mesh_info()
        assert "strategy" in info
        assert info["strategy"] == "data"
        assert "total_devices" in info

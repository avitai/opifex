"""Distributed mesh management for Opifex.

Wraps ``datarax.distributed.DeviceMeshManager`` to create and manage
JAX device meshes for distributed PDE training.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec


if TYPE_CHECKING:
    from opifex.distributed.config import DistributedConfig


logger = logging.getLogger(__name__)


class DistributedManager:
    """Manage JAX device meshes for distributed training.

    Wraps ``datarax.distributed.DeviceMeshManager`` for mesh creation
    and provides sharding/replication utilities.

    Args:
        config: Distributed training configuration.
        mesh_manager: Optional ``DeviceMeshManager`` for dependency injection
            (SWE Rule 3). If ``None``, imports from datarax.
    """

    def __init__(
        self,
        config: DistributedConfig,
        mesh_manager: Any | None = None,
    ) -> None:
        self._config = config
        if mesh_manager is None:
            from datarax.distributed import DeviceMeshManager

            mesh_manager = DeviceMeshManager
        self._mesh_manager = mesh_manager
        self._mesh: Mesh | None = None

    @property
    def config(self) -> DistributedConfig:
        """Return the distributed config."""
        return self._config

    @property
    def mesh(self) -> Mesh:
        """Return the active mesh, creating it lazily if needed."""
        if self._mesh is None:
            self._mesh = self.create_mesh()
        return self._mesh

    def create_mesh(self) -> Mesh:
        """Create a JAX device mesh from the config.

        Resolves ``-1`` in ``mesh_shape`` to the actual device count.

        Returns:
            A configured JAX ``Mesh``.
        """
        shape = self._config.mesh_shape
        axis_names = self._config.mesh_axis_names

        # Resolve -1 to actual device count
        resolved_shape = tuple(
            jax.device_count() if dim == -1 else dim for dim in shape
        )

        mesh_spec = list(zip(axis_names, resolved_shape, strict=True))
        mesh = self._mesh_manager.create_device_mesh(mesh_spec)
        self._mesh = mesh

        logger.info(
            "Created device mesh: shape=%s, axes=%s, strategy=%s",
            resolved_shape,
            axis_names,
            self._config.strategy,
        )
        return mesh

    def shard_array(
        self,
        array: jax.Array,
        partition_spec: PartitionSpec,
    ) -> jax.Array:
        """Shard an array across the mesh according to a partition spec.

        Args:
            array: The JAX array to shard.
            partition_spec: How to distribute across mesh axes.

        Returns:
            The sharded array.
        """
        sharding = NamedSharding(self.mesh, partition_spec)
        return jax.device_put(array, sharding)

    def replicate_array(self, array: jax.Array) -> jax.Array:
        """Replicate an array across all mesh devices.

        Args:
            array: The JAX array to replicate.

        Returns:
            The replicated array.
        """
        sharding = NamedSharding(self.mesh, PartitionSpec())
        return jax.device_put(array, sharding)

    def get_mesh_info(self) -> dict[str, Any]:
        """Return information about the current mesh.

        Returns:
            Dictionary with ``device_count``, ``axes``, and ``strategy``.
        """
        return self._mesh_manager.get_mesh_info(self.mesh) | {
            "strategy": self._config.strategy,
        }

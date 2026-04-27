"""Distributed training module for Opifex.

Provides PDE-specific distributed training infrastructure on top of
``datarax.distributed``, using modern JAX SPMD patterns.

Components:
    - :class:`DistributedConfig`: Immutable training configuration.
    - :class:`DistributedManager`: Mesh creation and array sharding.
    - :func:`create_distributed_train_step`: JIT-compiled sharded training.
    - :func:`shard_batch`: Shard data across the data axis.
    - :func:`create_sharded_model`: Init model under a mesh context.
    - :class:`MeshRules`: Logical-to-physical axis mapping (from datarax).
    - :func:`partition_spec_for_names`: Convert logical axis names to a
      ``PartitionSpec`` using Datarax's current sharding API.
"""

from opifex.distributed.config import DistributedConfig
from opifex.distributed.manager import DistributedManager
from opifex.distributed.training import (
    create_distributed_train_step,
    create_sharded_model,
    data_parallel_rules,
    MeshRules,
    partition_spec_for_names,
    shard_batch,
)


__all__ = [
    "DistributedConfig",
    "DistributedManager",
    "MeshRules",
    "create_distributed_train_step",
    "create_sharded_model",
    "data_parallel_rules",
    "partition_spec_for_names",
    "shard_batch",
]

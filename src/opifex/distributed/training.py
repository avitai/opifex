"""Distributed training utilities for PDE solvers.

Provides functions to create sharded training steps, shard batches,
and initialize models in a mesh context using modern JAX SPMD patterns
(``nnx.jit`` + ``NamedSharding``).
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import jax
from datarax.distributed import (
    apply_sharding_rules,
    create_data_parallel_sharding,
    data_parallel_rules,
    MeshRules,
    shard_batch as datarax_shard_batch,
    spmd_train_step,
)
from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.sharding import Mesh


logger = logging.getLogger(__name__)

# Re-export useful datarax sharding utilities
__all__ = [
    "MeshRules",
    "apply_sharding_rules",
    "create_distributed_train_step",
    "create_sharded_model",
    "data_parallel_rules",
    "shard_batch",
]


def create_distributed_train_step(
    loss_fn: Callable[[nnx.Module, Any], jax.Array],
) -> Callable[..., jax.Array]:
    """Create a JIT-compiled, mesh-aware training step.

    Uses datarax's ``spmd_train_step`` under the hood, which leverages
    ``nnx.value_and_grad`` and relies on the XLA compiler for automatic
    gradient AllReduce based on input sharding.

    Args:
        loss_fn: A function ``(model, batch) -> loss_scalar``.

    Returns:
        A JIT-compiled training step function with signature
        ``(model, optimizer, batch) -> loss``.

    Example::

        mesh = jax.make_mesh((4,), ("data",))
        train_step = create_distributed_train_step(my_loss_fn)

        with jax.set_mesh(mesh):
            loss = train_step(model, optimizer, sharded_batch)
    """

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: Any,
    ) -> jax.Array:
        return spmd_train_step(model, optimizer, loss_fn, batch)

    return train_step


def shard_batch(
    batch: Any,
    mesh: Mesh,
    data_axis: str = "data",
) -> Any:
    """Shard a batch of data across devices along the data axis.

    Args:
        batch: A pytree of JAX arrays (the training batch).
        mesh: The device mesh.
        data_axis: The mesh axis name for data parallelism.

    Returns:
        The batch with arrays sharded along the first dimension.
    """
    sharding = create_data_parallel_sharding(mesh, data_axis)
    return datarax_shard_batch(batch, sharding)


def create_sharded_model(
    init_fn: Callable[..., nnx.Module],
    mesh: Mesh,
    *init_args: Any,
    **init_kwargs: Any,
) -> nnx.Module:
    """Initialize a model within a mesh context for automatic sharding.

    When ``nnx.with_partitioning`` is used in the model definition,
    parameter sharding annotations are resolved against the active mesh.

    Args:
        init_fn: A callable that returns an ``nnx.Module`` (e.g., a class).
        mesh: The device mesh to use during initialization.
        *init_args: Positional arguments forwarded to ``init_fn``.
        **init_kwargs: Keyword arguments forwarded to ``init_fn``.

    Returns:
        The initialized model with sharding annotations applied.

    Example::

        mesh = jax.make_mesh((4,), ("data",))
        model = create_sharded_model(MyPDEModel, mesh, features=64)
    """
    with jax.set_mesh(mesh):
        model = init_fn(*init_args, **init_kwargs)

    logger.info(
        "Initialized model %s under mesh with axes %s",
        type(model).__name__,
        mesh.axis_names,
    )
    return model

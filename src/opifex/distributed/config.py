"""Configuration for distributed training.

This module provides the ``DistributedConfig`` dataclass for specifying
distributed training strategies, mesh topology, and gradient reduction.
"""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DistributedConfig:
    """Immutable configuration for distributed training.

    Attributes:
        mesh_shape: Device mesh shape as a tuple of ints. ``(-1,)`` means
            use all available devices on a single axis.
        mesh_axis_names: Names for each mesh axis. Must have the same
            length as ``mesh_shape``.
        strategy: Parallelism strategy.
        gradient_reduce_type: How to reduce gradients (``"mean"`` or ``"sum"``).
    """

    mesh_shape: tuple[int, ...] = (-1,)
    mesh_axis_names: tuple[str, ...] = ("data",)
    strategy: Literal["data", "fsdp", "hybrid"] = "data"
    gradient_reduce_type: Literal["mean", "sum"] = "mean"

    def __post_init__(self) -> None:
        """Validate configuration on creation (fail-fast)."""
        if len(self.mesh_shape) != len(self.mesh_axis_names):
            raise ValueError(
                f"mesh_shape length ({len(self.mesh_shape)}) must match "
                f"mesh_axis_names length ({len(self.mesh_axis_names)})"
            )
        valid_strategies = {"data", "fsdp", "hybrid"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. Must be one of {valid_strategies}"
            )
        valid_reductions = {"mean", "sum"}
        if self.gradient_reduce_type not in valid_reductions:
            raise ValueError(
                f"Invalid gradient_reduce_type '{self.gradient_reduce_type}'. "
                f"Must be one of {valid_reductions}"
            )

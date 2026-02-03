"""Tests for opifex.distributed.config."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.distributed

from opifex.distributed.config import DistributedConfig


class TestDistributedConfigDefaults:
    """Test default values of DistributedConfig."""

    def test_default_mesh_shape(self) -> None:
        cfg = DistributedConfig()
        assert cfg.mesh_shape == (-1,)

    def test_default_mesh_axis_names(self) -> None:
        cfg = DistributedConfig()
        assert cfg.mesh_axis_names == ("data",)

    def test_default_strategy(self) -> None:
        cfg = DistributedConfig()
        assert cfg.strategy == "data"

    def test_default_gradient_reduce_type(self) -> None:
        cfg = DistributedConfig()
        assert cfg.gradient_reduce_type == "mean"


class TestDistributedConfigImmutability:
    """Test that DistributedConfig is frozen."""

    def test_frozen_mesh_shape(self) -> None:
        cfg = DistributedConfig()
        with pytest.raises(AttributeError):
            cfg.mesh_shape = (2,)  # type: ignore[misc]

    def test_frozen_strategy(self) -> None:
        cfg = DistributedConfig()
        with pytest.raises(AttributeError):
            cfg.strategy = "fsdp"  # type: ignore[misc]


class TestDistributedConfigCustomValues:
    """Test custom configurations."""

    def test_hybrid_config(self) -> None:
        cfg = DistributedConfig(
            mesh_shape=(2, 4),
            mesh_axis_names=("data", "model"),
            strategy="hybrid",
            gradient_reduce_type="sum",
        )
        assert cfg.mesh_shape == (2, 4)
        assert cfg.mesh_axis_names == ("data", "model")
        assert cfg.strategy == "hybrid"
        assert cfg.gradient_reduce_type == "sum"

    def test_fsdp_config(self) -> None:
        cfg = DistributedConfig(
            mesh_shape=(4, 2),
            mesh_axis_names=("data", "fsdp"),
            strategy="fsdp",
        )
        assert cfg.strategy == "fsdp"


class TestDistributedConfigValidation:
    """Test fail-fast validation."""

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="mesh_shape length"):
            DistributedConfig(
                mesh_shape=(2, 4),
                mesh_axis_names=("data",),
            )

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid strategy"):
            DistributedConfig(strategy="invalid")  # type: ignore[arg-type]

    def test_invalid_gradient_reduce_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid gradient_reduce_type"):
            DistributedConfig(gradient_reduce_type="max")  # type: ignore[arg-type]

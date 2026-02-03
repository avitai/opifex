"""Tests for BenchmarkRegistry with calibrax Registry composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from pathlib import Path

from opifex.benchmarking.benchmark_registry import (
    BenchmarkConfig,
    BenchmarkRegistry,
    DomainConfig,
)


class _DummyOperator:
    """Minimal operator class for registration tests."""


class _AnotherOperator:
    """Another operator for multi-registration tests."""


@pytest.fixture
def registry(tmp_path: Path) -> BenchmarkRegistry:
    """Create a BenchmarkRegistry with a temporary config path."""
    return BenchmarkRegistry(config_path=str(tmp_path / "registry.json"))


@pytest.fixture
def sample_benchmark() -> BenchmarkConfig:
    """Create a sample benchmark configuration."""
    return BenchmarkConfig(
        name="darcy_flow_64",
        domain="fluid_dynamics",
        problem_type="steady_state",
        input_shape=(1, 64, 64),
        output_shape=(1, 64, 64),
    )


class TestOperatorRegistration:
    """Tests for operator registration via calibrax Registry composition."""

    def test_register_operator(self, registry: BenchmarkRegistry) -> None:
        """Register a single operator and retrieve it."""
        registry.register_operator(_DummyOperator)
        assert registry.get_operator_class("_DummyOperator") is _DummyOperator

    def test_list_operators(self, registry: BenchmarkRegistry) -> None:
        """List registered operators."""
        registry.register_operator(_DummyOperator)
        registry.register_operator(_AnotherOperator)
        names = registry.list_available_operators()
        assert "_DummyOperator" in names
        assert "_AnotherOperator" in names

    def test_get_unknown_operator_raises(self, registry: BenchmarkRegistry) -> None:
        """Getting an unregistered operator raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            registry.get_operator_class("NonExistent")

    def test_register_with_metadata(self, registry: BenchmarkRegistry) -> None:
        """Metadata is stored alongside the operator."""
        metadata = {"framework": "flax_nnx", "supports_gpu": True}
        registry.register_operator(_DummyOperator, metadata=metadata)
        assert registry._operator_metadata["_DummyOperator"]["framework"] == "flax_nnx"

    def test_auto_metadata_when_none(self, registry: BenchmarkRegistry) -> None:
        """Auto-generates metadata when none is provided."""
        registry.register_operator(_DummyOperator)
        assert "_DummyOperator" in registry._operator_metadata
        assert "module" in registry._operator_metadata["_DummyOperator"]

    def test_operator_count(self, registry: BenchmarkRegistry) -> None:
        """Internal operator registry tracks count correctly."""
        assert len(registry._operator_registry) == 0
        registry.register_operator(_DummyOperator)
        assert len(registry._operator_registry) == 1


class TestBenchmarkRegistration:
    """Tests for benchmark configuration registration."""

    def test_register_benchmark(
        self, registry: BenchmarkRegistry, sample_benchmark: BenchmarkConfig
    ) -> None:
        """Register a benchmark and retrieve it."""
        registry.register_benchmark(sample_benchmark)
        config = registry.get_benchmark_config("darcy_flow_64")
        assert config.domain == "fluid_dynamics"

    def test_list_benchmarks(
        self, registry: BenchmarkRegistry, sample_benchmark: BenchmarkConfig
    ) -> None:
        """List registered benchmarks."""
        registry.register_benchmark(sample_benchmark)
        names = registry.list_available_benchmarks()
        assert "darcy_flow_64" in names

    def test_get_unknown_benchmark_raises(self, registry: BenchmarkRegistry) -> None:
        """Getting an unregistered benchmark raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            registry.get_benchmark_config("nonexistent")

    def test_benchmark_suite_by_domain(
        self, registry: BenchmarkRegistry, sample_benchmark: BenchmarkConfig
    ) -> None:
        """Get benchmarks filtered by domain."""
        registry.register_benchmark(sample_benchmark)
        suite = registry.get_benchmark_suite("fluid_dynamics")
        assert len(suite) == 1
        assert suite[0].name == "darcy_flow_64"

    def test_empty_suite_for_unknown_domain(self, registry: BenchmarkRegistry) -> None:
        """Empty list for a domain with no benchmarks."""
        assert registry.get_benchmark_suite("nonexistent") == []


class TestDomainConfig:
    """Tests for domain configuration."""

    def test_default_domains_initialized(self, registry: BenchmarkRegistry) -> None:
        """Registry has default scientific domains."""
        domains = registry.list_available_domains()
        assert "fluid_dynamics" in domains
        assert "quantum_computing" in domains
        assert "materials_science" in domains

    def test_get_domain_config(self, registry: BenchmarkRegistry) -> None:
        """Retrieve domain configuration with tolerance ranges."""
        config = registry.get_domain_specific_config("fluid_dynamics")
        assert isinstance(config, DomainConfig)
        assert "mse" in config.tolerance_ranges

    def test_unknown_domain_raises(self, registry: BenchmarkRegistry) -> None:
        """Unknown domain raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            registry.get_domain_specific_config("astrophysics")


class TestCompatibility:
    """Tests for operator-benchmark compatibility tracking."""

    def test_compatible_operators_listed(
        self, registry: BenchmarkRegistry, sample_benchmark: BenchmarkConfig
    ) -> None:
        """Operators marked compatible after benchmark registration."""
        registry.register_operator(_DummyOperator)
        registry.register_benchmark(sample_benchmark)
        compatible = registry.list_compatible_operators("darcy_flow_64")
        assert "_DummyOperator" in compatible

    def test_compatibility_report(
        self, registry: BenchmarkRegistry, sample_benchmark: BenchmarkConfig
    ) -> None:
        """Compatibility report includes coverage stats."""
        registry.register_operator(_DummyOperator)
        registry.register_benchmark(sample_benchmark)
        report = registry.generate_compatibility_report()
        assert report["total_operators"] == 1
        assert report["total_benchmarks"] == 1


class TestPersistence:
    """Tests for JSON save/load of registry state."""

    def test_save_and_load(
        self, tmp_path: Path, sample_benchmark: BenchmarkConfig
    ) -> None:
        """Save registry and reload from file preserves benchmarks."""
        config_path = str(tmp_path / "registry.json")
        reg = BenchmarkRegistry(config_path=config_path)
        reg.register_benchmark(sample_benchmark)
        reg.save_registry()

        # Reload
        reg2 = BenchmarkRegistry(config_path=config_path)
        assert "darcy_flow_64" in reg2.list_available_benchmarks()

    def test_corrupted_file_uses_defaults(self, tmp_path: Path) -> None:
        """Corrupted config file falls back gracefully."""
        config_path = tmp_path / "registry.json"
        config_path.write_text("{invalid json")
        reg = BenchmarkRegistry(config_path=str(config_path))
        # Should not crash, just log warning
        assert reg.list_available_benchmarks() == []

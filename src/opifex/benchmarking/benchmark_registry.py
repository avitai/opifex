"""Benchmark Registry for Opifex Advanced Benchmarking System

Manages available benchmarks and neural operators with domain organization.
Provides registration, discovery, and configuration management for the
comprehensive benchmarking ecosystem.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from calibrax.core.registry import Registry


# Set up logger for this module
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainConfig:
    """Configuration for a specific scientific domain."""

    name: str
    tolerance_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    required_metrics: list[str] = field(default_factory=list)
    reference_methods: list[str] = field(default_factory=list)
    default_problem_sizes: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set default configurations for common domains."""
        if not self.tolerance_ranges:
            domain_defaults: dict[str, dict[str, tuple[float, float]]] = {
                "fluid_dynamics": {
                    "mse": (1e-6, 1e-2),
                    "relative_error": (1e-4, 1e-1),
                    "mae": (1e-5, 1e-2),
                },
                "quantum_computing": {
                    "mse": (1e-8, 1e-4),
                    "relative_error": (1e-6, 1e-3),
                    "chemical_accuracy": (1e-3, 5e-3),
                },
                "materials_science": {
                    "mse": (1e-7, 1e-3),
                    "formation_energy": (1e-2, 5e-2),
                    "force_accuracy": (1e-2, 1e-1),
                },
                "climate_modeling": {
                    "mse": (1e-5, 1e-2),
                    "spatial_correlation": (0.8, 0.99),
                    "temporal_correlation": (0.7, 0.95),
                },
            }
            object.__setattr__(
                self,
                "tolerance_ranges",
                domain_defaults.get(
                    self.name,
                    {
                        "mse": (1e-6, 1e-2),
                        "relative_error": (1e-4, 1e-1),
                        "mae": (1e-5, 1e-2),
                    },
                ),
            )

        if not self.required_metrics:
            object.__setattr__(
                self, "required_metrics", ["mse", "mae", "relative_error"]
            )

        if not self.default_problem_sizes:
            object.__setattr__(self, "default_problem_sizes", [64, 128, 256, 512])


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkConfig:
    """Configuration for a specific benchmark."""

    name: str
    domain: str
    problem_type: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dataset_path: str | None = None
    reference_solution_path: str | None = None
    physics_constraints: dict[str, Any] = field(default_factory=dict)
    computational_requirements: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate benchmark configuration."""
        if not self.name:
            raise ValueError("Benchmark name is required")
        if not self.domain:
            raise ValueError("Domain is required")
        if not self.input_shape or not self.output_shape:
            raise ValueError("Input and output shapes are required")


class BenchmarkRegistry:
    """Manages available benchmarks and neural operators with domain organization.

    This registry provides centralized management of:
    - Neural operator architectures available for benchmarking
    - Benchmark problems organized by scientific domain
    - Domain-specific configurations and requirements
    - Compatibility checking between operators and benchmarks
    """

    def __init__(self, config_path: str | None = None):
        """Initialize the benchmark registry.

        Args:
            config_path: Path to registry configuration file
        """
        self.config_path = (
            Path(config_path) if config_path else Path("benchmark_registry.json")
        )

        # Internal storage — operator lookup via calibrax Registry[type]
        self._operator_registry: Registry[type] = Registry()
        self._benchmarks: dict[str, BenchmarkConfig] = {}
        self._domains: dict[str, DomainConfig] = {}
        self._operator_metadata: dict[str, dict[str, Any]] = {}
        self._benchmark_operator_compatibility: dict[str, set[str]] = defaultdict(set)

        # Initialize with default domains
        self._setup_default_domains()
        self._load_registry()

    def _setup_default_domains(self) -> None:
        """Setup default scientific domain configurations."""
        domains = [
            DomainConfig(name="fluid_dynamics"),
            DomainConfig(name="quantum_computing"),
            DomainConfig(name="materials_science"),
            DomainConfig(name="climate_modeling"),
            DomainConfig(name="molecular_dynamics"),
            DomainConfig(name="plasma_physics"),
        ]

        for domain in domains:
            self._domains[domain.name] = domain

    def _load_registry(self) -> None:
        """Load registry configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    config = json.load(f)

                # Load benchmarks
                for benchmark_data in config.get("benchmarks", []):
                    benchmark = BenchmarkConfig(**benchmark_data)
                    self._benchmarks[benchmark.name] = benchmark

                # Load operator metadata
                self._operator_metadata = config.get("operator_metadata", {})

                # Load compatibility mappings
                compatibility = config.get("compatibility", {})
                for benchmark_name, compatible_ops in compatibility.items():
                    self._benchmark_operator_compatibility[benchmark_name] = set(
                        compatible_ops
                    )

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("Could not load registry config: %s", e)

    def save_registry(self) -> None:
        """Save registry configuration to file."""
        config = {
            "benchmarks": [
                {
                    "name": benchmark.name,
                    "domain": benchmark.domain,
                    "problem_type": benchmark.problem_type,
                    "input_shape": benchmark.input_shape,
                    "output_shape": benchmark.output_shape,
                    "dataset_path": benchmark.dataset_path,
                    "reference_solution_path": benchmark.reference_solution_path,
                    "physics_constraints": benchmark.physics_constraints,
                    "computational_requirements": benchmark.computational_requirements,
                }
                for benchmark in self._benchmarks.values()
            ],
            "operator_metadata": self._operator_metadata,
            "compatibility": {
                name: list(compatible_ops)
                for name, compatible_ops in (
                    self._benchmark_operator_compatibility.items()
                )
            },
        }

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def register_operator(
        self, operator_class: type, metadata: dict[str, Any] | None = None
    ) -> None:
        """Register a neural operator for benchmarking.

        Args:
            operator_class: Neural operator class to register
            metadata: Additional metadata about the operator
        """
        operator_name = operator_class.__name__
        self._operator_registry.register(operator_name, operator_class)

        if metadata:
            self._operator_metadata[operator_name] = metadata
        else:
            # Auto-detect basic metadata
            self._operator_metadata[operator_name] = {
                "module": operator_class.__module__,
                "supports_gpu": callable(operator_class),
                "framework": "flax_nnx",
            }

    def register_benchmark(self, benchmark_config: BenchmarkConfig) -> None:
        """Register a benchmark configuration.

        Args:
            benchmark_config: Benchmark configuration to register
        """
        self._benchmarks[benchmark_config.name] = benchmark_config

        # Auto-detect compatible operators based on input/output shapes
        self._update_compatibility(benchmark_config)

    def _update_compatibility(self, benchmark: BenchmarkConfig) -> None:
        """Update operator-benchmark compatibility based on shapes."""
        for operator_name in self._operator_registry:
            try:
                operator_class = self._operator_registry.get(operator_name)
                if callable(operator_class):
                    self._benchmark_operator_compatibility[benchmark.name].add(
                        operator_name
                    )
            except (KeyError, TypeError):
                continue

    def get_benchmark_suite(self, domain: str) -> list[BenchmarkConfig]:
        """Get all benchmarks for a specific domain.

        Args:
            domain: Scientific domain name

        Returns:
            List of benchmark configurations for the domain
        """
        return [
            benchmark
            for benchmark in self._benchmarks.values()
            if benchmark.domain == domain
        ]

    def list_compatible_operators(self, benchmark_name: str) -> list[str]:
        """Get list of operators compatible with a benchmark.

        Args:
            benchmark_name: Name of the benchmark

        Returns:
            List of compatible operator names
        """
        return list(self._benchmark_operator_compatibility.get(benchmark_name, set()))

    def get_domain_specific_config(self, domain: str) -> DomainConfig:
        """Get configuration for a specific domain.

        Args:
            domain: Domain name

        Returns:
            Domain configuration

        Raises:
            ValueError: If domain not found
        """
        if domain not in self._domains:
            raise ValueError(f"Domain '{domain}' not found in registry")
        return self._domains[domain]

    def get_operator_class(self, operator_name: str) -> type:
        """Get operator class by name.

        Args:
            operator_name: Name of the operator

        Returns:
            Operator class

        Raises:
            ValueError: If operator not found
        """
        if not self._operator_registry.has(operator_name):
            raise ValueError(f"Operator '{operator_name}' not found in registry")
        return self._operator_registry.get(operator_name)

    def get_operator_metadata(self, operator_name: str) -> dict[str, Any]:
        """Get metadata for a registered operator.

        Args:
            operator_name: Name of the operator.

        Returns:
            Metadata dictionary for the operator, or empty dict if not found.
        """
        return self._operator_metadata.get(operator_name, {})

    def get_benchmark_config(self, benchmark_name: str) -> BenchmarkConfig:
        """Get benchmark configuration by name.

        Args:
            benchmark_name: Name of the benchmark

        Returns:
            Benchmark configuration

        Raises:
            ValueError: If benchmark not found
        """
        if benchmark_name not in self._benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not found in registry")
        return self._benchmarks[benchmark_name]

    def list_available_domains(self) -> list[str]:
        """Get list of available domains."""
        return list(self._domains.keys())

    def list_available_operators(self) -> list[str]:
        """Get list of available operators."""
        return self._operator_registry.list_names()

    def list_available_benchmarks(self) -> list[str]:
        """Get list of available benchmarks."""
        return list(self._benchmarks.keys())

    def auto_discover_operators(self) -> None:
        """Auto-discover neural operators from opifex.neural.operators module."""
        try:
            from opifex.neural import operators

            # Get all operator classes from the module
            operator_classes = []
            for attr_name in dir(operators):
                attr = getattr(operators, attr_name)
                if (
                    isinstance(attr, type)
                    and callable(attr)
                    and attr_name not in ["Module", "nnx"]
                ):  # Skip base classes
                    operator_classes.append(attr)

            # Register discovered operators
            for operator_class in operator_classes:
                if not self._operator_registry.has(operator_class.__name__):
                    self.register_operator(operator_class)

        except ImportError as e:
            logger.warning("Could not auto-discover operators: %s", e)

    def generate_compatibility_report(self) -> dict[str, Any]:
        """Generate a report of benchmark-operator compatibility.

        Returns:
            Comprehensive compatibility report
        """
        report = {
            "total_operators": len(self._operator_registry),
            "total_benchmarks": len(self._benchmarks),
            "total_domains": len(self._domains),
            "compatibility_matrix": {},
            "operator_coverage": {},
            "benchmark_coverage": {},
        }

        # Generate compatibility matrix
        for benchmark_name in self._benchmarks:
            compatible_ops = self.list_compatible_operators(benchmark_name)
            report["compatibility_matrix"][benchmark_name] = compatible_ops

        # Operator coverage (how many benchmarks each operator supports)
        for operator_name in self._operator_registry:
            coverage_count = sum(
                1
                for benchmark_name in self._benchmarks
                if operator_name in self.list_compatible_operators(benchmark_name)
            )
            report["operator_coverage"][operator_name] = coverage_count

        # Benchmark coverage (how many operators support each benchmark)
        for benchmark_name in self._benchmarks:
            compatible_count = len(self.list_compatible_operators(benchmark_name))
            report["benchmark_coverage"][benchmark_name] = compatible_count

        return report

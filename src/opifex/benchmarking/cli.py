"""Benchmarking CLI - Command-line interface for running Opifex benchmarks.

Usage:
    python -m opifex.benchmarking.cli -b PDEBench_2D_DarcyFlow -o TFNO
    python -m opifex.benchmarking.cli --list-benchmarks
    python -m opifex.benchmarking.cli --list-operators
"""

# ruff: noqa: T201  # CLI module - print statements are intentional

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="opifex-benchmark",
        description="Run Opifex neural operator benchmarks",
    )

    # Benchmark selection
    parser.add_argument(
        "-b",
        "--benchmark",
        help="Benchmark name to run (e.g., PDEBench_2D_DarcyFlow)",
    )
    parser.add_argument(
        "-o",
        "--operator",
        help="Operator name to benchmark (e.g., TensorizedFourierNeuralOperator)",
    )

    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Output options
    parser.add_argument(
        "--output",
        help="Output file for results (JSON format)",
    )

    # List commands
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmarks and exit",
    )
    parser.add_argument(
        "--list-operators",
        action="store_true",
        help="List available operators and exit",
    )

    return parser.parse_args(args)


def _get_registry():
    """Get benchmark registry with PDEBench benchmarks and standard operators."""
    from opifex.benchmarking.benchmark_registry import BenchmarkRegistry
    from opifex.benchmarking.pdebench_configs import register_pdebench_benchmarks
    from opifex.neural.operators.fno.tensorized import TensorizedFourierNeuralOperator

    registry = BenchmarkRegistry()

    # Register standard benchmarks
    register_pdebench_benchmarks(registry)

    # Register standard operators with metadata
    registry.register_operator(
        TensorizedFourierNeuralOperator,
        metadata={"operator_type": "fno"},
    )

    return registry


def _list_benchmarks() -> None:
    """Print available benchmarks."""
    registry = _get_registry()

    print("Available Benchmarks:")
    print("-" * 40)

    for name in sorted(registry.list_available_benchmarks()):
        config = registry.get_benchmark_config(name)
        loader_type = config.computational_requirements.get("loader_type", "unknown")
        print(f"  {name}")
        print(f"    Domain: {config.domain}")
        print(f"    Loader: {loader_type}")
        print()


def _list_operators() -> None:
    """Print available operators."""
    registry = _get_registry()

    print("Available Operators:")
    print("-" * 40)

    for name in sorted(registry.list_available_operators()):
        metadata = registry._operator_metadata.get(name, {})
        op_type = metadata.get("operator_type", "unknown")
        print(f"  {name}")
        print(f"    Type: {op_type}")
        print()


def _run_benchmark(
    benchmark_name: str,
    operator_name: str,
    epochs: int,
    batch_size: int,
    seed: int,
    output_file: str | None,
) -> int:
    """Run a benchmark and return exit code.

    Args:
        benchmark_name: Name of benchmark to run
        operator_name: Name of operator to benchmark
        epochs: Number of training epochs
        batch_size: Training batch size
        seed: Random seed
        output_file: Optional output file for JSON results

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    from opifex.benchmarking.benchmark_runner import BenchmarkRunner

    registry = _get_registry()

    # Validate benchmark exists
    if benchmark_name not in registry.list_available_benchmarks():
        print(f"Error: Benchmark '{benchmark_name}' not found.", file=sys.stderr)
        print("Use --list-benchmarks to see available benchmarks.", file=sys.stderr)
        return 1

    # Validate operator exists
    if operator_name not in registry.list_available_operators():
        print(f"Error: Operator '{operator_name}' not found.", file=sys.stderr)
        print("Use --list-operators to see available operators.", file=sys.stderr)
        return 1

    # Create runner with our registry
    runner = BenchmarkRunner()
    runner.registry = registry

    # Get benchmark config
    benchmark_config = registry.get_benchmark_config(benchmark_name)

    # Override config with CLI args
    benchmark_config.computational_requirements["n_epochs"] = epochs
    benchmark_config.computational_requirements["batch_size"] = batch_size
    benchmark_config.computational_requirements["seed"] = seed

    print(f"Running benchmark: {benchmark_name}")
    print(f"Operator: {operator_name}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Seed: {seed}")
    print("-" * 40)

    try:
        result = runner._run_single_benchmark(operator_name, benchmark_config)
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        return 1

    # Print results
    print("\nResults:")
    print(f"  Model: {result.model_name}")
    print(f"  Dataset: {result.dataset_name}")
    print(f"  Execution time: {result.execution_time:.2f}s")
    print("\n  Metrics:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")

    # Save to file if requested
    if output_file:
        result_dict = {
            "model_name": result.model_name,
            "dataset_name": result.dataset_name,
            "execution_time": result.execution_time,
            "metrics": result.metrics,
            "framework_version": result.framework_version,
        }
        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return 0


def run_cli(args: Sequence[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success)
    """
    parsed = parse_args(args)

    # Handle list commands
    if parsed.list_benchmarks:
        _list_benchmarks()
        sys.exit(0)

    if parsed.list_operators:
        _list_operators()
        sys.exit(0)

    # Validate required args for benchmark run
    if not parsed.benchmark or not parsed.operator:
        print(
            "Error: --benchmark (-b) and --operator (-o) are required.",
            file=sys.stderr,
        )
        print(
            "Use --list-benchmarks and --list-operators to see options.",
            file=sys.stderr,
        )
        sys.exit(2)

    return _run_benchmark(
        benchmark_name=parsed.benchmark,
        operator_name=parsed.operator,
        epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        seed=parsed.seed,
        output_file=parsed.output,
    )


def main() -> None:
    """Main entry point for module execution."""
    sys.exit(run_cli())


if __name__ == "__main__":
    main()

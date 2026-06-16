# Opifex Benchmarking: Evaluation & Validation Framework

This package provides benchmarking infrastructure built into the framework core,
enabling evaluation and validation of Opifex methods, including Neural DFT
chemical-accuracy validation. It supports statistical analysis with bootstrap
confidence intervals and significance testing, multi-model comparison across FNO,
DeepONet, and custom neural operators, and JAX-JIT / GPU-accelerated execution.

## Benchmarking Components

### 1. BenchmarkRegistry (`benchmark_registry.py`)

Domain-specific configuration management with physics-aware settings:

```python
from opifex.benchmarking import BenchmarkRegistry

# Initialize registry with domain-specific configurations
registry = BenchmarkRegistry()

# Register domain-specific benchmarks
registry.register_benchmark(
    name="darcy_flow_fno",
    domain="fluid_dynamics",
    operator_type="FNO",
    input_resolution=(64, 64),
    target_accuracy=0.01
)

# Auto-discover neural operators with domain validation
operators = registry.discover_operators("fluid_dynamics")
benchmark_suite = registry.get_benchmark_suite("quantum_computing")
```

### 2. ValidationFramework (`validation_framework.py`)

Domain-specific validation with tolerance checking for scientific computing:

```python
from opifex.benchmarking import ValidationFramework

# Initialize validation with domain-specific tolerances
validator = ValidationFramework(domain="quantum_computing")

# Chemical accuracy assessment for quantum chemistry
chemical_accuracy = validator.assess_chemical_accuracy(
    predicted_energy=-76.4,
    reference_energy=-76.42,
    tolerance_kcal_mol=1.0
)

# Physics-informed validation
physics_compliance = validator.validate_conservation_laws(
    results, conservation_type="energy"
)
```

### 3. AnalysisEngine (`analysis_engine.py`)

Performance analysis with statistical insights and operator comparison:

```python
from opifex.benchmarking import AnalysisEngine

# Initialize analysis engine
analyzer = AnalysisEngine()

# Multi-operator performance comparison
comparison = analyzer.compare_operators(
    operators=["FNO", "DeepONet", "PINN"],
    benchmark_results=results,
    statistical_significance=True
)

# Scaling behavior analysis
scaling_analysis = analyzer.analyze_scaling_behavior(
    operator="FNO",
    resolutions=[(32, 32), (64, 64), (128, 128)],
    performance_metrics=["accuracy", "inference_time"]
)

# Performance insights generation
insights = analyzer.generate_performance_insights(
    results, domain="fluid_dynamics"
)
```

### 4. ResultsManager (`results_manager.py`)

Database persistence and publication capabilities:

```python
from opifex.benchmarking import ResultsManager

# Initialize results manager with database
manager = ResultsManager(storage_path="./benchmark_results")

# Store benchmark results with metadata
manager.store_results(
    benchmark_name="darcy_flow_comparison",
    results=benchmark_results,
    metadata={"domain": "fluid_dynamics", "timestamp": "2025-02-09"}
)

# Generate publication-ready plots
plots = manager.export_publication_plots(
    results, plot_type="comparison", format="png"
)

# Generate LaTeX/HTML tables
latex_table = manager.generate_comparison_tables(
    operators=["FNO", "DeepONet"],
    metrics=["accuracy", "speed"],
    format="latex"
)
```

### 5. BenchmarkRunner (`benchmark_runner.py`)

Orchestration engine with end-to-end workflow:

```python
from opifex.benchmarking import BenchmarkRunner

# Initialize runner with all components
runner = BenchmarkRunner(
    registry=registry,
    validator=validator,
    analyzer=analyzer,
    results_manager=manager
)

# Run domain-specific benchmark suite
results = runner.run_benchmark_suite(
    domain="fluid_dynamics",
    operators=["FNO", "DeepONet"],
    datasets=["darcy_flow", "navier_stokes"]
)

# Generate publication report
report = runner.generate_publication_report(
    results, output_format="pdf"
)
```

## Evaluation Foundation

- **`evaluation_engine.py`**: Evaluation infrastructure with JAX-JIT optimization.
- **`pdebench_integration.py`** / **`pdebench_configs.py`**: PDEBench-compatible
  data handling and benchmark registration.
- **`baseline_repository.py`**: Reference implementations and performance tracking.
- **`report_generator.py`**: Report generation.
- **`visualization_tools.py`**: Plotting and visualization.
- **`cli.py`**: Command-line entry point (`run_benchmark_cli`).

## Key Features

- **Statistical Analysis**: Bootstrap confidence intervals and permutation testing.
- **Multi-Model Support**: FNO, DeepONet, and custom neural operators with batch
  processing.
- **Performance Profiling**: Multi-run execution timing and memory tracking.
- **GPU Acceleration**: CUDA support with JAX-JIT optimization.
- **Result Management**: JSON serialization with automatic timestamp management.
- **Publication Output**: LaTeX table and figure generation.
- **Chemical Accuracy Validation**: Energy-accuracy benchmarking for Neural DFT.

## Dependencies

- **JAX**: Core array operations and JIT-accelerated evaluation.
- **calibrax**: `BenchmarkResult` and `StatisticalAnalyzer` primitives.
- **Python 3.11+**: Modern Python features and type system.
</content>

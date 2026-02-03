# Benchmarking System

The Opifex framework includes a comprehensive benchmarking system designed specifically for scientific machine learning applications. This system provides domain-specific evaluation, publication-ready output, and enterprise-grade reliability.

## Overview

The Benchmarking System consists of 6 specialized components that work together to provide comprehensive evaluation of scientific machine learning models:

1. **BenchmarkRegistry** - Domain-specific configuration management
2. **ValidationFramework** - Physics-aware validation and accuracy assessment
3. **AnalysisEngine** - Statistical analysis and performance comparison
4. **ResultsManager** - Database persistence and publication output
5. **BenchmarkRunner** - End-to-end workflow orchestration
6. **Comprehensive Test Suite** - 25 tests ensuring reliability

## Key Features

### Domain-Specific Intelligence

- **Physics-aware validation** for quantum chemistry, fluid dynamics, and materials science
- **Chemical accuracy assessment** with <1 kcal/mol tolerance for quantum applications
- **Conservation law validation** for energy, momentum, and mass conservation
- **Domain-specific metrics** tailored to scientific computing requirements

### Publication-Ready Output

- **LaTeX table generation** for academic papers
- **HTML report generation** for web-based sharing
- **CSV export** for data analysis
- **Publication-quality plots** with matplotlib integration
- **Automated figure generation** with comparison visualizations

### Statistical Rigor

- **Bootstrap confidence intervals** for robust uncertainty quantification
- **Permutation significance testing** for statistical validation
- **Multi-operator comparison** with significance testing
- **Scaling behavior analysis** across different problem sizes

### Enterprise Reliability

- **25/25 tests passing** with comprehensive validation
- **Database isolation** for reliable benchmarking
- **Perfect pre-commit compliance** with zero errors
- **Production-ready architecture** with modular design

## Quick Start

### Basic Usage

```python
from opifex.benchmarking import (
    BenchmarkRegistry, ValidationFramework, AnalysisEngine,
    ResultsManager, BenchmarkRunner
)

# Initialize components
registry = BenchmarkRegistry()
validator = ValidationFramework(domain="fluid_dynamics")
analyzer = AnalysisEngine()
manager = ResultsManager(storage_path="./benchmark_results")

# Create runner
runner = BenchmarkRunner(
    registry=registry,
    validator=validator,
    analyzer=analyzer,
    results_manager=manager
)

# Run benchmark suite
results = runner.run_benchmark_suite(
    domain="fluid_dynamics",
    operators=["FNO", "DeepONet"],
    datasets=["darcy_flow", "navier_stokes"]
)

# Generate publication report
report = runner.generate_publication_report(results)
```

### Domain-Specific Benchmarking

```python
# Quantum chemistry benchmarking
quantum_validator = ValidationFramework(domain="quantum_computing")

# Chemical accuracy assessment
chemical_accuracy = quantum_validator.assess_chemical_accuracy(
    predicted_energy=-76.4,
    reference_energy=-76.42,
    tolerance_kcal_mol=1.0
)

# Fluid dynamics benchmarking
fluid_validator = ValidationFramework(domain="fluid_dynamics")

# Conservation law validation
conservation_check = fluid_validator.validate_conservation_laws(
    results, conservation_type="energy"
)
```

## Component Details

### BenchmarkRegistry

The BenchmarkRegistry manages domain-specific configurations and operator discovery:

```python
from opifex.benchmarking import BenchmarkRegistry

registry = BenchmarkRegistry()

# Register domain-specific benchmark
registry.register_benchmark(
    name="darcy_flow_fno",
    domain="fluid_dynamics",
    operator_type="FNO",
    input_resolution=(64, 64),
    target_accuracy=0.01
)

# Auto-discover operators
operators = registry.discover_operators("fluid_dynamics")

# Get benchmark suite
suite = registry.get_benchmark_suite("quantum_computing")
```

**Key Features:**

- Domain-specific configuration management
- Automatic operator discovery
- Benchmark suite generation
- Compatibility checking
- JSON persistence

### ValidationFramework

The ValidationFramework provides physics-aware validation:

```python
from opifex.benchmarking import ValidationFramework

# Initialize with domain
validator = ValidationFramework(domain="quantum_computing")

# Chemical accuracy assessment
accuracy = validator.assess_chemical_accuracy(
    predicted_energy=-76.4,
    reference_energy=-76.42,
    tolerance_kcal_mol=1.0
)

# Physics compliance validation
compliance = validator.validate_conservation_laws(
    results, conservation_type="energy"
)

# Convergence analysis
convergence = validator.analyze_convergence(
    training_history, tolerance=1e-6
)
```

**Key Features:**

- Chemical accuracy assessment
- Conservation law validation
- Convergence analysis
- Domain-specific tolerances
- Physics-informed validation

### AnalysisEngine

The AnalysisEngine provides statistical analysis and performance comparison:

```python
from opifex.benchmarking import AnalysisEngine

analyzer = AnalysisEngine()

# Multi-operator comparison
comparison = analyzer.compare_operators(
    operators=["FNO", "DeepONet", "PINN"],
    benchmark_results=results,
    statistical_significance=True
)

# Scaling behavior analysis
scaling = analyzer.analyze_scaling_behavior(
    operator="FNO",
    resolutions=[(32, 32), (64, 64), (128, 128)],
    performance_metrics=["accuracy", "inference_time"]
)

# Performance insights
insights = analyzer.generate_performance_insights(
    results, domain="fluid_dynamics"
)
```

**Key Features:**

- Multi-operator performance comparison
- Statistical significance testing
- Scaling behavior analysis
- Performance insights generation
- Bootstrap confidence intervals

### ResultsManager

The ResultsManager handles database persistence and publication output:

```python
from opifex.benchmarking import ResultsManager

manager = ResultsManager(storage_path="./benchmark_results")

# Store results
manager.store_results(
    benchmark_name="darcy_flow_comparison",
    results=benchmark_results,
    metadata={"domain": "fluid_dynamics"}
)

# Generate publication plots
plots = manager.export_publication_plots(
    results, plot_type="comparison", format="png"
)

# Generate LaTeX tables
latex_table = manager.generate_comparison_tables(
    operators=["FNO", "DeepONet"],
    metrics=["accuracy", "speed"],
    format="latex"
)

# Query database
query_results = manager.query_results(
    domain="fluid_dynamics",
    operator_type="FNO",
    min_accuracy=0.95
)
```

**Key Features:**

- JSON-based database persistence
- Publication-quality plot generation
- LaTeX/HTML table generation
- Query functionality
- Statistics computation

### BenchmarkRunner

The BenchmarkRunner orchestrates end-to-end benchmarking workflows:

```python
from opifex.benchmarking import BenchmarkRunner

runner = BenchmarkRunner(
    registry=registry,
    validator=validator,
    analyzer=analyzer,
    results_manager=manager
)

# Run domain-specific suite
results = runner.run_benchmark_suite(
    domain="fluid_dynamics",
    operators=["FNO", "DeepONet"],
    datasets=["darcy_flow", "navier_stokes"]
)

# Generate comprehensive report
report = runner.generate_publication_report(
    results, output_format="pdf"
)

# Run single benchmark
single_result = runner.run_single_benchmark(
    operator="FNO",
    dataset="darcy_flow",
    config=benchmark_config
)
```

**Key Features:**

- End-to-end workflow orchestration
- Component integration
- Error handling and recovery
- Publication report generation
- Comprehensive validation

## Usage

### Custom Domain Configuration

```python
# Define custom domain
custom_domain = {
    "name": "custom_physics",
    "validation_tolerances": {
        "energy_conservation": 1e-6,
        "momentum_conservation": 1e-5,
        "mass_conservation": 1e-7
    },
    "accuracy_metrics": ["l2_error", "max_error", "physics_residual"],
    "reference_methods": ["analytical", "high_fidelity_simulation"]
}

# Register custom domain
registry.register_domain(custom_domain)

# Use custom validation
validator = ValidationFramework(domain="custom_physics")
```

### Statistical Analysis

```python
# Bootstrap confidence intervals
confidence_intervals = analyzer.compute_bootstrap_confidence_intervals(
    results, confidence_level=0.95, n_bootstrap=1000
)

# Permutation significance test
p_value = analyzer.permutation_significance_test(
    results_a, results_b, n_permutations=10000
)

# Effect size analysis
effect_size = analyzer.compute_effect_size(
    results_a, results_b, method="cohen_d"
)
```

### Publication Output

```python
# Generate comprehensive publication package
publication_package = runner.generate_publication_package(
    results,
    include_plots=True,
    include_tables=True,
    include_statistics=True,
    output_format="latex"
)

# Custom plot generation
custom_plots = manager.generate_custom_plots(
    results,
    plot_configs=[
        {"type": "accuracy_vs_resolution", "operators": ["FNO", "DeepONet"]},
        {"type": "scaling_behavior", "metric": "inference_time"},
        {"type": "error_distribution", "operator": "FNO"}
    ]
)
```

## Testing and Validation

The benchmarking system includes comprehensive testing:

```bash
# Run all benchmarking tests
uv run pytest tests/benchmarking/ -v

# Run specific component tests
uv run pytest tests/benchmarking/test_benchmark_registry.py -v
uv run pytest tests/benchmarking/test_validation_framework.py -v
uv run pytest tests/benchmarking/test_analysis_engine.py -v
uv run pytest tests/benchmarking/test_results_manager.py -v
uv run pytest tests/benchmarking/test_benchmark_runner.py -v

# Run integration tests
uv run pytest tests/benchmarking/test_integration.py -v
```

**Test Coverage:**

- Component unit tests with database isolation
- Integration tests with end-to-end workflows
- Performance tests with timing validation
- Error handling tests with recovery scenarios

## Best Practices

### Database Management

- Use unique database paths for different benchmark runs
- Implement proper cleanup in test environments
- Use transaction-based operations for data integrity
- Regular backup of benchmark databases

### Statistical Analysis

- Use appropriate sample sizes for statistical tests
- Apply multiple comparison corrections when needed
- Report confidence intervals alongside point estimates
- Validate assumptions before applying statistical tests

### Publication Output

- Follow journal-specific formatting requirements
- Include comprehensive metadata in tables
- Use consistent color schemes across figures
- Provide clear captions and legends

### Performance Optimization

- Use JAX-JIT compilation for computational kernels
- Implement efficient database queries
- Cache frequently accessed results
- Use parallel processing for independent benchmarks

## Troubleshooting

### Common Issues

1. **Database Lock Errors**

    ```python
    # Solution: Use unique database paths
    manager = ResultsManager(
        storage_path="./benchmark_results",
        database_path="./benchmark_results/unique_db.json"
    )
    ```

2. **Memory Issues with Large Datasets**

    ```python
    # Solution: Use batch processing
    runner.run_benchmark_suite(
        domain="fluid_dynamics",
        operators=["FNO"],
        datasets=["darcy_flow"],
        batch_size=32
    )
    ```

3. **Statistical Test Failures**

    ```python
    # Solution: Check sample sizes and assumptions
    if len(results) < 30:
        print("Warning: Small sample size may affect statistical tests")
    ```

### Performance Optimization

```python
# Enable JAX-JIT compilation
import jax
jax.config.update("jax_enable_x64", True)

# Use efficient data structures
results = manager.query_results(
    domain="fluid_dynamics",
    use_cache=True,
    optimize_queries=True
)
```

## Integration with Other Components

### MLOps Integration

```python
from opifex.mlops import ExperimentTracker
from opifex.benchmarking import BenchmarkRunner

# Combine benchmarking with experiment tracking
tracker = ExperimentTracker.create(backend="mlflow")
runner = BenchmarkRunner(...)

with tracker.start_run():
    results = runner.run_benchmark_suite(...)
    tracker.log_metrics(results.summary_metrics)
    tracker.log_artifacts(results.publication_plots)
```

### Neural Operator Integration

```python
from opifex.neural.operators import FNO, DeepONet
from opifex.benchmarking import BenchmarkRunner

# Benchmark neural operators
operators = {
    "FNO": FNO(modes=12, width=64),
    "DeepONet": DeepONet(branch_net=branch, trunk_net=trunk)
}

results = runner.benchmark_operators(
    operators=operators,
    datasets=["darcy_flow", "navier_stokes"]
)
```

## Future Features

### Planned Features

- **Automated hyperparameter optimization** for benchmark configurations
- **Multi-GPU benchmarking** for large-scale experiments
- **Real-time benchmarking** with streaming results
- **Federated benchmarking** across multiple institutions
- **Interactive dashboards** for result exploration

### Research Directions

- **Uncertainty-aware benchmarking** with probabilistic metrics
- **Transfer learning evaluation** across different domains
- **Robustness testing** with adversarial examples
- **Fairness evaluation** for scientific applications

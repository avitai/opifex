# Benchmarking System

The Opifex framework includes a benchmarking system designed specifically for scientific machine learning applications. This system provides domain-specific evaluation, publication-ready output, and statistical rigor.

## Overview

The Benchmarking System consists of 8+ specialized components that work together to provide evaluation of scientific machine learning models:

1. **BenchmarkRegistry** - Domain-specific configuration management
2. **ValidationFramework** - Reference comparison, convergence analysis, and error analysis
3. **ChemicalAccuracyValidator** - Chemical accuracy assessment with domain-specific thresholds
4. **ConservationValidator** - Physics conservation law validation
5. **AnalysisEngine** - Statistical analysis and performance comparison
6. **ResultsManager** - JSON persistence and publication output
7. **BenchmarkRunner** - End-to-end workflow orchestration
8. **Adapters** - Bridge to calibrax `Run` objects for cross-tool analysis

Core types (`BenchmarkResult`, `Metric`, `Run`) and statistical analysis (`StatisticalAnalyzer`) are provided by [calibrax](https://pypi.org/project/calibrax/).

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

- **Welch t-test and Mann-Whitney U** via calibrax for significance testing
- **Multi-operator comparison** with per-metric rankings
- **Scaling behavior analysis** across different problem sizes
- **Performance insights** with bottleneck detection

### Enterprise Reliability

- **Database isolation** for reliable benchmarking
- **Pre-commit compliance** with zero errors
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
validator = ValidationFramework()
analyzer = AnalysisEngine()
manager = ResultsManager(storage_path="./benchmark_results")

# Create runner with all components
runner = BenchmarkRunner(
    registry=registry,
    validator=validator,
    analyzer=analyzer,
    results_manager=manager,
    output_dir="./benchmark_results",
)

# Run benchmark suite
results = runner.run_comprehensive_benchmark(
    operators=["FNO", "DeepONet"],
)

# Generate publication report
report = runner.generate_publication_report(results)
```

### Domain-Specific Benchmarking

```python
from opifex.benchmarking.validators.chemical_accuracy import ChemicalAccuracyValidator
from opifex.benchmarking.validators.conservation import ConservationValidator

# Quantum chemistry — chemical accuracy assessment
chem_validator = ChemicalAccuracyValidator()
assessment = chem_validator.assess(result, domain="quantum_computing")
print(f"Passed: {assessment.passed}, Achieved: {assessment.achieved:.4f}")

# Fluid dynamics — conservation law validation
conservation = ConservationValidator(laws=["energy", "momentum"])
report = conservation.validate(y_pred, y_true)
print(f"All conserved: {report.all_conserved}")
```

## Component Details

### BenchmarkResult (from calibrax)

`BenchmarkResult` is the central data container for all benchmark outputs:

```python
from calibrax.core import BenchmarkResult, Metric

result = BenchmarkResult(
    name="darcy_flow_fno",
    domain="scientific_ml",
    tags={"dataset": "darcy_flow", "operator": "FNO"},
    metrics={
        "mse": Metric(value=0.0012),
        "relative_error": Metric(value=0.034, lower=0.029, upper=0.041),
    },
    metadata={
        "execution_time": 1.23,
        "framework_version": "1.0.0",
    },
)

# Access fields
print(result.metrics["mse"].value)          # 0.0012
print(result.metadata["execution_time"])     # 1.23
print(result.tags["dataset"])                # "darcy_flow"
```

### BenchmarkRegistry

The BenchmarkRegistry manages domain-specific configurations and operator discovery:

```python
from opifex.benchmarking import BenchmarkRegistry
from opifex.benchmarking.benchmark_registry import BenchmarkConfig

registry = BenchmarkRegistry()

# Register domain-specific benchmark
config = BenchmarkConfig(
    name="darcy_flow_fno",
    domain="fluid_dynamics",
    problem_type="elliptic_pde",
    input_shape=(64, 64, 1),
    output_shape=(64, 64, 1),
)
registry.register_benchmark(config)

# Auto-discover operators
registry.auto_discover_operators()

# Get benchmark suite for a domain
suite = registry.get_benchmark_suite("quantum_computing")
```

**Key Features:**

- Domain-specific configuration management
- Automatic operator discovery
- Benchmark suite generation per domain
- Compatibility checking
- JSON persistence

### ValidationFramework

The ValidationFramework provides reference comparison and convergence analysis:

```python
from opifex.benchmarking import ValidationFramework

# Initialize (no domain parameter — domain is inferred from results)
validator = ValidationFramework(
    default_tolerances=[1e-3, 1e-4, 1e-5],
    reference_methods={"analytical": analytical_solver},
)

# Validate against a reference method
report = validator.validate_against_reference(
    result=benchmark_result,
    reference_method="analytical",
    reference_data=reference_array,
    predictions=pred_array,
)

# Check convergence rates across a sequence of results
convergence = validator.check_convergence_rates(
    results_sequence=[result_32, result_64, result_128],
    tolerances=[1e-3, 1e-4, 1e-5],
)

# Generate detailed error analysis
error_analysis = validator.generate_error_analysis(
    predictions=pred_array,
    ground_truth=truth_array,
)
```

**Key Features:**

- Reference method comparison with pluggable solvers
- Convergence rate analysis across resolution sequences
- Detailed error analysis with spatial/temporal patterns
- Chemical accuracy assessment (delegates to `ChemicalAccuracyValidator` for detailed analysis)

### AnalysisEngine

The AnalysisEngine provides statistical analysis and performance comparison:

```python
from opifex.benchmarking import AnalysisEngine

analyzer = AnalysisEngine(significance_threshold=0.05)

# Multi-operator comparison (single run per operator)
comparison = analyzer.compare_operators(
    results_dict={"FNO": fno_result, "DeepONet": deeponet_result, "PINN": pinn_result}
)
print(f"Overall winner: {comparison.overall_winner}")
print(f"Rankings: {comparison.performance_rankings}")

# Multi-run statistical significance testing
significance = analyzer.test_statistical_significance_multi_run(
    multi_run_results={
        "FNO": [fno_run1, fno_run2, fno_run3],
        "DeepONet": [don_run1, don_run2, don_run3],
    }
)

# Scaling behavior analysis
scaling = analyzer.analyze_scaling_behavior(
    performance_data={32: result_32, 64: result_64, 128: result_128}
)
print(f"Complexity estimates: {scaling.complexity_estimates}")

# Performance insights for a single result
insights = analyzer.generate_performance_insights(result=fno_result)
print(f"Key insights: {insights.key_insights}")
print(f"Bottlenecks: {insights.performance_bottlenecks}")
```

**Key Features:**

- Multi-operator performance comparison with per-metric rankings
- Statistical significance testing via calibrax (Welch t-test, Mann-Whitney U)
- Scaling behavior analysis with complexity estimation
- Performance insights with bottleneck detection
- Operator recommendations by problem type and domain

### ResultsManager

The ResultsManager handles JSON persistence and publication output:

```python
from opifex.benchmarking import ResultsManager

manager = ResultsManager(storage_path="./benchmark_results")

# Save results
result_id = manager.save_benchmark_results(result)

# Load a specific result
loaded = manager.load_result(result_id)

# Query stored results
matching = manager.query_results(
    name="darcy",
    dataset="darcy_flow",
    metric_filter={"mse": (0.0, 0.01)},
)

# Generate publication plots
plots = manager.export_publication_plots(
    results=[result1, result2],
    plot_type="comparison",
    output_format="png",
)

# Generate LaTeX tables
table_path = manager.generate_comparison_tables(
    operators=["FNO", "DeepONet"],
    metrics=["mse", "relative_error"],
    output_format="latex",
)
```

**Key Features:**

- JSON-based database persistence
- Publication-quality plot generation
- LaTeX/HTML/CSV table generation
- Query functionality with metric filtering
- Database statistics and export

### BenchmarkRunner

The BenchmarkRunner orchestrates end-to-end benchmarking workflows:

```python
from opifex.benchmarking import BenchmarkRunner

runner = BenchmarkRunner(
    registry=registry,
    validator=validator,
    analyzer=analyzer,
    results_manager=manager,
    output_dir="./benchmark_results",
)

# Run full benchmark suite
results = runner.run_comprehensive_benchmark(
    operators=["FNO", "DeepONet"],
    benchmarks=["darcy_flow", "navier_stokes"],
    validate_results=True,
    generate_analysis=True,
)

# Run domain-specific suite
domain_results = runner.execute_domain_specific_suite(domain="fluid_dynamics")

# Generate publication report
report = runner.generate_publication_report(
    results=results,
    title="Neural Operator Comparison on Fluid Dynamics",
)
print(f"Key findings: {report.key_findings}")
print(f"Tables: {report.comparison_tables}")
```

**Key Features:**

- End-to-end workflow orchestration
- Component integration with registry, validator, analyzer, results manager
- Domain-specific suite execution
- Publication report generation (`PublicationReport` dataclass)
- Database update functionality

### Adapters

The adapters module bridges opifex `BenchmarkResult` objects to calibrax `Run` objects:

```python
from opifex.benchmarking.adapters import results_to_run, default_metric_defs

# Convert benchmark results to a calibrax Run for cross-tool analysis
run = results_to_run(
    results=[result1, result2, result3],
    commit="abc123",
    branch="main",
    metric_defs=default_metric_defs(),
)
```

### Profiling

The profiling subsystem delegates hardware detection, roofline analysis, FLOPS counting, and compilation profiling to calibrax while providing an opifex-specific harness and event coordinator:

```python
from opifex.benchmarking.profiling import (
    OpifexProfilingHarness,
    EventCoordinator,
    # From calibrax:
    CompilationProfiler,
    FlopsCounter,
    ResourceMonitor,
    RooflineAnalyzer,
    detect_hardware_specs,
    analyze_complexity,
)

# Profile a neural operator
harness = OpifexProfilingHarness(
    enable_hardware_profiling=True,
    enable_roofline_analysis=True,
)

with harness.profiling_session():
    metrics, report = harness.profile_neural_operator(
        operator=fno_model,
        inputs=[input_array],
        operation_name="FNO forward pass",
    )
    print(report.render())
```

## Usage

### Custom Domain Configuration

```python
from opifex.benchmarking.benchmark_registry import DomainConfig

# Define custom domain with specific tolerances and metrics
config = DomainConfig(
    name="custom_physics",
    tolerance_ranges={
        "energy_conservation": (1e-7, 1e-5),
        "momentum_conservation": (1e-6, 1e-4),
    },
    required_metrics=["l2_error", "max_error", "physics_residual"],
    reference_methods=["analytical", "high_fidelity_simulation"],
)
```

### Statistical Analysis

```python
# Multi-run significance testing delegates to calibrax
significance = analyzer.test_statistical_significance_multi_run(
    multi_run_results={
        "FNO": fno_runs,
        "DeepONet": deeponet_runs,
    }
)

# Results include Welch t-test and Mann-Whitney U per metric pair
for pair, metrics in significance.items():
    for metric_name, stats in metrics.items():
        print(f"{pair} / {metric_name}: p={stats.get('p_value', 'N/A')}")
```

### Publication Output

```python
# Generate publication report with tables and figures
report = runner.generate_publication_report(
    results=results,
    title="Neural Operator Benchmark Results",
)

# Access report fields
print(report.abstract)
print(report.methodology)
for finding in report.key_findings:
    print(f"  - {finding}")
for table in report.comparison_tables:
    print(f"  Table: {table}")
```

## Testing and Validation

The benchmarking system includes testing across all components:

```bash
# Run all benchmarking tests
uv run pytest tests/benchmarking/ -v

# Run specific component tests
uv run pytest tests/benchmarking/test_benchmark_registry.py -v
uv run pytest tests/benchmarking/test_validation_framework.py -v
uv run pytest tests/benchmarking/test_analysis_engine.py -v
uv run pytest tests/benchmarking/test_adapters.py -v
uv run pytest tests/benchmarking/test_baseline_repository.py -v
uv run pytest tests/benchmarking/test_chemical_accuracy_validator.py -v
uv run pytest tests/benchmarking/test_conservation_validator.py -v
uv run pytest tests/benchmarking/test_operator_execution.py -v
```

**Test Coverage:**

- Component unit tests with database isolation
- Integration tests with end-to-end workflows
- Performance tests with timing validation
- Error handling tests with recovery scenarios

## Best Practices

### Database Management

- Use unique storage paths for different benchmark runs
- Implement proper cleanup in test environments
- Use the `ResultsManager` query API to find past results before re-running

### Statistical Analysis

- Use appropriate sample sizes for statistical tests
- Apply multiple comparison corrections when needed
- Report confidence intervals alongside point estimates
- Validate assumptions before applying statistical tests

### Publication Output

- Follow journal-specific formatting requirements
- Include metadata in tables via `ResultsManager.generate_comparison_tables()`
- Use consistent color schemes across figures
- Provide clear captions and legends

### Performance Optimization

- Use JAX-JIT compilation for computational kernels
- Cache frequently accessed results
- Use parallel processing for independent benchmarks

## Troubleshooting

### Common Issues

1. **Storage Path Errors**

    ```python
    # Use unique storage paths per experiment
    manager = ResultsManager(
        storage_path="./benchmark_results/experiment_001",
    )
    ```

2. **Memory Issues with Large Datasets**

    ```python
    # Use batch processing via the evaluator
    evaluator = BenchmarkEvaluator(output_dir="./results")
    # Evaluate in smaller batches
    for batch_x, batch_y in batched_data:
        result = evaluator.evaluate_model(
            model=model_fn, model_name="FNO",
            input_data=batch_x, target_data=batch_y,
            dataset_name="darcy_flow",
        )
    ```

3. **Statistical Test Failures**

    ```python
    # Check sample sizes before multi-run significance testing
    if all(len(runs) >= 3 for runs in multi_run_results.values()):
        significance = analyzer.test_statistical_significance_multi_run(
            multi_run_results
        )
    ```

### Performance Optimization

```python
# Enable JAX-JIT compilation
import jax
jax.config.update("jax_enable_x64", True)

# Query results efficiently with filters
results = manager.query_results(
    name="darcy",
    metric_filter={"mse": (0.0, 0.01)},
)
```

## Future Features

### Planned Features

- **Automated hyperparameter optimization** for benchmark configurations
- **Multi-GPU benchmarking** for large-scale experiments
- **Real-time benchmarking** with streaming results
- **Interactive dashboards** for result exploration

### Research Directions

- **Uncertainty-aware benchmarking** with probabilistic metrics
- **Transfer learning evaluation** across different domains
- **Robustness testing** with adversarial examples

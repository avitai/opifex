# Optimization API Reference

## Overview

The Opifex optimization module provides full optimization algorithms and meta-learning approaches for scientific computing, including production optimization, learn-to-optimize algorithms, control systems, and quantum-aware optimization.

## Core Optimization Components

### Meta-Optimization Framework

Advanced meta-optimization algorithms that learn to optimize across families of related problems.

::: opifex.optimization.meta_optimization

### Production Optimization

Optimization systems for deployment and scaling in production environments.

::: opifex.optimization.production

### Scientific Computing Integration

Physics-aware optimization with scientific validation and benchmarking.

::: opifex.optimization.scientific_integration

## Learn-to-Optimize (L2O) Algorithms

Advanced neural optimization methods that achieve significant speedups on learned problem families.

### Tasks and the optimiser interface

Objective-carrying `Task`/`TaskFamily` abstractions and the shared stateful `Optimizer`
interface (with the optax-wrapped baseline family).

::: opifex.optimization.l2o.core

::: opifex.optimization.l2o.optimizers

### Concrete tasks

`QuadraticTaskFamily` (convex smoke task) and `MLPTaskFamily` (the non-convex small-MLP
training showcase task).

::: opifex.optimization.l2o.tasks

### Learned optimisers

Coordinatewise learned optimisers (the per-parameter MLP of Metz et al. 2020) and their
input features.

::: opifex.optimization.l2o.features

::: opifex.optimization.l2o.learned

### PES meta-training

Persistent Evolution Strategies meta-training (Vicol et al. 2021).

::: opifex.optimization.l2o.meta_train

### Baselines and benchmarking

optimistix classical baselines and honest learning-curve / speedup-at-target benchmarking.

::: opifex.optimization.l2o.baselines

::: opifex.optimization.l2o.benchmark

### L2O Engine

High-level orchestrator: meta-train a learned optimiser on a task family, apply it,
benchmark it honestly, and persist `theta`.

::: opifex.optimization.l2o.engine

## Control Systems

Differentiable predictive control components for scientific machine learning.

### System Identification

Neural networks that learn system dynamics from data.

::: opifex.optimization.control.system_id

### Model Predictive Control

Differentiable MPC frameworks with safety guarantees.

::: opifex.optimization.control.mpc

## Module Overview

The optimization module is organized into several key components:

### Core Components

- **`meta_optimization/`**: Meta-optimization framework with L2O algorithms (modular package)
- **`production.py`**: Production optimization (adaptive JIT, GPU memory planning)
- **`scientific_integration.py`**: Physics-aware optimization integration

### L2O Submodule (`l2o/`)

- **`core.py`**: `Task`/`TaskFamily` (objective-carrying) and the `Optimizer` interface
- **`optimizers.py`**: `Optimizer` ABC + `OptaxOptimizer` (hand-designed baseline family)
- **`tasks.py`**: `QuadraticTaskFamily` and the `MLPTaskFamily` showcase task
- **`features.py`**: per-parameter input features (momentum/RMS, time embedding)
- **`learned.py`**: `LearnedOptimizer` ABC, `MLPLearnedOptimizer`, `LearnableSGD`
- **`meta_train.py`**: Persistent Evolution Strategies (PES) meta-training
- **`baselines.py`**: optimistix classical baselines and tuned-optax baselines
- **`benchmark.py`**: honest learning-curve and speedup-at-target benchmarking
- **`engine.py`**: high-level `L2OEngine` orchestrator

### Control Submodule (`control/`)

- **`system_id.py`**: System identification networks
- **`mpc.py`**: Model predictive control frameworks

## Key Features

### Meta-Optimization Features

- Learn-to-Optimize (L2O) algorithms with meta-learned update rules
- Adaptive learning rate scheduling
- Warm-starting strategies for related problems

### Production Optimization Features

- Hybrid performance platform with adaptive JIT
- Intelligent GPU memory management
- Physics-aware scientific validation of optimized models

### Control Systems Features

- Differentiable model predictive control
- Physics-constrained system identification
- Safety-critical control with barrier functions
- Real-time optimization capabilities

### Scientific Integration Features

- Physics-informed optimization
- Conservation law enforcement
- Numerical validation and stability checks
- Domain-specific profiling and benchmarking

## Usage Examples

### Basic Meta-Optimization

```python
from opifex.optimization.meta_optimization import LearnToOptimize, MetaOptimizerConfig

config = MetaOptimizerConfig(
    meta_learning_rate=1e-4,
    adaptation_steps=5,
    warm_start_strategy="previous_params"
)

l2o = LearnToOptimize(config=config, rngs=nnx.Rngs(42))
optimized_params = l2o.optimize(params, objective_fn, num_steps=1000)
```

### Production Optimization

```python
from opifex.optimization.production import HybridPerformancePlatform, WorkloadProfile

platform = HybridPerformancePlatform()

workload = WorkloadProfile(
    batch_size=32,
    sequence_length=128,
    memory_footprint=2.0,
    compute_intensity=8.0,
    latency_requirement=10.0,
    throughput_requirement=100.0,
    model_complexity="medium",
)

optimized = platform.optimize_for_production(model, workload)
```

### Control System

```python
from opifex.optimization.control import DifferentiableMPC, SystemIdentifier

# Learn system dynamics
system_id = SystemIdentifier(model=dynamics_model)
trained_model = system_id.fit(state_data, input_data)

# Create MPC controller
mpc = DifferentiableMPC(system_model=trained_model, config=mpc_config)
control_action = mpc.solve(current_state, reference_trajectory)
```

## Performance Characteristics

- **L2O Speedup**: meta-learned optimizers accelerate convergence on learned problem families
- **Meta-Optimization**: faster convergence on related problems via warm-starting
- **Production Optimization**: adaptive JIT kernel fusion with measured speedups
- **Memory Efficiency**: pool-based GPU memory planning for co-located models

## Integration

The optimization module integrates seamlessly with:

- **Training**: Meta-optimization for training workflows
- **Neural Networks**: Compatible with all neural architectures
- **Physics**: Physics-informed optimization constraints
- **Deployment**: Production-ready optimization systems

## Second-Order Optimization {: #second-order }

Curvature-based optimization methods including L-BFGS and hybrid optimizers.

### Configuration Classes

::: opifex.optimization.second_order.config
    options:
        show_root_heading: true
        show_source: false
        members:
            - LBFGSConfig
            - GaussNewtonConfig
            - LevenbergMarquardtConfig
            - HybridOptimizerConfig

### L-BFGS and Gauss-Newton Wrappers

::: opifex.optimization.second_order.wrappers
    options:
        show_root_heading: true
        show_source: false
        members:
            - create_lbfgs_optimizer
            - create_gauss_newton_solver

### Hybrid Adam → L-BFGS Optimizer

::: opifex.optimization.second_order.hybrid_optimizer
    options:
        show_root_heading: true
        show_source: false
        members:
            - HybridOptimizer

### NNX Integration

::: opifex.optimization.second_order.nnx_integration
    options:
        show_root_heading: true
        show_source: false
        members:
            - NNXLBFGSWrapper
            - create_nnx_lbfgs_optimizer

For detailed algorithms and best practices, see the [Second-Order Optimization Guide](../methods/second-order-optimization.md).

## See Also

- [Optimization User Guide](../user-guide/optimization.md) - Practical usage guide
- [Second-Order Optimization](../methods/second-order-optimization.md) - L-BFGS, hybrid optimizers
- [Meta-Optimization Methods](../methods/meta-optimization.md) - Detailed algorithms
- [Production Optimization](../methods/production-optimization.md) - Enterprise features
- [Control Systems](../methods/control-systems.md) - Control theory integration

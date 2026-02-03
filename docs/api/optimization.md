# Optimization API Reference

## Overview

The Opifex optimization module provides comprehensive optimization algorithms and meta-learning approaches for scientific computing, including production optimization, learn-to-optimize algorithms, control systems, and quantum-aware optimization.

## Core Optimization Components

### Meta-Optimization Framework

Advanced meta-optimization algorithms that learn to optimize across families of related problems.

::: opifex.optimization.meta_optimization

### Production Optimization

Enterprise-grade optimization systems for deployment and scaling in production environments.

::: opifex.optimization.production

### Performance Monitoring

AI-powered performance monitoring with predictive scaling and anomaly detection.

::: opifex.optimization.performance_monitoring

### Adaptive Deployment

AI-driven deployment strategies with automatic rollback capabilities.

::: opifex.optimization.adaptive_deployment

### Global Resource Management

Multi-cloud optimization with cost intelligence and sustainability tracking.

::: opifex.deployment.resource_management

### Intelligent Edge Network

Global edge computing with sub-millisecond latency optimization.

::: opifex.optimization.edge_network

### Scientific Computing Integration

Physics-aware optimization with scientific validation and benchmarking.

::: opifex.optimization.scientific_integration

## Learn-to-Optimize (L2O) Algorithms

Advanced neural optimization methods that achieve significant speedups on learned problem families.

### L2O Engine

Core learn-to-optimize engine with parametric optimization solvers.

::: opifex.optimization.l2o.l2o_engine

### Advanced Meta-Learning

MAML, Reptile, and gradient-based meta-learning approaches.

::: opifex.optimization.l2o.advanced_meta_learning

### Adaptive Schedulers

Bayesian and performance-aware scheduling algorithms.

::: opifex.optimization.l2o.adaptive_schedulers

### Multi-Objective Optimization

Pareto frontier approximation and multi-objective optimization.

::: opifex.optimization.l2o.multi_objective

### Parametric Solvers

Neural networks for parametric programming and constraint satisfaction.

::: opifex.optimization.l2o.parametric_solver

### Constraint Learning

Automated constraint satisfaction learning algorithms.

::: opifex.optimization.l2o.constraint_learning

### Reinforcement Learning Optimization

RL-based optimization strategy selection and learning.

::: opifex.optimization.l2o.rl_optimization

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
- **`production.py`**: Production optimization and deployment systems
- **`performance_monitoring.py`**: AI-powered performance monitoring
- **`adaptive_deployment.py`**: Adaptive deployment with rollback automation
- **`../deployment/resource_management.py`**: Global resource management and cost optimization (imported from deployment module)
- **`edge_network.py`**: Intelligent edge network optimization
- **`scientific_integration.py`**: Physics-aware optimization integration

### L2O Submodule (`l2o/`)

- **`l2o_engine.py`**: Core L2O engine and parametric solvers
- **`advanced_meta_learning.py`**: MAML, Reptile, and gradient-based methods
- **`adaptive_schedulers.py`**: Bayesian and performance-aware schedulers
- **`multi_objective.py`**: Multi-objective optimization algorithms
- **`parametric_solver.py`**: Parametric programming solvers
- **`constraint_learning.py`**: Constraint satisfaction learning
- **`rl_optimization.py`**: Reinforcement learning optimization

### Control Submodule (`control/`)

- **`system_id.py`**: System identification networks
- **`mpc.py`**: Model predictive control frameworks

## Key Features

### Meta-Optimization Features

- Learn-to-Optimize (L2O) algorithms with >100x speedup
- Adaptive learning rate scheduling
- Warm-starting strategies for related problems
- Performance monitoring and analytics

### Production Optimization Features

- Hybrid performance platform with adaptive JIT
- Intelligent GPU memory management
- AI-powered deployment strategies
- Global resource management across cloud providers

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

### Production Deployment

```python
from opifex.optimization.production import HybridPerformancePlatform
from opifex.optimization.adaptive_deployment import AdaptiveDeploymentSystem

platform = HybridPerformancePlatform(
    gpu_memory_optimization=True,
    adaptive_jit=True
)

deployment = AdaptiveDeploymentSystem(
    canary_percentage=10,
    ai_driven_strategies=True
)
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

- **L2O Speedup**: >100x on learned problem families
- **Meta-Optimization**: 10-50x faster convergence on related problems
- **Production Optimization**: 40-60% reduction in computational costs
- **Edge Network**: Sub-millisecond latency optimization
- **Memory Efficiency**: Up to 80% memory usage reduction

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

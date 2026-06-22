# Optimization in Opifex

## Overview

The Opifex optimization module provides a full suite of advanced optimization algorithms specifically designed for scientific machine learning applications. It includes meta-optimization engines, learn-to-optimize (L2O) algorithms, production optimization systems, and quantum-aware optimization workflows.

## Key Components

### 1. Meta-Optimization Framework

The meta-optimization system learns to optimize across families of related problems, providing significant speedups and improved convergence:

- **Learn-to-Optimize (L2O)**: Neural networks that learn optimization algorithms
- **Adaptive Learning Rate Scheduling**: Performance-based adaptation with multiple strategies
- **Warm-Starting**: Parameter transfer between related optimization problems

### 2. Production Optimization

Enterprise-grade optimization systems for deployment and scaling:

- **Hybrid Performance Platform**: Adaptive JIT optimization with kernel-fusion strategies
- **Intelligent GPU Memory Management**: Optimized memory allocation and usage
- **Scientific Validation**: Physics-aware checks folded into the optimization pipeline

### 3. Learn-to-Optimize (L2O) Algorithms

Neural optimizers whose update rule is meta-learned over a family of related tasks. The implementation follows Andrychowicz et al. (2016, arXiv:1606.04474), Metz et al. (2020, arXiv:2009.11243), and the Persistent Evolution Strategies estimator of Vicol et al. (2021, arXiv:2112.13835):

- **Task and TaskFamily abstractions**: each task carries its own `init`, `loss`, and `normalizer`; a family exposes `sample` to draw fresh tasks
- **Per-parameter learned optimizers**: an MLP maps gradient features to per-coordinate updates (`MLPLearnedOptimizer`), plus a learnable scalar step size (`LearnableSGD`)
- **PES meta-training**: unbiased gradient estimates over truncated unrolls without backpropagating through the full optimization trajectory
- **Honest benchmarking**: held-out tasks compared against a *tuned* optax baseline, reporting per-task and median speedups to a target loss

### 4. Control Systems

Differentiable predictive control components:

- **System Identification Networks**: Learning system dynamics from data
- **Model Predictive Control (MPC)**: Differentiable MPC frameworks
- **Safety-Critical Control**: Control barriers and constraint handling
- **Real-Time Optimization**: High-performance control optimization

### 5. Scientific Computing Integration

Physics-aware optimization with scientific validation:

- **Physics-Informed Optimization**: Conservation law enforcement
- **Numerical Validation**: Stability and accuracy verification
- **Scientific Benchmarking**: Standardized performance evaluation
- **Domain-Specific Profiling**: Physics domain optimization profiling

## Core Concepts

### Meta-Learning for Optimization

Meta-learning allows optimization algorithms to learn from experience across multiple related problems. Instead of starting from scratch for each new optimization task, meta-learned optimizers can quickly adapt to new problems by leveraging knowledge from previously solved similar problems.

**Key Benefits:**

- Faster convergence on new problems
- Better initialization strategies
- Adaptive learning rate scheduling
- Transfer learning between related domains

### Learn-to-Optimize (L2O)

L2O algorithms use neural networks to learn optimization update rules. The learned optimizer is meta-trained on a `TaskFamily` and can outperform a *tuned* classical baseline on held-out tasks drawn from that same family (in-distribution). The high-level entry point is `L2OEngine`:

```python
import jax
from opifex.optimization.l2o import L2OEngine, MLPLearnedOptimizer, MLPTaskFamily

# A family of non-convex teacher-student MLP training tasks
family = MLPTaskFamily(input_dim=8, hidden_dim=16, output_dim=4)

# Wrap a per-parameter learned optimizer in the engine
engine = L2OEngine(MLPLearnedOptimizer(step_mult=0.03), family)

# Meta-train with Persistent Evolution Strategies (PES)
losses = engine.meta_train(
    jax.random.key(0),
    num_outer_steps=3000,
    num_tasks=32,
    meta_learning_rate=3e-3,
)

# Benchmark on held-out tasks against a tuned optax baseline
result = engine.benchmark(jax.random.key(1), num_tasks=48, num_steps=100)
# result["median_speedup"] ~ 1.8x vs tuned Adam, ~2.7x vs tuned SGD
```

The reported speedups hold only on tasks drawn from the meta-training family. As the VeLO-scaling analysis of Thérien et al. (2023, arXiv:2310.18191) documents, learned optimizers do not reliably transfer outside their training distribution, so benchmark numbers should always be read as in-distribution.

### Quantum-Aware Optimization

Specialized optimization algorithms for quantum mechanical systems:

- **SCF Acceleration**: Self-consistent field convergence acceleration
- **Energy Optimization**: Chemical accuracy targeting (<1 kcal/mol)
- **Quantum Constraints**: Built-in quantum mechanical constraint handling
- **Density Matrix Optimization**: Specialized algorithms for density functional theory

### Physics-Informed Optimization

Integration with physics-based constraints and conservation laws:

```python
from opifex.optimization.scientific_integration import ScientificComputingIntegrator
from opifex.optimization.meta_optimization import MetaOptimizer

# Create physics-aware optimizer
integrator = ScientificComputingIntegrator(
    conservation_laws=["energy", "momentum"],
    numerical_stability_checks=True
)

config = MetaOptimizerConfig(
    quantum_aware=True,
    meta_algorithm="l2o",
)

optimizer = MetaOptimizer(config=config, rngs=nnx.Rngs(42))
```

## Usage Patterns

### Basic Meta-Optimization

```python
from opifex.optimization.meta_optimization import LearnToOptimize, MetaOptimizerConfig

# Configure meta-optimizer
config = MetaOptimizerConfig(
    meta_learning_rate=1e-3,
    adaptation_steps=20,
    base_optimizer="adam",
    warm_start_strategy="previous_params",
)

# Initialize L2O optimizer
l2o = LearnToOptimize(config=config, rngs=nnx.Rngs(42))

# Optimize parameters
optimized_params = l2o.optimize(
    initial_params=params,
    objective_fn=loss_function,
    num_steps=1000
)
```

### Production Optimization

```python
from opifex.optimization.production import HybridPerformancePlatform, WorkloadProfile

# Create the production optimization platform
platform = HybridPerformancePlatform()

# Describe the production workload
workload = WorkloadProfile(
    batch_size=32,
    sequence_length=128,
    memory_footprint=2.0,
    compute_intensity=8.0,
    latency_requirement=10.0,
    throughput_requirement=100.0,
    model_complexity="medium",
)

# Optimize a trained model for production
optimized = platform.optimize_for_production(model, workload)
print(optimized.optimization_metadata["production_ready"])
```

### Low-Level L2O Building Blocks

When more control is needed than `L2OEngine` provides, the same components can be composed directly. `meta_train` returns the learned parameters `theta` and a loss curve; `benchmark_on_held_out_tasks` evaluates against a tuned optax baseline:

```python
import jax
from opifex.optimization.l2o import (
    MLPLearnedOptimizer,
    MLPTaskFamily,
    meta_train,
    benchmark_on_held_out_tasks,
)

family = MLPTaskFamily(input_dim=8, hidden_dim=16, output_dim=4)
learned_optimizer = MLPLearnedOptimizer(step_mult=0.03)

theta, loss_curve = meta_train(
    learned_optimizer,
    family,
    jax.random.key(0),
    num_outer_steps=3000,
    num_tasks=32,
    trunc_length=20,
    total_horizon=100,
    meta_learning_rate=3e-3,
)

report = benchmark_on_held_out_tasks(
    learned_optimizer,
    theta,
    family,
    jax.random.key(1),
    num_tasks=48,
    num_steps=100,
)
print(report["median_speedup"], report["fraction_reached_target"])
```

For a convex smoke test, swap `MLPTaskFamily` for `QuadraticTaskFamily(dim=...)`. A single fixed task can be lifted into a family with `single_task_to_family`.

## Integration with Other Components

### Training Integration

The optimization module seamlessly integrates with the training system:

```python
from opifex.training.basic_trainer import BasicTrainer
from opifex.optimization.meta_optimization import LearnToOptimize

# Use L2O with training
trainer = BasicTrainer(model=model, config=training_config)
l2o_optimizer = LearnToOptimize(config=l2o_config, rngs=nnx.Rngs(42))

trained_model = trainer.train(
    train_data=data,
    meta_optimizer=l2o_optimizer
)
```

### Applying a Learned Optimizer to a Task

A meta-trained optimizer is applied by constructing a concrete `Optimizer` from `theta` and running it on a task. `L2OEngine.optimize` (or the lower-level `loss_curve` helper) drives this loop and returns the per-step training loss:

```python
import jax
from opifex.optimization.l2o import L2OEngine, MLPLearnedOptimizer, MLPTaskFamily

family = MLPTaskFamily(input_dim=8, hidden_dim=16, output_dim=4)
engine = L2OEngine(MLPLearnedOptimizer(step_mult=0.03), family)
engine.meta_train(jax.random.key(0), num_outer_steps=3000, num_tasks=32)

# Draw a fresh held-out task and roll out the learned optimizer on it
task = family.sample(jax.random.key(2))
start = task.init(jax.random.key(3))
curve = engine.optimize(task, start, num_steps=100, key=jax.random.key(4))
```

Because each `Task` carries its own `init`, `loss`, and `normalizer`, any differentiable training objective (including neural-operator training losses) can be expressed as a `Task` and wrapped into a `TaskFamily` for meta-training.

## Performance Characteristics

### L2O Speedups

Speedups are measured as the ratio of steps a tuned classical baseline needs to reach a target loss versus the steps the learned optimizer needs, and are only meaningful on held-out tasks drawn from the meta-training family:

- On the showcase `MLPTaskFamily`, a meta-trained `MLPLearnedOptimizer` reaches the target loss roughly 1.8x faster than tuned Adam and 2.7x faster than tuned SGD.
- `benchmark` / `benchmark_on_held_out_tasks` report `median_speedup` and `fraction_reached_target` so claims stay reproducible.
- Per the VeLO-scaling analysis (Thérien et al., 2023, arXiv:2310.18191), learned optimizers do not reliably generalize beyond their training distribution; out-of-distribution speedups should not be assumed.

## Best Practices

### 1. Task Family Selection

For L2O to be effective, the tasks a `TaskFamily` samples should share structural similarities, since speedups only hold in-distribution:

- Similar parameter spaces and dimensionality
- Related loss surfaces (e.g. the same architecture with resampled data)
- Consistent loss normalization across sampled tasks

### 2. Meta-Training Strategy

- Sample diverse tasks from the family during meta-training
- Tune `trunc_length` and `total_horizon` so PES unrolls cover the optimization regime of interest
- Evaluate with `benchmark` on held-out tasks before trusting any speedup claim

### 3. Production Optimization

- Profile the workload with an accurate `WorkloadProfile` before optimizing
- Rely on the measured `improvement_factor`, not assumed speedups
- Gate physics workloads on the scientific validation score

### 4. Physics-Informed Optimization

- Always validate conservation laws
- Use domain-specific constraints
- Monitor numerical stability

## Troubleshooting

### Common Issues

1. **Slow Convergence**: Check learning rate scheduling and warm-starting
2. **Memory Issues**: Enable intelligent GPU memory management
3. **Numerical Instability**: Use physics-informed constraints
4. **Poor Generalization**: Increase task diversity in the meta-training `TaskFamily`; expect degradation on out-of-distribution tasks

### Performance Optimization

1. **Enable JIT Compilation**: Use adaptive JIT for production
2. **Optimize Memory Usage**: Enable intelligent GPU memory planning
3. **Use Appropriate Batch Sizes**: Match the workload's batch size to its latency target

## Second-Order Optimization

For problems where first-order methods like Adam struggle (e.g., ill-conditioned PINNs), second-order optimization provides faster convergence by exploiting curvature information.

### L-BFGS Optimization

L-BFGS approximates the inverse Hessian using limited memory, enabling quasi-Newton optimization for large-scale problems:

```python
from opifex.optimization.second_order import (
    create_lbfgs_optimizer,
    LBFGSConfig,
)

# Configure L-BFGS
config = LBFGSConfig(
    memory_size=10,           # History for Hessian approximation
    max_iterations=1000,
    tolerance=1e-8,
    line_search="zoom",       # Accurate line search
)

# Create optimizer
optimizer = create_lbfgs_optimizer(config)
```

### Hybrid Adam → L-BFGS

The hybrid approach combines Adam's robust early exploration with L-BFGS's fast late-stage convergence:

```python
from opifex.optimization.second_order import (
    HybridOptimizer,
    HybridOptimizerConfig,
)

config = HybridOptimizerConfig(
    adam_lr=1e-3,
    switch_threshold=1e-3,    # Switch when loss < threshold
    max_adam_steps=5000,
    lbfgs_config=LBFGSConfig(memory_size=20),
)

hybrid = HybridOptimizer(config)

# Automatic phase switching during training
for step in range(max_steps):
    loss, grads = compute_loss_and_grads(model)
    hybrid.step(model, grads)

    if hybrid.has_switched:
        print(f"Switched to L-BFGS at step {step}")
```

### Gauss-Newton / Levenberg-Marquardt

For nonlinear least-squares problems common in PINNs:

```python
from opifex.optimization.second_order import (
    GaussNewtonConfig,
    create_gauss_newton_solver,
)

config = GaussNewtonConfig(
    method="levenberg_marquardt",  # More robust than pure Gauss-Newton
    damping=1e-4,
    max_iterations=100,
)

solver = create_gauss_newton_solver(config)
```

For complete details on algorithms, NNX integration, and best practices, see the [Second-Order Optimization Guide](../methods/second-order-optimization.md).

## See Also

- [Second-Order Optimization](../methods/second-order-optimization.md) - L-BFGS, hybrid optimizers, Gauss-Newton
- [Learn-to-Optimize Methods](../methods/l2o.md) - Detailed L2O algorithms
- [Training Guide](training.md) - Integration with training workflows
- [Neural Networks](neural-networks.md) - Compatible architectures
- [API Reference](../api/optimization.md) - Complete API documentation

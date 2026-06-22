# Opifex Optimization: Learn-to-Optimize Engine & Advanced Solvers

This package provides advanced optimization algorithms for scientific machine learning, including meta-optimization engines, learn-to-optimize (L2O) algorithms, and quantum-aware optimization workflows.

## Components

### Meta-Optimization Infrastructure

- **`meta_optimization/`**: Modular meta-optimization package with L2O algorithms
  - `config.py`: Meta-optimizer configuration system
  - `schedulers.py`: Adaptive learning rate scheduling
  - `warm_starting.py`: Parameter transfer strategies
  - `neural_learner.py`: Neural network-based meta-learning optimizer (LearnToOptimize)
  - `monitoring.py`: Performance monitoring and analytics
  - `meta_optimizer.py`: Integrated meta-optimization system

- **`l2o/`**: Learn-to-Optimize — per-parameter learned optimisers meta-trained with Persistent
  Evolution Strategies (after Google's `learned_optimization`; Andrychowicz 2016, Metz 2020,
  Vicol 2021)
  - `core.py`: objective-carrying `Task` / `TaskFamily` and the `Optimizer` interface
  - `optimizers.py`: `Optimizer` ABC + `OptaxOptimizer` (hand-designed baseline family)
  - `tasks.py`: `QuadraticTaskFamily` and the `MLPTaskFamily` showcase task
  - `features.py`: per-parameter input features (momentum/RMS EMAs, tanh time embedding)
  - `learned.py`: `LearnedOptimizer` ABC, `MLPLearnedOptimizer`, `LearnableSGD`
  - `meta_train.py`: PES meta-training estimator + outer-Adam loop
  - `baselines.py`: optimistix classical baselines and tuned-optax baselines
  - `benchmark.py`: honest learning-curve and speedup-at-target benchmarking
  - `engine.py`: high-level `L2OEngine` (meta-train / apply / benchmark / persist)

The `l2o/` package is self-contained and independent of `meta_optimization/`; see
`docs/methods/l2o.md` for the method description.

### Meta-Optimization Components

The meta-optimization package (`opifex/optimization/meta_optimization/`) is split
into focused modules and provides:

- **MetaOptimizerConfig** - Configuration system for meta-optimization algorithms
- **AdaptiveLearningRateScheduler** - Multiple scheduling strategies (cosine annealing, linear, exponential)
- **WarmStartingStrategy** - Parameter transfer and similarity-based warm-starting
- **LearnToOptimize (L2O)** - Neural meta-learning optimization engine using FLAX NNX
- **PerformanceMonitor** - Performance tracking and analytics
- **MetaOptimizer** - Integrated meta-optimization system with quantum-aware adaptations
- **Quantum Extensions** - SCF convergence acceleration and energy tracking

Technical characteristics:

- **Neural Meta-Learning**: L2O implementation using FLAX NNX neural networks
- **Adaptive Scheduling**: Performance-based learning rate adaptation with multiple strategies
- **Warm-Starting**: Parameter transfer between related optimization problems
- **Performance Monitoring**: Tracking of optimization convergence and efficiency
- **Quantum-Aware Optimization**: Specialized algorithms for quantum mechanical problems
- **Multi-Strategy Support**: Flexible framework supporting various optimization strategies
- **JAX Integration**: Native JAX Array support with automatic differentiation
- **Type Safety**: Type annotations with jaxtyping for scientific computing

## Key Features

- **Learn-to-Optimize (L2O)**: Neural meta-learning optimization engines
- **Adaptive Learning Rates**: Performance-based adaptation with multiple strategies
- **Warm-Starting**: Parameter transfer and similarity-based initialization
- **Performance Monitoring**: Full tracking and analytics
- **Quantum-Aware Optimization**: SCF acceleration and energy convergence
- **Meta-Optimization**: Complete meta-optimization system with quantum extensions
- **JAX Integration**: Native JAX Array support with automatic differentiation
- **Type Safety**: Full type annotations with jaxtyping
- **Performance Optimized**: FLAX NNX transformations for efficiency

## Usage Examples

### Basic Meta-Optimization

```python
from opifex.optimization.meta_optimization import LearnToOptimize, MetaOptimizerConfig
import flax.nnx as nnx
import jax.numpy as jnp

# Create meta-optimization configuration
config = MetaOptimizerConfig(
    meta_learning_rate=1e-3,
    num_unroll_steps=20,
    num_meta_epochs=100,
    adaptation_strategy="cosine_annealing"
)

# Initialize L2O optimizer
l2o = LearnToOptimize(config=config, rngs=nnx.Rngs(42))

# Use for optimization problem
params = {"weights": jnp.ones((10, 1)), "bias": jnp.zeros((1,))}
optimized_params = l2o.optimize(params, objective_fn, num_steps=1000)
```

### Adaptive Learning Rate Scheduling

```python
from opifex.optimization.meta_optimization import AdaptiveLearningRateScheduler

# Create adaptive scheduler
scheduler = AdaptiveLearningRateScheduler(
    initial_lr=1e-3,
    strategy="cosine_annealing",
    adaptation_frequency=10,
    performance_threshold=0.01
)

# Use in training loop
for epoch in range(num_epochs):
    # Compute loss and update parameters
    loss = compute_loss(params, data)

    # Adapt learning rate based on performance
    current_lr = scheduler.adapt(loss, epoch)

    # Update parameters with adapted learning rate
    params = update_params(params, gradients, current_lr)
```

### Warm-Starting Strategy

```python
from opifex.optimization.meta_optimization import WarmStartingStrategy

# Create warm-starting strategy
warm_starter = WarmStartingStrategy(
    strategy_type="parameter_transfer",
    similarity_threshold=0.8,
    transfer_fraction=0.5
)

# Use for related optimization problems
source_params = {"weights": source_weights, "bias": source_bias}
target_params = warm_starter.initialize_from_source(
    source_params=source_params,
    target_shape=target_shape,
    problem_similarity=0.9
)
```

### Performance Monitoring

```python
from opifex.optimization.meta_optimization import PerformanceMonitor

# Create performance monitor
monitor = PerformanceMonitor(
    track_convergence=True,
    track_efficiency=True,
    track_stability=True,
    save_trajectory=True
)

# Monitor optimization process
for step in range(optimization_steps):
    # Perform optimization step
    params, loss = optimization_step(params, data)

    # Track performance
    monitor.update(step, loss, params)

    # Get performance analytics
    if step % 100 == 0:
        metrics = monitor.get_metrics()
        print(f"Convergence rate: {metrics['convergence_rate']}")
        print(f"Efficiency score: {metrics['efficiency_score']}")
```

### Electronic-structure problems

The electronic-structure problem is backed by the real Kohn-Sham DFT solver
(`opifex.neural.quantum.dft.SCFSolver`); its energy and analytic forces are the
converged Kohn-Sham quantities and are `jit` / `grad` / `vmap` compatible, so
they slot into geometry-optimisation and meta-optimisation loops.

```python
import jax

from opifex.core import create_neural_dft_problem
from opifex.core.quantum.molecular_system import create_molecular_system

with jax.enable_x64(True):
    molecular_system = create_molecular_system(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
    )
    problem = create_neural_dft_problem(molecular_system)  # functional_type -> LDA/PBE

    energy = problem.compute_energy()   # converged Kohn-Sham energy (Hartree)
    forces = problem.compute_forces()   # analytic -dE/dR
```

### Meta-Optimization for Multiple Problems

```python
from opifex.optimization.meta_optimization import MetaOptimizer

# Create meta-optimizer for multiple related problems
meta_optimizer = MetaOptimizer(
    config=config,
    rngs=nnx.Rngs(42)
)

# Train meta-optimizer on multiple problems
problems = [problem1, problem2, problem3]  # Related optimization problems
meta_optimizer.meta_train(
    problems=problems,
    num_meta_epochs=50,
    num_inner_steps=20
)

# Use trained meta-optimizer for new problem
new_problem_solution = meta_optimizer.optimize(
    problem=new_problem,
    num_steps=100,
    use_meta_initialization=True
)
```

### Integration with Physics-Informed Training

```python
from opifex.optimization.meta_optimization import LearnToOptimize, MetaOptimizerConfig
from opifex.core.training.trainer import Trainer
from opifex.core.physics.losses import PhysicsInformedLoss

# Create L2O optimizer for physics-informed training
l2o_config = MetaOptimizerConfig(
    meta_learning_rate=1e-3,
    physics_aware=True,
    conservation_weighting=True
)

l2o_optimizer = LearnToOptimize(config=l2o_config, rngs=nnx.Rngs(42))

# Use with physics-informed training
physics_loss = PhysicsInformedLoss()
trainer = Trainer(model=model, config=config)
trainer.set_physics_loss(physics_loss)

# Train with L2O meta-optimization
trained_model, metrics = trainer.train(
    train_data=(x_train, y_train),
    boundary_data=(x_boundary, y_boundary),
    meta_optimizer=l2o_optimizer  # Use L2O for optimization
)
```

## Technical Architecture

### Meta-Learning Framework

The meta-optimization system is built on a hierarchical architecture:

1. **Meta-Optimizer Layer**: Coordinates overall optimization strategy
2. **Algorithm Layer**: Implements specific optimization algorithms (L2O, adaptive schedules)
3. **Monitoring Layer**: Tracks performance and provides analytics
4. **Integration Layer**: Seamless integration with training and physics-informed workflows

### Learn-to-Optimize (L2O) Implementation

The L2O system uses neural networks to learn optimization algorithms:

- **Meta-Network**: FLAX NNX neural network that learns optimization updates
- **Unrolled Optimization**: Differentiable optimization loops for meta-learning
- **Parameter Transfer**: Warm-starting capabilities for related problems
- **Performance Adaptation**: Dynamic adaptation based on optimization performance

### Quantum-Aware Extensions

Specialized optimization for quantum mechanical problems:

- **Energy Optimization**: Specialized algorithms for energy minimization
- **Quantum Constraints**: Built-in handling of quantum mechanical constraints
- **Real Kohn-Sham DFT**: Differentiable energy and analytic forces from
  `opifex.neural.quantum.dft.SCFSolver`

## Integration with Other Packages

- **[Training Package](../training/README.md)**: Seamless integration with physics-informed training workflows
- **[Neural Package](../neural/README.md)**: Full compatibility with neural network architectures
- **[Core Package](../core/README.md)**: Integration with problem definitions and constraints
- **[Physics Package](../physics/README.md)**: Support for physics-informed optimization

For implementation history, see the main [CHANGELOG.md](../../CHANGELOG.md).

# Learn-to-Optimize (L2O) Methods

## Overview

Learn-to-Optimize (L2O) represents an advanced approach to optimization where neural networks learn to optimize other neural networks. Instead of using hand-crafted optimization algorithms like Adam or SGD, L2O algorithms learn update rules that are specifically tailored to families of related problems, achieving significant speedups and improved convergence properties.

The Opifex L2O framework provides implementations of parametric programming solvers, constraint satisfaction learning, multi-objective optimization, reinforcement learning-based strategy selection, advanced meta-learning algorithms (MAML, Reptile), and adaptive schedulers.

## Theoretical Foundation

### Learning Optimization Algorithms

The core idea of L2O is to parameterize the optimization update rule with a neural network:

$$\theta_{t+1} = \theta_t + \alpha \cdot g_{\phi}(\nabla_{\theta} \mathcal{L}(\theta_t), h_t, \theta_t)$$

where:

- $g_{\phi}$ is a neural network (the learned optimizer) parameterized by $\phi$
- $h_t$ is the hidden state for recurrent optimizers
- $\alpha$ is a learned or fixed step size
- $\nabla_{\theta} \mathcal{L}(\theta_t)$ is the gradient of the loss function

### Meta-Learning Framework

L2O operates within a meta-learning framework where:

- **Meta-learner**: The L2O algorithm that learns optimization strategies
- **Base-learner**: The model being optimized
- **Task distribution**: A family of related optimization problems

The meta-learner is trained to minimize:

$$\mathcal{L}_{meta}(\phi) = \mathbb{E}_{\tau \sim \mathcal{T}} \left[ \mathcal{L}_{\tau}(\theta_T^{(\tau)}) \right]$$

where $\theta_T^{(\tau)}$ is the final parameters after $T$ optimization steps on task $\tau$.

## Core L2O Components

### 1. L2O Engine

The central component that orchestrates learn-to-optimize algorithms. It requires both an `L2OEngineConfig` and a `MetaOptimizerConfig`:

```python
from opifex.optimization.l2o import L2OEngine, L2OEngineConfig
from opifex.core.training.config import MetaOptimizerConfig
import flax.nnx as nnx

# Configure L2O engine
l2o_config = L2OEngineConfig(
    solver_type="parametric",        # "parametric", "gradient", or "hybrid"
    problem_encoder_layers=[128, 64, 32],
    use_traditional_fallback=True,
    enable_meta_learning=True,
    integration_mode="unified",      # "unified", "parametric_only", "gradient_only"
    performance_tracking=True,
    adaptive_selection=True,
)

meta_config = MetaOptimizerConfig(
    meta_algorithm="l2o",
    base_optimizer="adam",
    meta_learning_rate=1e-4,
)

# Create L2O engine
l2o_engine = L2OEngine(
    l2o_config=l2o_config,
    meta_config=meta_config,
    rngs=nnx.Rngs(42),
)
```

The engine provides several solving methods:

```python
from opifex.optimization.l2o import OptimizationProblem
import jax.numpy as jnp

# Define an optimization problem
problem = OptimizationProblem(
    problem_type="quadratic",  # "quadratic", "linear", or "nonlinear"
    dimension=20,
)
problem_params = jnp.ones(50)  # Problem parameters

# Solve using parametric solver
solution = l2o_engine.solve_parametric_problem(problem, problem_params)

# Solve using gradient-based L2O
def loss_fn(x):
    return jnp.sum(x**2)

solution = l2o_engine.solve_gradient_problem(loss_fn, jnp.zeros(20), steps=100)

# Automatic algorithm selection
algorithm_used, solution = l2o_engine.solve_automatically(problem, problem_params)

# Get algorithm recommendation
recommendation = l2o_engine.recommend_algorithm(problem, problem_params)

# Compare all available solvers
comparison = l2o_engine.compare_all_solvers(problem, problem_params)

# Solve with meta-learning (stores solutions for future adaptation)
solution, metadata = l2o_engine.solve_with_meta_learning(problem, problem_params)

# Optimize with full meta-framework (returns history)
final_params, history = l2o_engine.optimize_with_meta_framework(
    loss_fn, initial_params=jnp.zeros(20), steps=50,
)

# Physics-informed optimization with adaptive momentum
solution = l2o_engine.solve_physics_informed(
    physics_loss_fn=loss_fn,
    initial_params=jnp.zeros(20),
    steps=100,
)
```

### 2. Parametric Programming Solvers

Neural networks that learn to solve parametric optimization problems directly:

```python
from opifex.optimization.l2o import ParametricProgrammingSolver, SolverConfig
import flax.nnx as nnx

# Configure parametric solver
solver_config = SolverConfig(
    hidden_sizes=[256, 128, 64],
    use_traditional_fallback=True,
)

# Create parametric solver
solver = ParametricProgrammingSolver(
    config=solver_config,
    input_dim=100,
    output_dim=20,
    rngs=nnx.Rngs(42),
)

# Forward pass: map problem parameters to solutions
import jax.numpy as jnp
problem_params = jnp.ones((1, 100))
solution = solver(problem_params)

# Solve with traditional solver fallback
solution_with_fallback = solver.solve_with_fallback(problem_params)
```

### 3. Multi-Objective L2O

Learn to optimize problems with multiple competing objectives:

```python
from opifex.optimization.l2o import MultiObjectiveL2OEngine, MultiObjectiveConfig
import flax.nnx as nnx

# Configure multi-objective L2O
mo_config = MultiObjectiveConfig(
    num_objectives=3,
    pareto_points_target=100,
    scalarization_strategy="learned",  # "learned", "weighted_sum", "chebyshev"
    diversity_pressure=0.1,
    adaptive_weights=True,
    dominated_solution_filtering=True,
)

# Create multi-objective L2O engine (requires an L2OEngine instance)
mo_l2o = MultiObjectiveL2OEngine(config=mo_config, rngs=nnx.Rngs(42))
```

### 4. Constraint Learning

Automatically learn to satisfy constraints during optimization:

```python
from opifex.optimization.l2o import ConstraintHandler

# Define constraint handling
constraint_handler = ConstraintHandler(
    method="penalty",        # "penalty", "barrier", or "projection"
    penalty_weight=1.0,
    barrier_parameter=0.1,
)

# Compute penalty for constraint violation
import jax.numpy as jnp
x = jnp.array([1.0, 2.0, 3.0])
constraint_value = jnp.array([0.5])  # Should be zero for equality
penalty = constraint_handler.compute_penalty(x, constraint_value, "equality")
```

### 5. Reinforcement Learning Optimization

Use RL to learn optimization strategies via a DQN agent:

```python
from opifex.optimization.l2o import RLOptimizationEngine, RLOptimizationConfig
import flax.nnx as nnx

# Configure RL-based optimization
rl_config = RLOptimizationConfig(
    state_dim=64,
    action_dim=12,              # Algorithm selection + hyperparameter adjustments
    hidden_dims=(256, 256, 128),
    learning_rate=1e-4,
    discount_factor=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    replay_buffer_size=10000,
    batch_size=32,
    max_episode_length=1000,
)

# Create RL optimization engine
rl_optimizer = RLOptimizationEngine(config=rl_config, rngs=nnx.Rngs(42))
```

## Advanced L2O Algorithms

### 1. Adaptive Schedulers

Learn adaptive learning rate schedules based on optimization progress:

```python
from opifex.optimization.l2o import (
    PerformanceAwareScheduler,
    BayesianSchedulerOptimizer,
    MetaSchedulerConfig,
    create_l2o_engine_with_adaptive_schedulers,
)
import flax.nnx as nnx

# Configure adaptive scheduling
scheduler_config = MetaSchedulerConfig(
    base_learning_rate=1e-3,
    min_learning_rate=1e-6,
    max_learning_rate=1e-1,
    convergence_window=10,
    patience=5,
    adaptation_factor=0.5,
    enable_performance_awareness=True,
    enable_bayesian_optimization=False,
)

# Performance-aware scheduler adapts based on convergence detection
perf_scheduler = PerformanceAwareScheduler(
    config=scheduler_config,
    rngs=nnx.Rngs(42),
)

# Integrate adaptive schedulers with L2O engine
l2o_with_schedulers = create_l2o_engine_with_adaptive_schedulers(
    l2o_engine=l2o_engine,
    scheduler_config=scheduler_config,
    rngs=nnx.Rngs(42),
)
```

### 2. Advanced Meta-Learning Integration

Combine L2O with MAML and Reptile:

```python
from opifex.optimization.l2o import (
    MAMLOptimizer,
    MAMLConfig,
    ReptileOptimizer,
    ReptileConfig,
    GradientBasedMetaLearner,
    MetaL2OIntegration,
)

# MAML configuration
maml_config = MAMLConfig(
    inner_learning_rate=1e-3,
    meta_learning_rate=1e-4,
    inner_steps=5,
    meta_batch_size=8,
    second_order=True,
)

# Reptile configuration (simpler, first-order)
reptile_config = ReptileConfig(
    meta_learning_rate=1e-3,
    inner_learning_rate=1e-2,
    inner_steps=10,
    meta_batch_size=16,
    task_sampling_strategy="uniform",
)

# Meta-L2O integration for self-improving optimization
meta_l2o = MetaL2OIntegration(
    base_l2o_engine=l2o_engine,
    rngs=nnx.Rngs(42),
)
```

## Scientific Computing Applications

### 1. Physics-Informed Optimization

L2O for physics-informed optimization problems:

```python
from opifex.neural.base import StandardMLP
from opifex.core.physics.losses import PhysicsInformedLoss, PhysicsLossConfig
import flax.nnx as nnx
import jax.numpy as jnp

rngs = nnx.Rngs(42)

# Create PINN model
pinn_model = StandardMLP(
    layer_sizes=[2, 50, 50, 50, 1],
    activation="tanh",
    rngs=rngs,
)

# Physics-informed loss
config = PhysicsLossConfig(
    data_loss_weight=1.0,
    physics_loss_weight=1.0,
    boundary_loss_weight=10.0,
)

physics_loss = PhysicsInformedLoss(
    config=config,
    equation_type="heat",
    domain_type="2d",
)

# Use L2O engine for physics-informed optimization
solution = l2o_engine.solve_physics_informed(
    physics_loss_fn=lambda params: jnp.sum(params**2),  # Simplified
    initial_params=jnp.zeros(20),
    steps=100,
)
```

### 2. Neural Operator Training

L2O for neural operator training:

```python
from opifex.neural.operators.fno import FourierNeuralOperator
from opifex.neural.operators.deeponet import DeepONet
import flax.nnx as nnx

rngs = nnx.Rngs(42)

# Fourier Neural Operator
fno_model = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    modes=16,
    num_layers=4,
    rngs=rngs,
)

# DeepONet
deeponet_model = DeepONet(
    branch_sizes=[100, 64, 64, 32],
    trunk_sizes=[2, 64, 64, 32],
    activation="gelu",
    rngs=rngs,
)

# Use Trainer with standard training loop
from opifex.core.training import Trainer, TrainingConfig

config = TrainingConfig(num_epochs=100, learning_rate=1e-3, batch_size=32)
trainer = Trainer(model=fno_model, config=config, rngs=rngs)

# Train using .fit() (not .train())
trained_model, metrics = trainer.fit(
    train_data=(x_train, y_train),
    val_data=(x_val, y_val),
)
```

## Performance Analysis

### Speedup Characteristics

L2O algorithms achieve significant speedups across different problem domains. Use the `compare_all_solvers` method to benchmark:

```python
from opifex.optimization.l2o import OptimizationProblem
import jax.numpy as jnp

# Define test problem
problem = OptimizationProblem(problem_type="quadratic", dimension=20)
problem_params = jnp.ones(50)

# Compare all available solvers
results = l2o_engine.compare_all_solvers(problem, problem_params)

for solver_name, metrics in results.items():
    print(f"{solver_name}:")
    print(f"  Time: {metrics['time']:.4f}s")
    if 'speedup' in metrics:
        print(f"  Speedup: {metrics['speedup']:.1f}x")
```

### Typical Performance Gains

- **Similar Problems**: 50-100x speedup with learned optimizers
- **Related Problem Families**: 10-50x improvement in convergence
- **Transfer Learning**: 5-20x speedup on new but related problems
- **Multi-Objective**: 20-40x faster Pareto frontier discovery

### Memory and Computational Efficiency

```python
# Memory-efficient L2O configuration
efficient_config = L2OEngineConfig(
    solver_type="parametric",
    performance_tracking=True,
    adaptive_selection=True,
)
```

## Integration with Opifex Ecosystem

### Training Integration

The L2O engine integrates with Opifex training through the meta-framework:

```python
from opifex.optimization.l2o import L2OEngine, L2OEngineConfig
from opifex.core.training.config import MetaOptimizerConfig
import jax.numpy as jnp

# Create L2O engine
l2o_config = L2OEngineConfig(solver_type="gradient", integration_mode="unified")
meta_config = MetaOptimizerConfig(meta_algorithm="l2o", base_optimizer="adam")

l2o_engine = L2OEngine(
    l2o_config=l2o_config,
    meta_config=meta_config,
    rngs=nnx.Rngs(42),
)

# Optimize using the meta-framework
def objective(params):
    return jnp.sum(params**2)

final_params, history = l2o_engine.optimize_with_meta_framework(
    loss_fn=objective,
    initial_params=jnp.ones(10),
    steps=50,
)

# History contains per-step loss, learning rate, and gradient norm
for step_info in history[:5]:
    print(f"Step {step_info['step']}: loss={step_info['loss']:.6f}")
```

## Best Practices

### 1. Problem Family Design

For effective L2O training, problems must be related (same structure, varying parameters):

```python
from opifex.optimization.l2o import OptimizationProblem

# Good: Related optimization problems
problem_family = [
    OptimizationProblem(problem_type="quadratic", dimension=d)
    for d in [10, 20, 50]
]

# Good: Physics problems with varying parameters
# (e.g., Burgers equation with viscosity in [0.005, 0.05])
# Tasks MUST share structure for meta-learning to be effective
```

### 2. Meta-Training Strategy

Key insight: For meta-learning (MAML/Reptile) to show significant improvement, tasks must be structurally related. Random or unrelated problems do not benefit from meta-learning.

Results from Opifex examples:

- MAML: 60% lower loss, 10x speedup vs random initialization on related Burgers problems
- Reptile: 30% lower loss (simpler but less effective than MAML)

### 3. Algorithm Selection Guidelines

Use `l2o_engine.recommend_algorithm()` for automatic recommendations, or follow these guidelines:

- **Small linear/quadratic problems** (dim <= 20): Use `"parametric"` solver
- **Large-scale problems** (dim > 100): Use `"gradient"` solver
- **Mixed or uncertain**: Use `"hybrid"` mode with automatic selection

## Troubleshooting

### Common Issues and Solutions

**L2O not converging during meta-training:**

- Ensure problem family is structurally related
- Try reducing meta learning rate
- Increase inner loop steps

**Parametric solver inaccuracy:**

- Enable traditional fallback: `use_traditional_fallback=True`
- Increase hidden layer sizes in `SolverConfig`

**Slow L2O optimization:**

- Use JIT compilation on the objective function
- Reduce `steps` parameter and check early convergence
- Use `performance_tracking=True` to identify bottlenecks

## Future Directions

### Research Areas

1. **Automated L2O Design**: Learning to design L2O architectures
2. **Few-Shot L2O**: Rapid adaptation with minimal data
3. **Continual L2O**: Learning new optimization strategies without forgetting

### Planned Enhancements

1. **Distributed L2O**: Multi-device L2O training and optimization
2. **Federated L2O**: Privacy-preserving L2O across institutions

## References

1. Andrychowicz, M., et al. "Learning to learn by gradient descent by gradient descent." NIPS 2016.
2. Li, K., & Malik, J. "Learning to optimize." ICLR 2017.
3. Wichrowska, O., et al. "Learned optimizers that scale and generalize." ICML 2017.
4. Metz, L., et al. "Understanding and correcting pathologies in the training of learned optimizers." ICML 2019.
5. Chen, Y., et al. "Learning to optimize: A primer and a benchmark." JMLR 2022.

## See Also

- [Meta-Optimization Methods](meta-optimization.md) - Broader meta-optimization framework
- [Optimization User Guide](../user-guide/optimization.md) - Practical usage guide
- [L2O Example](../examples/optimization/learn-to-optimize.md) - Learn-to-optimize example
- [API Reference](../api/optimization.md) - API documentation

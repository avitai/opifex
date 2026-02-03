# Second-Order Optimization

Second-order optimization methods leverage curvature information (Hessian or approximations) for faster convergence, particularly beneficial in the later stages of physics-informed neural network training.

## Overview

Second-order methods offer significant advantages for PINN training:

- **Faster convergence** near optima due to curvature information
- **Better handling** of ill-conditioned loss landscapes
- **Reduced sensitivity** to learning rate selection
- **More effective** in smooth regions of the loss landscape

!!! tip "Survey Reference"
    This implementation follows recommendations from Section 7 of the PINN survey (arXiv:2601.10222v1).

## Methods

### L-BFGS

L-BFGS (Limited-memory BFGS) approximates the inverse Hessian using a limited history of gradient differences, making it suitable for large-scale optimization.

```python
from opifex.optimization.second_order import (
    create_lbfgs_optimizer,
    LBFGSConfig,
    LinesearchType,
)

# Configure L-BFGS
config = LBFGSConfig(
    memory_size=10,                    # Gradient pairs to store (typically 3-20)
    scale_init_precond=True,           # Scale initial preconditioner
    linesearch=LinesearchType.ZOOM,    # Line search algorithm
    max_linesearch_steps=20,           # Max line search iterations
    max_iterations=100,                # Max L-BFGS iterations
    tolerance=1e-6,                    # Convergence tolerance
)

# Create optimizer
optimizer = create_lbfgs_optimizer(config)
```

**Line Search Options:**

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `ZOOM` | Strong Wolfe conditions with zoom | General use, guaranteed descent |
| `BACKTRACKING` | Simple Armijo backtracking | Faster per-step, less robust |

**Using L-BFGS with optax:**

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

# Define loss function
def loss_fn(model, x, y_true):
    y_pred = model(x)
    return jnp.mean((y_pred - y_true) ** 2)

# L-BFGS requires value_and_grad_fn for line search
def value_and_grad_fn(params, model_template, x, y_true):
    def loss(params):
        model = model_template.replace(params=params)
        return loss_fn(model, x, y_true)
    return jax.value_and_grad(loss)(params)

# Training with L-BFGS
optimizer = create_lbfgs_optimizer()
opt_state = optimizer.init(params)

for step in range(num_steps):
    loss, grads = value_and_grad_fn(params, model, x, y_true)
    updates, opt_state = optimizer.update(
        grads, opt_state, params,
        value=loss,
        grad=grads,
        value_fn=lambda p: loss_fn(model.replace(params=p), x, y_true),
    )
    params = optax.apply_updates(params, updates)
```

### Gauss-Newton

Gauss-Newton is effective for nonlinear least-squares problems, approximating the Hessian using only first derivatives.

```python
from opifex.optimization.second_order import (
    create_gauss_newton_solver,
    GaussNewtonConfig,
)
import optimistix as optx

# Configure solver
config = GaussNewtonConfig(
    max_iterations=100,
    rtol=1e-6,
    atol=1e-6,
)

# Create solver
solver = create_gauss_newton_solver(config)

# Use with optimistix for least-squares
def residual_fn(params, args):
    """Residual function for least-squares."""
    model = args['model']
    x, y_true = args['x'], args['y_true']
    y_pred = model(x)
    return y_pred - y_true

# Solve
result = optx.least_squares(
    residual_fn,
    solver=solver,
    y0=initial_params,
    args={'model': model, 'x': x, 'y_true': y_true},
)
optimal_params = result.value
```

### Levenberg-Marquardt

Levenberg-Marquardt adds damping to Gauss-Newton for improved robustness, especially when far from the optimum.

```python
from opifex.optimization.second_order import (
    create_levenberg_marquardt_solver,
    GaussNewtonConfig,
)

# Configure with damping parameters
config = GaussNewtonConfig(
    damping_factor=1e-3,           # Initial damping (λ)
    damping_increase_factor=10.0,  # Factor to increase on failure
    damping_decrease_factor=0.1,   # Factor to decrease on success
    min_damping=1e-10,             # Minimum damping
    max_damping=1e10,              # Maximum damping
    max_iterations=100,
    rtol=1e-6,
    atol=1e-6,
)

solver = create_levenberg_marquardt_solver(config)
```

**Damping Behavior:**

- **Large damping:** Behaves like gradient descent (robust, slow)
- **Small damping:** Behaves like Gauss-Newton (fast, less robust)
- **Adaptive:** Increases damping on failed steps, decreases on success

### BFGS

Full-memory BFGS for smaller-scale problems where storing the complete inverse Hessian approximation is feasible.

```python
from opifex.optimization.second_order import (
    create_bfgs_solver,
    GaussNewtonConfig,
)
import optimistix as optx

solver = create_bfgs_solver(
    GaussNewtonConfig(rtol=1e-6, atol=1e-6)
)

# Use with optimistix minimise
result = optx.minimise(
    loss_fn,
    solver=solver,
    y0=initial_params,
)
```

### Hybrid Adam to L-BFGS Optimizer

The hybrid optimizer combines Adam's robustness in early training with L-BFGS's efficiency for final convergence.

!!! Insight
    L-BFGS is more effective in later stages when loss varies smoothly.

```python
from opifex.optimization.second_order import (
    HybridOptimizer,
    HybridOptimizerConfig,
    SwitchCriterion,
    LBFGSConfig,
)
import optax

# Configure hybrid optimizer
config = HybridOptimizerConfig(
    # Adam phase
    first_order_steps=1000,        # Steps before considering switch
    adam_learning_rate=1e-3,
    adam_b1=0.9,
    adam_b2=0.999,

    # Switching criterion
    switch_criterion=SwitchCriterion.LOSS_VARIANCE,
    loss_variance_threshold=1e-4,
    loss_history_window=50,

    # L-BFGS phase
    lbfgs_config=LBFGSConfig(
        memory_size=10,
        max_linesearch_steps=20,
    ),
)

# Create and use optimizer
optimizer = HybridOptimizer(config)
state = optimizer.init(params)

# Training loop
for step in range(num_steps):
    loss, grads = jax.value_and_grad(loss_fn)(params)

    updates, state = optimizer.update(
        grads, state, params,
        loss=loss,
        value_fn=lambda p: loss_fn(p),
    )
    params = optax.apply_updates(params, updates)

    # Check current optimizer mode
    if state.switched:
        print(f"Step {step}: Using L-BFGS")
```

**Switch Criteria:**

| Criterion | Description | When to Use |
|-----------|-------------|-------------|
| `EPOCH` | Switch after fixed steps | Simple, predictable |
| `LOSS_VARIANCE` | Switch when loss variance drops | Detects smooth regions |
| `GRADIENT_NORM` | Switch when gradients are small | Near convergence |
| `RELATIVE_IMPROVEMENT` | Switch when improvement slows | Adaptive to progress |

```python
# Example: Gradient norm-based switching
config = HybridOptimizerConfig(
    first_order_steps=500,
    switch_criterion=SwitchCriterion.GRADIENT_NORM,
    gradient_norm_threshold=1e-3,
)

# Example: Relative improvement-based switching
config = HybridOptimizerConfig(
    first_order_steps=500,
    switch_criterion=SwitchCriterion.RELATIVE_IMPROVEMENT,
    relative_improvement_threshold=1e-4,
)
```

## NNX Integration

The second-order optimizers integrate seamlessly with FLAX NNX models.

```python
from flax import nnx
from opifex.optimization.second_order import HybridOptimizer, HybridOptimizerConfig

# Create NNX model
class MyPINN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.layers = nnx.List([
            nnx.Linear(2, 64, rngs=rngs),
            nnx.Linear(64, 64, rngs=rngs),
            nnx.Linear(64, 1, rngs=rngs),
        ])

    def __call__(self, x):
        for layer in list(self.layers)[:-1]:
            x = nnx.tanh(layer(x))
        return list(self.layers)[-1](x)

model = MyPINN(rngs=nnx.Rngs(0))

# Define loss with NNX
def loss_fn(model):
    predictions = model(x_collocation)
    residual = compute_pde_residual(model, x_collocation)
    return jnp.mean(residual ** 2)

# Training with hybrid optimizer
optimizer = HybridOptimizer(HybridOptimizerConfig())
graphdef, state = nnx.split(model)
opt_state = optimizer.init(state)

for step in range(num_steps):
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    state = nnx.state(model)

    updates, opt_state = optimizer.update(
        grads, opt_state, state,
        loss=loss,
    )

    # Apply updates
    new_state = optax.apply_updates(state, updates)
    nnx.update(model, new_state)
```

## Configuration Reference

### LBFGSConfig

```python
@dataclass(frozen=True)
class LBFGSConfig:
    memory_size: int = 10           # Gradient pairs to store
    scale_init_precond: bool = True # Scale initial preconditioner
    linesearch: LinesearchType = LinesearchType.ZOOM
    max_linesearch_steps: int = 20
    max_iterations: int = 100
    tolerance: float = 1e-6
```

### GaussNewtonConfig

```python
@dataclass(frozen=True)
class GaussNewtonConfig:
    damping_factor: float = 1e-3
    damping_increase_factor: float = 10.0
    damping_decrease_factor: float = 0.1
    min_damping: float = 1e-10
    max_damping: float = 1e10
    max_iterations: int = 100
    rtol: float = 1e-6
    atol: float = 1e-6
```

### HybridOptimizerConfig

```python
@dataclass(frozen=True)
class HybridOptimizerConfig:
    first_order_steps: int = 1000
    switch_criterion: SwitchCriterion = SwitchCriterion.LOSS_VARIANCE
    loss_variance_threshold: float = 1e-4
    loss_history_window: int = 50
    gradient_norm_threshold: float = 1e-3
    relative_improvement_threshold: float = 1e-4
    adam_learning_rate: float = 1e-3
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    lbfgs_config: LBFGSConfig = field(default_factory=LBFGSConfig)
```

## Best Practices

### When to Use Second-Order Methods

| Scenario | Recommended Method |
|----------|-------------------|
| General PINN training | Hybrid Adam→L-BFGS |
| Well-conditioned problems | Pure L-BFGS |
| Least-squares formulation | Gauss-Newton or LM |
| Ill-conditioned problems | Levenberg-Marquardt |
| Small models (< 10K params) | Full BFGS |

### Tuning L-BFGS Memory Size

```python
# Small memory (3-5): Less memory, faster iterations
LBFGSConfig(memory_size=5)

# Medium memory (10-15): Good balance (default)
LBFGSConfig(memory_size=10)

# Large memory (20+): Better approximation, more memory
LBFGSConfig(memory_size=20)
```

### Handling Convergence Issues

```python
# If L-BFGS oscillates, increase line search steps
config = LBFGSConfig(
    max_linesearch_steps=40,  # Increased from 20
)

# If hybrid switch is too early, increase first_order_steps
config = HybridOptimizerConfig(
    first_order_steps=2000,  # More Adam steps
)

# If convergence stalls, try different criterion
config = HybridOptimizerConfig(
    switch_criterion=SwitchCriterion.GRADIENT_NORM,
    gradient_norm_threshold=1e-4,
)
```

### Batch Size Considerations

Second-order methods typically work best with:

- **Full batch:** Most accurate gradient/curvature estimates
- **Large mini-batch:** Good balance of noise and efficiency
- **Small mini-batch:** May cause instability in L-BFGS

```python
# For stochastic settings, use more conservative L-BFGS
config = LBFGSConfig(
    memory_size=5,  # Smaller memory
    max_linesearch_steps=10,  # Fewer line search steps
)
```

## See Also

- [Optimization Guide](../user-guide/optimization.md) - General optimization strategies
- [Training Guide](../user-guide/training.md) - Training procedures
- [GradNorm](gradnorm.md) - Multi-task loss balancing
- [API Reference](../api/optimization.md#second-order) - Complete API documentation

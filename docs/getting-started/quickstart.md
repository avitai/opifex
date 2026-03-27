# Quick Start Guide

Welcome to **Opifex**! This guide shows three core workflows: solving a PDE with PINNs, learning an operator from data, and discovering equations from trajectories.

## Prerequisites

Opifex is installed and running. If not, see the [Installation Guide](installation.md).

```bash
source ./activate.sh
```

---

## Example 1: Solving a PDE with PINNs

Solve the 1D Poisson equation: $-u''(x) = \pi^2 \sin(\pi x)$ on $[-1, 1]$ with $u(-1)=u(1)=0$.

```python
import jax.numpy as jnp
from flax import nnx
from opifex.geometry import Interval
from opifex.neural.base import StandardMLP
from opifex.solvers import PINNSolver
from opifex.solvers.pinn import PINNConfig, poisson_residual

# 1. Define the problem
geometry = Interval(-1.0, 1.0)
source_fn = lambda x: jnp.pi**2 * jnp.sin(jnp.pi * x[..., 0:1])
residual_fn = poisson_residual(source_fn)
bc_fn = lambda x: jnp.zeros(x.shape[:-1])

# 2. Create model and solve
model = StandardMLP(layer_sizes=[1, 32, 32, 1], activation="tanh", rngs=nnx.Rngs(42))
solver = PINNSolver(model)
result = solver.solve(geometry, residual_fn, bc_fn, PINNConfig(num_iterations=1000, print_every=0))

print(f"Final loss: {result.final_loss:.2e}")
print(f"Training time: {result.training_time:.1f}s")
```

```text
Final loss: 5.93e-02
Training time: 1.3s
```

---

## Example 2: Learning an Operator (FNO)

Train a Fourier Neural Operator to map input fields to output fields.

```python
import jax
from flax import nnx
from opifex.neural.operators.fno import FourierNeuralOperator
from opifex.core.training.trainer import Trainer
from opifex.core.training.config import TrainingConfig

# 1. Create synthetic data (batch, channels, height, width)
x_train = jax.random.normal(jax.random.PRNGKey(0), (100, 1, 32, 32))
y_train = jax.random.normal(jax.random.PRNGKey(1), (100, 1, 32, 32))

# 2. Create FNO model
fno = FourierNeuralOperator(
    in_channels=1, out_channels=1, hidden_channels=32,
    modes=8, num_layers=4, rngs=nnx.Rngs(42),
)

# 3. Train
trainer = Trainer(model=fno, config=TrainingConfig(num_epochs=5, batch_size=16))
_, metrics = trainer.fit(train_data=(x_train, y_train))

print(f"FNO training complete")
print(f"Input/output shape: {x_train.shape} -> {y_train.shape}")
```

```text
FNO training complete
Input/output shape: (100, 1, 32, 32) -> (100, 1, 32, 32)
```

---

## Example 3: Equation Discovery (SINDy)

Discover the Lorenz equations from trajectory data.

```python
import jax.numpy as jnp
from opifex.discovery.sindy import SINDy, SINDyConfig

# 1. Generate Lorenz trajectory with RK4
def lorenz(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

dt, state = 0.001, jnp.array([1.0, 1.0, 1.0])
trajectory, derivatives = [state], [lorenz(state)]
for _ in range(10000):
    k1 = lorenz(state)
    k2 = lorenz(state + 0.5 * dt * k1)
    k3 = lorenz(state + 0.5 * dt * k2)
    k4 = lorenz(state + dt * k3)
    state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    trajectory.append(state)
    derivatives.append(lorenz(state))

# 2. Discover governing equations
model = SINDy(SINDyConfig(polynomial_degree=2, threshold=0.3))
model.fit(jnp.stack(trajectory), jnp.stack(derivatives))

for eq in model.equations(["x", "y", "z"]):
    print(eq)
```

```text
dx/dt = -9.999 x + 10.000 y
dy/dt = 28.000 x + -1.000 y + -1.000 x z
dz/dt = -2.667 z + 1.000 x y
```

Recovers the true Lorenz coefficients ($\sigma=10$, $\rho=28$, $\beta=8/3$) to high accuracy.

---

## Next Steps

- **[Examples](../examples/index.md)**: 51 working examples across neural operators, PINNs, discovery, and more
- **[Neural Operators Guide](../methods/neural-operators.md)**: Theory and architecture details
- **[PINNs Guide](../methods/pinns.md)**: Physics-informed methods
- **[API Reference](../api/core.md)**: Full API documentation

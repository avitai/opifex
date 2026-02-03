# Quick Start Guide

Welcome to **Opifex**! This guide will get you solving differential equations in minutes using the **Unified SciMLSolver API**.

## Prerequisites

Opifex is installed and running. If not, see the [Installation Guide](installation.md).

## 🚀 Concept: The Solver Protocol

In Opifex, you don't write training loops. You define a **Problem** and pass it to a **Solver**.

```python
solution = solver.solve(problem)
```

---

## Example 1: Solving a PDE (Physics-Informed)

Let's solve the 2D Heat Equation on a Rectangle.

```python
import jax
import jax.numpy as jnp
from flax import nnx
from opifex.core.problems import create_pde_problem
from opifex.geometry import Rectangle
from opifex.neural.base import StandardMLP
from opifex.solvers import PINNSolver

# 1. Define the Physics (Poisson Equation)
def poisson_residual(x, u, u_derivatives, params):
    # Laplace Equation: u_xx + u_yy = 0
    return u_derivatives["xx"] + u_derivatives["yy"]

# 2. Define the Problem (Geometry + Physics)
problem = create_pde_problem(
    geometry=Rectangle(center=(0.5, 0.5), width=1.0, height=1.0), # Geometry Object
    equation=poisson_residual,
    parameters={},
    boundary_conditions=[{"type": "dirichlet", "value": 0.0}] # Simplified config
)

# 3. Create a PINN Model
model = StandardMLP(
    layer_sizes=[2, 32, 32, 1], # Input: 2 (x,y), Hidden: 32, Output: 1 (u)
    rngs=nnx.Rngs(42)
)

# 4. Solve!
solver = PINNSolver(model=model)
solution = solver.solve(problem)

print(f"Converged: {solution.converged}")
print(f"Final Loss: {solution.stats['loss']:.2e}")
```
```text
Converged: True
Final Loss: 4.09e-05
```
---

## Example 2: Operator Learning (Data-Driven)

Learn the mapping from initial conditions to solutions using a **Neural Operator**.

```python
from opifex.neural.operators import FourierNeuralOperator
from opifex.solvers import NeuralOperatorSolver
from opifex.core.problems import DataDrivenProblem

# 1. Load Data (e.g., existing simulation data)
# shape: (n_samples, resolution, resolution, 1)
x_train = jax.random.normal(jax.random.key(0), (100, 64, 64, 1))
y_train = jax.random.normal(jax.random.key(1), (100, 64, 64, 1))

problem = DataDrivenProblem(train_dataset=(x_train, y_train))

# 2. Create FNO Model
fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    modes=16,
    num_layers=4,
    rngs=nnx.Rngs(42)
)

# 3. Solve!
solver = NeuralOperatorSolver(model=fno)
solution = solver.solve(problem)

print(f"Training Complete. Validation Metrics: {solution.metrics}")
```
```text
Training Complete. Validation Metrics: {'final_train_loss': 1.002, 'avg_epoch_time': 0.045}
```
---

## Example 3: Uncertainty Quantification

Any solver can be wrapped for uncertainty quantification. Here we use an **Ensemble**.

```python
from opifex.solvers import PINNSolver, EnsembleWrapper

# 1. Create multiple solvers (with different random seeds)
solvers = [
    PINNSolver(model=StandardMLP(layer_sizes=[2, 32, 32, 1], rngs=nnx.Rngs(i)))
    for i in range(3)
]

# 2. Wrap them
ensemble = EnsembleWrapper(solvers=solvers)

# 3. Solve! (Returns mean and standard deviation)
solution = ensemble.solve(problem)

u_mean = solution.fields["u_mean"]
u_std = solution.fields["u_std"]

print(f"UQ Complete. Mean field shape: {u_mean.shape}")
```
```text
UQ Complete. Mean field shape: (1000, 1)
```
---

## 🔧 Going Deeper?

- **[Benchmarks](../benchmarks.md)**: See how PINNs compare to Operators.
- **[Generative AI](../methods/probabilistic.md)**: Use `ArtifexSolverAdapter` for diffusion models.
- **[API Reference](../api/core.md)**: Full documentation.

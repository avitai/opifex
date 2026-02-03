# PDE Solving Examples

## Heat Equation

### 2D Heat Equation with Dirichlet Boundaries

```python
import jax
import jax.numpy as jnp
from opifex.core.problems import create_pde_problem
from opifex.neural.pinns import create_heat_equation_pinn
from opifex.training.basic_trainer import BasicTrainer
from opifex.core.training.config import TrainingConfig
from opifex.geometry import Rectangle

# Define heat equation residual
def heat_equation(x, y, t, u, u_derivatives, params):
    """Heat equation: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)"""
    alpha = params.get('diffusivity', 0.01)
    u_t = u_derivatives['t']
    u_xx = u_derivatives['xx']
    u_yy = u_derivatives['yy']
    return u_t - alpha * (u_xx + u_yy)

# Define domain and boundary conditions
domain = {"x": (0, 1), "y": (0, 1), "t": (0, 1)}
boundary_conditions = [
    {"type": "dirichlet", "boundary": "all", "value": 0.0}
]

# Create PDE problem
problem = create_pde_problem(
    domain=domain,
    equation=heat_equation,
    boundary_conditions=boundary_conditions,
    initial_conditions={"u": lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)},
    parameters={"diffusivity": 0.01}
)

# Create specialized heat equation PINN
pinn = create_heat_equation_pinn(
    layers=[3, 50, 50, 50, 1],
    domain=domain,
    diffusivity=0.01
)

# Train
config = TrainingConfig(optimizer="adam", learning_rate=1e-3, num_epochs=10000)
trainer = BasicTrainer(model=pinn, training_config=config)
history = trainer.train()
```

## Wave Equation

### 1D Wave Equation

```python
# ∂²u/∂t² = c²∂²u/∂x²
def wave_pde(x, t, u, params):
    c = params['wave_speed']
    u_tt = jax.grad(jax.grad(u, argnums=1), argnums=1)(x, t)
    u_xx = jax.grad(jax.grad(u, argnums=0), argnums=0)(x, t)
    return u_tt - c**2 * u_xx
```

## Burgers' Equation

### Viscous Burgers' Equation

```python
# ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
def burgers_pde(x, t, u, params):
    nu = params['viscosity']
    u_t = jax.grad(u, argnums=1)(x, t)
    u_x = jax.grad(u, argnums=0)(x, t)
    u_xx = jax.grad(jax.grad(u, argnums=0), argnums=0)(x, t)
    return u_t + u * u_x - nu * u_xx
```

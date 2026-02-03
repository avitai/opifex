# Fluid Dynamics Examples

## Darcy Flow Equation

### Complete FNO Implementation

Opifex includes a comprehensive implementation of Fourier Neural Operators for solving Darcy flow problems. This example demonstrates operator learning for the 2D elliptic PDE:

**∇·(a(x)∇u(x)) = f(x)** in Ω with homogeneous Dirichlet boundary conditions.

```bash
# Complete example available at: examples/darcy_fno_opifex.py
from opifex.neural.operators.foundations import FourierNeuralOperator
from opifex.training.basic_trainer import BasicTrainer, TrainingConfig

# Run the complete example
python examples/darcy_fno_opifex.py

# This will:
# 1. Generate synthetic Darcy flow dataset with realistic permeability fields
# 2. Train an FNO model using Opifex infrastructure
# 3. Save organized results to timestamped directories
# 4. Provide comprehensive visualization and analysis
```

**Key Features:**

- **Automated Dataset Generation**: Creates realistic permeability coefficient fields using Fourier modes
- **Vectorized Solver**: Efficient finite difference Darcy equation solver using JAX vectorization
- **Production Training**: Full training pipeline with error recovery and monitoring
- **Organized Output**: Timestamped result directories under `examples_output/`
- **Comprehensive Analysis**: Visual comparisons and detailed error statistics

**Results Structure:**

```
examples_output/darcy_fno_run_YYYYMMDD_HHMMSS/
├── darcy_fno_results.png       # Input/target/prediction visualizations
├── darcy_training_curves.png   # Training and validation loss curves
├── error_statistics.txt        # Mean/max absolute and relative errors
└── training_data.txt           # Complete training history
```

**Quick Start:**

```bash
# Activate Opifex environment
source ./activate.sh

# Run the example
cd examples/
python darcy_fno_opifex.py

# Results automatically saved to organized directories
```

## Navier-Stokes Equations

### Lid-Driven Cavity

```python
from opifex.neural.pinns import create_navier_stokes_pinn
from opifex.core.problems import PDEProblem
from opifex.core.conditions import DirichletBC
from opifex.core.training.trainer import Trainer
from opifex.core.training.config import TrainingConfig
import flax.nnx as nnx
import jax
import jax.numpy as jnp

# Define Navier-Stokes Problem
class NavierStokesProblem(PDEProblem):
    """2D incompressible Navier-Stokes equations."""

    def __init__(self, reynolds_number=100):
        domain = {
            "x": (0.0, 1.0),
            "y": (0.0, 1.0),
            "t": (0.0, 1.0)
        }

        # Boundary conditions (Lid-driven cavity)
        boundary_conditions = [
            DirichletBC(boundary="top", value=jnp.array([1.0, 0.0])),    # moving lid
            DirichletBC(boundary="bottom", value=jnp.array([0.0, 0.0])), # no-slip
            DirichletBC(boundary="left", value=jnp.array([0.0, 0.0])),   # no-slip
            DirichletBC(boundary="right", value=jnp.array([0.0, 0.0]))   # no-slip
        ]

        super().__init__(
            domain=domain,
            equation=self._navier_stokes,
            boundary_conditions=boundary_conditions,
            parameters={"Re": reynolds_number},
            time_dependent=True
        )

    def residual(self, x, u, u_derivatives):
        """Navier-Stokes residual."""
        Re = self.parameters["Re"]
        u_vel, v_vel, p = u[..., 0], u[..., 1], u[..., 2]

        u_t = u_derivatives["t"][..., 0]
        v_t = u_derivatives["t"][..., 1]
        u_x, u_y = u_derivatives["x"][..., 0], u_derivatives["y"][..., 0]
        v_x, v_y = u_derivatives["x"][..., 1], u_derivatives["y"][..., 1]
        u_xx, u_yy = u_derivatives["xx"][..., 0], u_derivatives["yy"][..., 0]
        v_xx, v_yy = u_derivatives["xx"][..., 1], u_derivatives["yy"][..., 1]
        p_x, p_y = u_derivatives["x"][..., 2], u_derivatives["y"][..., 2]

        momentum_x = u_t + u_vel * u_x + v_vel * u_y + p_x - (1/Re) * (u_xx + u_yy)
        momentum_y = v_t + u_vel * v_x + v_vel * v_y + p_y - (1/Re) * (v_xx + v_yy)
        continuity = u_x + v_y

        return jnp.stack([momentum_x, momentum_y, continuity], axis=-1)

    def _navier_stokes(self, x, u, u_derivatives, params):
        return self.residual(x, u, u_derivatives)

# Create PINN
rngs = nnx.Rngs(42)
pinn = create_navier_stokes_pinn(spatial_dim=2, rngs=rngs)

# Create Problem
problem = NavierStokesProblem(reynolds_number=100)

# Train (Simplified for demonstration)
print("Navier-Stokes PINN and Problem defined successfully.")
# In a real scenario:
# trainer = Trainer(model=pinn, config=TrainingConfig(num_epochs=1000))
# trainer.train(problem=problem)
```

## Flow Around Cylinder

```python
# Define geometry with cylinder obstacle
domain = FlowDomain(
    inlet=Rectangle([-2, 0], [0, 1]),
    outlet=Rectangle([2, 4], [0, 1]),
    obstacle=Circle([0, 0.5], radius=0.1)
)

# Inlet velocity profile
def inlet_velocity(y):
    return 4 * y * (1 - y)  # Parabolic profile
```

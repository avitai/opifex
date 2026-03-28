# Frequently Asked Questions

```python
from opifex.neural.operators.fno import FourierNeuralOperator
```

## Getting Started

### What is Opifex?

Opifex is a scientific machine learning framework that combines machine learning with scientific computing to solve problems in physics, engineering, and other scientific domains. It provides implementations of neural operators, physics-informed neural networks (PINNs), advanced optimization algorithms, and other modern scientific ML methods -- all built on JAX and Flax NNX.

### How do I install Opifex?

See the [installation guide](getting-started/installation.md) for detailed instructions. The basic installation is:

```bash
git clone https://github.com/avitai/opifex.git
cd opifex
./setup.sh
```

This runs the setup script, which creates a `.venv` and installs all dependencies. After setup, activate the environment:

```bash
source ./activate.sh
```

To install in editable mode manually:

```bash
uv pip install -e .
```

### What are the system requirements?

**Minimum Requirements:**

- Python 3.11+
- JAX 0.8.0+
- NumPy 1.24+
- Flax 0.12.0+

**Recommended:**

- CUDA-compatible GPU with 8GB+ VRAM
- 16GB+ system RAM

**Optional Dependencies:**

- CUDA Toolkit 12.0+ (for GPU acceleration)
- cuDNN 8.9+ (for optimized neural network operations)

### How does Opifex compare to other scientific ML frameworks?

Opifex distinguishes itself through:

- **Full Integration**: Unified framework covering neural operators, PINNs, optimization, and more
- **JAX-Native**: Built on JAX for high performance and automatic differentiation
- **Physics-First Design**: Deep integration of physical constraints and conservation laws
- **Flax NNX**: Modern Flax NNX patterns throughout, with proper RNG handling and state management

### What scientific domains does Opifex support?

Opifex supports a wide range of scientific applications:

- **Computational Fluid Dynamics**: Navier-Stokes, turbulence, Burgers equation
- **Heat Transfer**: Diffusion, advection-diffusion
- **Structural Mechanics**: Elasticity, Euler beam problems
- **Wave Propagation**: Wave equation, Helmholtz equation
- **Quantum Chemistry**: DFT, electronic structure via NeuralDFT
- **Materials Science**: Allen-Cahn, diffusion-reaction systems

## Neural Operators

### What are neural operators and when should I use them?

Neural operators learn mappings between function spaces, making them ideal for:

- **PDE Families**: Solving multiple related PDEs with different parameters
- **Multi-Resolution**: Working with varying discretizations
- **Parameter Studies**: Exploring parameter spaces efficiently
- **Real-Time Simulation**: Fast inference for interactive applications

Use neural operators when you need to solve many similar problems or when traditional solvers are too slow.

### Which neural operator should I choose?

**Fourier Neural Operators (FNO):**

- Best for: Regular grids, periodic problems, global interactions
- Examples: Turbulence, Darcy flow, wave propagation

```python
from opifex.neural.operators.fno import FourierNeuralOperator
from flax import nnx

rngs = nnx.Rngs(42)
fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    modes=16,
    num_layers=4,
    rngs=rngs,
)
```

**DeepONet:**

- Best for: Explicit function-to-function mappings, irregular sampling
- Examples: Antiderivatives, Green's functions, response operators

```python
from opifex.neural.operators.deeponet import DeepONet
from flax import nnx

rngs = nnx.Rngs(42)
deeponet = DeepONet(
    branch_sizes=[100, 64, 64, 32],
    trunk_sizes=[2, 64, 64, 32],
    activation="gelu",
    rngs=rngs,
)
```

**Graph Neural Operators (GNO):**

- Best for: Irregular geometries, unstructured meshes, complex domains
- Examples: Finite element problems, molecular systems

```python
from opifex.neural.operators.graph.gno import GraphNeuralOperator
from flax import nnx

rngs = nnx.Rngs(42)
gno = GraphNeuralOperator(
    node_dim=3,
    hidden_dim=64,
    num_layers=4,
    edge_dim=2,
    rngs=rngs,
)
```

### Can neural operators handle time-dependent problems?

Yes, neural operators excel at time-dependent problems:

- **Space-Time Operators**: Treat time as another spatial dimension
- **Sequential Prediction**: Use operators for time-stepping
- **Autoregressive**: Feed predictions back as inputs for multi-step rollout

## Physics-Informed Neural Networks (PINNs)

### When should I use PINNs vs traditional solvers?

**Use PINNs when:**

- Limited or noisy data available
- Complex geometries or boundary conditions
- Inverse problems (parameter identification)
- Multi-physics coupling required

**Use traditional solvers when:**

- Well-established, validated methods exist
- High accuracy requirements with known convergence
- Computational resources are unlimited

### How do I set up a PINN in Opifex?

Opifex provides `PINNSolver` in `opifex.solvers.pinn` for high-level PINN usage, along with the `PhysicsLossConfig` for configuring loss weights:

```python
from opifex.core.physics.losses import PhysicsLossConfig

# Configure physics loss weights
config = PhysicsLossConfig(
    data_loss_weight=1.0,
    physics_loss_weight=0.1,
    boundary_loss_weight=1.0,
)
```

For loss balancing, `PhysicsLossConfig` supports adaptive weighting:

```python
config = PhysicsLossConfig(
    data_loss_weight=1.0,
    physics_loss_weight=0.01,
    boundary_loss_weight=10.0,
    adaptive_weighting=True,
    weight_schedule="exponential",
)
```

You can also use the `GradNormBalancer` for automatic multi-loss balancing:

```python
from opifex.core.physics.gradnorm import GradNormBalancer, GradNormConfig

balancer = GradNormBalancer(
    num_losses=3,
    config=GradNormConfig(alpha=1.5),
    rngs=nnx.Rngs(0),
)
```

### Why is my PINN not converging?

Common convergence issues and solutions:

**1. Poor Collocation Point Distribution:**

Use adaptive refinement. Opifex provides `RARDRefiner` for residual-based adaptive refinement:

```python
from opifex.core.physics.adaptive_methods import RARDRefiner, RARDConfig

refiner = RARDRefiner(RARDConfig(num_new_points=50))
new_points = refiner.refine(points, residuals, bounds, key)
```

**2. Activation Function Choice:**

```python
# For smooth problems requiring derivatives
activation = "tanh"  # Good for derivatives

# For better gradient flow
activation = "gelu"  # or "silu"

# Avoid for high-order derivatives
activation = "relu"  # Can cause issues
```

**3. Network Architecture:**

Use `StandardMLP` as the network backbone:

```python
from opifex.neural.base import StandardMLP
from flax import nnx

model = StandardMLP(
    layer_sizes=[2, 50, 50, 50, 50, 1],
    activation="tanh",
    rngs=nnx.Rngs(0),
)
```

**4. Learning Rate and Optimization:**

```python
# Use learning rate scheduling via optax
import optax

schedule = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=1000,
    decay_rate=0.9,
)
optimizer = optax.adam(schedule)
```

### How do I solve inverse problems with PINNs?

Inverse problems involve learning unknown PDE parameters from data. Use `nnx.Param` for the unknown parameter with a log-transform for positivity:

```python
import jax.numpy as jnp
from flax import nnx

class InversePINN(nnx.Module):
    def __init__(self, layer_sizes, *, rngs):
        from opifex.neural.base import StandardMLP
        self.net = StandardMLP(layer_sizes, activation="tanh", rngs=rngs)
        # Log-transform for positivity
        self.log_C = nnx.Param(jnp.log(jnp.array(2.0)))  # Initial guess

    @property
    def diffusion_coef(self):
        return jnp.exp(self.log_C.value)

    def __call__(self, x):
        return self.net(x)
```

Key considerations:

- Use high weights for measurement data loss
- Validate with synthetic data first
- Use multiple measurement locations
- See `examples/pinns/inverse_diffusion.py` for a full example

## Optimization

### What optimization algorithms does Opifex provide?

**Meta-Optimization:**

- MAML (Model-Agnostic Meta-Learning)
- Reptile

**Learn-to-Optimize (L2O):**

- Parametric programming solvers (`ParametricProgrammingSolver`)
- Multi-objective optimization (`MultiObjectiveL2OEngine`)
- RL-based optimization strategy selection (`RLOptimizationEngine`)
- Adaptive schedulers (`PerformanceAwareScheduler`, `BayesianSchedulerOptimizer`)

**Second-Order Methods:**

- Hybrid optimizer (`opifex.optimization.second_order.hybrid_optimizer`)

### How do I use the standard optimizer pattern?

Opifex uses standard Optax and Flax NNX patterns:

```python
import optax
from flax import nnx

model = MyModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

# Training step
(loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
optimizer.update(model, grads)
```

Or use the `Trainer` for managed training:

```python
from opifex.core.training import Trainer, TrainingConfig

config = TrainingConfig(num_epochs=100, learning_rate=1e-3, batch_size=32)
trainer = Trainer(model=model, config=config, rngs=nnx.Rngs(0))

trained_model, metrics = trainer.fit(
    train_data=(x_train, y_train),
    val_data=(x_val, y_val),
)
```

## Performance and Scalability

### How do I optimize Opifex for GPU performance?

**GPU Optimization Strategies:**

```python
# 1. Enable JIT compilation
@jax.jit
def train_step(model, x, y):
    return loss_and_gradients(model, x, y)

# 2. Use appropriate batch sizes
batch_size = 32  # Start here, adjust based on memory

# 3. Gradient checkpointing for memory
config = TrainingConfig(gradient_checkpointing=True)
```

**Memory Management:**

```python
# Monitor GPU memory
import jax
print(f"GPU memory: {jax.devices()[0].memory_stats()}")

# Clear JAX cache if needed
jax.clear_caches()
```

### Why is my training slow and how can I speed it up?

**Common Performance Issues:**

**1. Missing Vectorization:**

```python
# Slow: Computing residuals sequentially
for point in collocation_points:
    residual = compute_pde_residual(point)

# Fast: Vectorized computation
residuals = jax.vmap(compute_pde_residual)(collocation_points)
```

**2. Missing JIT Compilation:**

```python
# Add @jax.jit to training functions
@jax.jit
def train_step(model, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    return loss, grads
```

**3. Large Batch Sizes:**

```python
# Use gradient accumulation instead of large batches
effective_batch_size = 128
mini_batch_size = 32
accumulation_steps = effective_batch_size // mini_batch_size
```

## Data and Preprocessing

### How do I prepare data for neural operators?

Opifex provides built-in data loaders for common PDE benchmarks:

```python
from opifex.data import (
    create_darcy_loader,
    create_burgers_loader,
    create_navier_stokes_loader,
    create_shallow_water_loader,
)

# Load Darcy flow data (elliptic PDE)
loader = create_darcy_loader()

# Load Burgers equation data (1D/2D)
loader = create_burgers_loader()
```

**Data Requirements for custom datasets:**

- Input-output function pairs as JAX arrays
- Consistent discretization for FNO (regular grids)
- Sufficient parameter coverage for generalization

### How do I handle irregular geometries and meshes?

For irregular geometries, use the Graph Neural Operator with `grid_to_graph_data`:

```python
from opifex.geometry.topology.graphs import grid_to_graph_data, graph_to_grid

# Convert grid to graph representation
graph_data = grid_to_graph_data(grid, connectivity="radius")

# Process with GNO
output = gno(graph_data.node_features, graph_data.edge_indices, graph_data.edge_features)

# Convert back to grid
grid_output = graph_to_grid(output, H, W)
```

## Integration and Deployment

### How do I deploy Opifex models in production?

Opifex provides a FastAPI-based serving infrastructure:

```python
from opifex.deployment.core_serving import DeploymentConfig

# Configure deployment
config = DeploymentConfig(
    model_name="darcy_fno",
    model_type="neural_operator",
    serving_port=8080,
    batch_size=32,
    gpu_enabled=True,
    precision="float32",
)
```

The server module (`opifex.deployment.server`) provides a FastAPI application with model registry, inference engine, health checks, and CORS support. See `src/opifex/deployment/server.py` for details.

## Troubleshooting

### Common Error Messages and Solutions

#### "CUDA out of memory"

**Solutions:**

```python
# 1. Reduce batch size
batch_size = batch_size // 2

# 2. Enable gradient checkpointing
config = TrainingConfig(gradient_checkpointing=True)

# 3. Clear JAX cache
jax.clear_caches()
```

#### "NaN in loss function"

**Debugging Steps:**

```python
# 1. Check input data
assert not jnp.any(jnp.isnan(input_data))

# 2. Reduce learning rate
learning_rate = learning_rate * 0.1

# 3. Add gradient clipping via optax
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-4),
)
```

#### "Slow convergence in PINNs"

**Solutions:**

```python
# 1. Use adaptive collocation point refinement
from opifex.core.physics.adaptive_methods import RARDRefiner, RARDConfig
refiner = RARDRefiner(RARDConfig(num_new_points=50))

# 2. Adjust loss weights
config = PhysicsLossConfig(
    physics_loss_weight=1.0,
    boundary_loss_weight=100.0,  # Increase for better BC satisfaction
)

# 3. Try different activation functions
activation = "gelu"  # Often better than tanh for gradient flow
```

## Advanced Topics

### How do I implement uncertainty quantification?

Opifex provides Bayesian layers and uncertainty quantification tools:

```python
from opifex.neural.bayesian.layers import BayesianLayer
from opifex.neural.bayesian import UncertaintyQuantifier
from flax import nnx
import jax.numpy as jnp

# BayesianLayer has variational weight distributions
layer = BayesianLayer(
    in_features=10,
    out_features=64,
    prior_std=1.0,
    rngs=nnx.Rngs(42),
)

# Forward pass samples from weight posterior during training
output = layer(x, training=True, sample=True)

# For mean prediction (no sampling)
output_mean = layer(x, training=False, sample=False)
```

For full probabilistic neural operators, use `AmortizedVariationalFramework`:

```python
from opifex.neural.bayesian import AmortizedVariationalFramework, VariationalConfig

config = VariationalConfig(input_dim=10, hidden_dims=(64, 32), num_samples=10)
framework = AmortizedVariationalFramework(
    base_model=model,
    config=config,
    rngs=nnx.Rngs(42),
)
```

### How do I use NTK analysis?

Opifex provides neural tangent kernel analysis for diagnosing spectral bias:

```python
from opifex.core.physics.ntk import NTKWrapper

wrapper = NTKWrapper(model)
ntk_matrix = wrapper.compute_ntk(x)
eigenvalues = wrapper.compute_eigenvalues(x)
bias_info = wrapper.detect_spectral_bias()
```

## Contributing and Development

### How can I contribute to Opifex?

**Development Workflow:**

```bash
# 1. Fork and clone the repository
git clone https://github.com/yourusername/opifex.git
cd opifex

# 2. Run setup script (creates .venv and installs dependencies)
./setup.sh

# 3. Activate the environment
source ./activate.sh

# 4. Create feature branch
git checkout -b feature/new-algorithm

# 5. Make changes and test
uv run pytest tests/
uv run pytest --cov=src/opifex tests/  # With coverage

# 6. Run pre-commit hooks
uv run pre-commit run --all-files

# 7. Submit pull request
git push origin feature/new-algorithm
```

### How do I ensure reproducibility?

```python
import jax
import numpy as np
from flax import nnx

# Set random seeds
np.random.seed(42)
key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(42)

# Use deterministic training config
from opifex.core.training import TrainingConfig
config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    batch_size=32,
)
```

## Resources and Learning

### Where can I find more examples and tutorials?

**In-Repository Resources:**

- [Examples Directory](https://github.com/avitai/opifex/tree/main/examples) -- runnable Python scripts and Jupyter notebooks
- [Documentation](https://github.com/avitai/opifex/tree/main/docs) -- method guides, API reference, user guide

**Academic Papers:**

- Physics-Informed Neural Networks: [Raissi et al. 2019](https://doi.org/10.1016/j.jcp.2018.10.045)
- Neural Operators: [Li et al. 2021](https://openreview.net/forum?id=c8P9NQVtmnO)
- DeepONet: [Lu et al. 2021](https://doi.org/10.1038/s42256-021-00302-5)

### What are the recommended learning resources?

**Books:**

- "Physics-Informed Machine Learning" by Karniadakis et al.
- "Data-Driven Science and Engineering" by Brunton & Kutz

**Courses:**

- MIT 18.337: Parallel Computing and Scientific Machine Learning
- Stanford CS229: Machine Learning

**Workshops and Conferences:**

- NeurIPS Workshop on Machine Learning and the Physical Sciences
- ICML Workshop on Scientific Machine Learning
- SIAM Conference on Computational Science and Engineering

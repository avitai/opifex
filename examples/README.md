# Opifex Examples

This directory contains comprehensive examples demonstrating the Opifex framework capabilities for scientific machine learning applications.

## 🚀 Available Examples

### 1. DISCO Convolutions (`layers/disco_convolutions_example.py`) ✅ **WORKING**

**Advanced discrete-continuous convolutions with 6x+ speedup on regular grids.**

**Demonstrates:**

- Basic DISCO convolution for structured and unstructured grids
- Equidistant optimization with significant performance improvements
- Complete encoder-decoder architectures for multi-scale processing
- Continuous kernel parameterization for flexible spatial processing
- Real-time performance comparison and speedup measurements

**Usage:**

```bash
# Activate environment first
source ./activate.sh

# Run DISCO convolutions demonstration
python layers/disco_convolutions_example.py

# View comprehensive visualizations and performance metrics
```

**Expected Output:**

```
🎯 DISCO Convolutions Example - Opifex Framework
============================================================
✅ Basic DISCO convolution: 954.9ms
⚡ Equidistant optimization: 9.92x speedup
🏗️ Encoder-decoder error: 6.04e+03
📊 Comprehensive visualization created
```

### 2. Grid Embeddings (`layers/grid_embeddings_example.py`) ✅ **WORKING**

**Advanced coordinate injection and positional encoding for enhanced spatial awareness.**

**Demonstrates:**

- Grid Embedding 2D with coordinate injection for structured grids
- N-dimensional embeddings for 1D, 2D, and 3D spatial problems
- Sinusoidal embeddings with frequency-based encoding
- Methods comparison with performance analysis
- Complete visualization of coordinate systems and embeddings

**Usage:**

```bash
# Activate environment first
source ./activate.sh

# Run grid embeddings demonstration
python layers/grid_embeddings_example.py

# Explore N-dimensional spatial awareness capabilities
```

**Expected Output:**

```
🔲 Opifex Grid Embeddings Layer Example
======================================================================
✅ Grid 2D: (8, 64, 64, 3) -> (8, 64, 64, 5)
✅ Grid 3D: (4, 32, 32, 32, 2) -> (4, 32, 32, 32, 5)
✅ Sinusoidal: (4, 4096, 2) -> (4, 4096, 64)
```

### 3. Darcy Flow FNO (`darcy_fno_opifex.py`) ✅ **WORKING**

**Complete Fourier Neural Operator implementation for Darcy flow equation.**

**Demonstrates:**

- Automated synthetic dataset generation for Darcy's equation: ∇·(a(x)∇u(x)) = f(x)
- Realistic permeability field generation using Fourier modes
- Vectorized finite difference PDE solver using JAX
- Full FNO training pipeline with Opifex framework
- Organized results output with timestamped directories
- Comprehensive visualization and error analysis

**Usage:**

```bash
# Activate Opifex environment
source ./activate.sh

# Run the example
cd examples/
python darcy_fno_opifex.py

# Results automatically saved to organized directories
```

**Expected Output:**

```
============================================================
Training FNO on Darcy Flow - Opifex Implementation
============================================================
Model has 69,857 parameters
Starting training for 20 epochs...
Epoch   0: Train Loss = 0.034057, Test Loss = 0.034095
...
Final test MSE: 0.033722
```

**Output Structure:**

```
examples_output/darcy_fno_run_YYYYMMDD_HHMMSS/
├── darcy_fno_results.png       # Input/target/prediction visualizations
├── darcy_training_curves.png   # Training and validation loss curves
├── error_statistics.txt        # Mean/max absolute and relative errors
└── training_data.txt           # Complete training loss history
```

**Features:**

- ✅ **Real PDE Solving**: Actual Darcy flow equation with physics-based data generation
- ✅ **Production Pipeline**: Error handling, monitoring, and organized output
- ✅ **Comprehensive Analysis**: Visual comparisons and quantitative error metrics
- ✅ **Modular Design**: Easily adaptable to other PDE operator learning problems

### 4. Neural Operators Comprehensive Demo (`neural_operators_comprehensive_demo.py`) ⚠️ **PARTIAL**

**Complete demonstration of neural operator architectures with practical applications.**

**Note**: This example may encounter GPU memory issues with certain operators (UQNO). The basic functionality works correctly.

**Demonstrates:**

- **FNO Variants**: TFNO, U-FNO, SFNO, Local FNO, AM-FNO with real-world scenarios
- **Specialized Operators**: GINO, MGNO for geometry and molecular dynamics
- **Operator Factory System**: Intelligent operator selection and recommendation engine
- **Performance Comparisons**: Parameter efficiency, memory usage, and accuracy analysis
- **Multi-Domain Applications**: Climate modeling, turbulent flow, CAD geometry

**Usage:**

```bash
# Run the comprehensive demo (may encounter GPU memory issues)
python neural_operators_comprehensive_demo.py

# For testing individual components, use smaller examples
```

### 5. Enhanced Calibration Demo (`enhanced_calibration_demo.py`)

**Advanced uncertainty quantification and calibration framework demonstration.**

**Demonstrates:**

- Multiple calibration methods (Platt scaling, temperature scaling, isotonic regression)
- Uncertainty decomposition (epistemic and aleatoric)
- Physics-aware temperature scaling with constraints
- Comprehensive uncertainty quality assessment

**Usage:**

```bash
# Activate environment first
source ./activate.sh

python enhanced_calibration_demo.py
```

## 🧪 Testing Framework Components

### Learn-to-Optimize (L2O) Framework ✅ **158/158 TESTS PASSING**

**Test the complete L2O framework:**

```bash
# Test all L2O components (158 tests)
uv run pytest tests/optimization/l2o/ -v

# Test specific L2O components
uv run pytest tests/optimization/l2o/test_parametric_solver.py -v     # Parametric programming
uv run pytest tests/optimization/l2o/test_l2o_engine.py -v            # Core L2O engine
uv run pytest tests/optimization/l2o/test_advanced_meta_learning.py -v # MAML, Reptile
uv run pytest tests/optimization/l2o/test_multi_objective.py -v        # Multi-objective opt
uv run pytest tests/optimization/l2o/test_adaptive_schedulers.py -v    # Adaptive scheduling
uv run pytest tests/optimization/l2o/test_rl_optimization.py -v        # RL-based optimization
```

**Demonstrates:**

- Parametric programming solvers with constraint handling
- Meta-learning algorithms: MAML, Reptile, gradient-based optimization
- Multi-objective optimization with Pareto frontier approximation
- Reinforcement learning-based optimization strategy selection
- Adaptive learning rate scheduling with performance monitoring
- Complete neural meta-optimization pipeline

### MLOps Integration ✅ **11/11 TESTS PASSING**

**Test the complete MLOps framework:**

```bash
# Test all MLOps components (11 tests)
uv run pytest tests/mlops/ -v

# Test specific MLOps components
uv run pytest tests/mlops/test_experiment.py -v          # Experiment management
uv run pytest tests/mlops/test_experiment_tracking.py -v # Multi-backend tracking
```

**Demonstrates:**

- Multi-backend experiment tracking (MLflow, Wandb, Neptune, Opifex)
- Physics-informed metadata for scientific computing domains
- Domain-specific metrics (Neural Operators, L2O, Neural DFT, PINNs, Quantum)
- Enterprise security with Keycloak authentication
- Production deployment with Kubernetes infrastructure

### Advanced Benchmarking System ✅ **25/25 TESTS PASSING**

**Test the complete benchmarking framework:**

```bash
# Test all benchmarking components (25 tests)
uv run pytest tests/benchmarking/ -v

# Test specific benchmarking components
uv run pytest tests/benchmarking/test_benchmark_registry.py -v    # Domain configuration
uv run pytest tests/benchmarking/test_validation_framework.py -v # Physics validation
uv run pytest tests/benchmarking/test_analysis_engine.py -v      # Statistical analysis
uv run pytest tests/benchmarking/test_results_manager.py -v      # Publication output
uv run pytest tests/benchmarking/test_benchmark_runner.py -v     # End-to-end workflow
```

**Demonstrates:**

- Domain-specific benchmarking for quantum chemistry, fluid dynamics, materials science
- Statistical analysis with bootstrap confidence intervals and significance testing
- Multi-operator comparison (FNO, DeepONet, PINNs) with performance insights
- Publication-ready output with LaTeX/HTML table generation
- Chemical accuracy validation (<1 kcal/mol) for quantum chemistry applications
- End-to-end workflow orchestration with component integration

## 🧠 Quick Code Examples

### Basic Neural Network

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.neural.base import StandardMLP

# Create a simple neural network
key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(key)

model = StandardMLP(
    layer_sizes=[3, 32, 32, 1],
    activation="tanh",
    rngs=rngs
)

# Forward pass
x = jax.random.normal(key, (10, 3))
y = model(x)
print(f"✅ Input: {x.shape}, Output: {y.shape}")
```

### Fourier Neural Operator

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.neural.operators.fno import FourierNeuralOperator

# Create FNO for PDE solving
key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(key)

fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    modes=8,
    num_layers=4,
    rngs=rngs
)

# Forward pass with 2D spatial data
x = jax.random.normal(key, (4, 1, 64, 64))
y = fno(x)
print(f"✅ FNO: {x.shape} -> {y.shape}")
```

### DISCO Convolutions

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.neural.operators.specialized import DiscreteContinuousConv2d

# Create DISCO convolution
key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(key)

disco_conv = DiscreteContinuousConv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=5,
    activation=nnx.gelu,
    rngs=rngs
)

# Forward pass
x = jax.random.normal(key, (8, 64, 64, 3))
y = disco_conv(x)
print(f"✅ DISCO: {x.shape} -> {y.shape}")
```

### Training Infrastructure

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.neural.base import StandardMLP
from opifex.core.training.trainer import Trainer
from opifex.core.training.config import TrainingConfig

# Create model and training setup
key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(key)

model = StandardMLP(
    layer_sizes=[1, 32, 32, 1],
    activation="tanh",
    rngs=rngs
)

# Training configuration
config = TrainingConfig(
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-3
)

# Create trainer
trainer = Trainer(model=model, config=config)
print("✅ Training infrastructure ready!")
```

## 🛠️ Running Examples

### Prerequisites

Ensure you have completed the Opifex installation and environment setup:

```bash
# One-time setup
./setup.sh
source ./activate.sh

# Verify installation
uv run pytest tests/ -v
```

### Environment Activation

**Always activate the Opifex environment before running examples:**

```bash
source ./activate.sh
```

### Running Individual Examples

```bash
# Navigate to examples directory
cd examples/

# Run specific examples
python darcy_fno_opifex.py                    # FNO with Darcy flow
python layers/disco_convolutions_example.py  # DISCO convolutions
python layers/grid_embeddings_example.py     # Grid embeddings
python enhanced_calibration_demo.py          # Uncertainty quantification

# Run quick tests
python -c "
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.neural.base import StandardMLP

key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(key)
model = StandardMLP(layer_sizes=[3, 32, 1], activation='tanh', rngs=rngs)
x = jax.random.normal(key, (10, 3))
y = model(x)
print(f'✅ Opifex working! Input: {x.shape}, Output: {y.shape}')
"
```

### Testing Framework Components

```bash
# Test neural networks
uv run pytest tests/neural/test_base.py -v

# Test neural operators
uv run pytest tests/neural/test_operators.py -v

# Test training infrastructure
uv run pytest tests/training/test_basic_trainer.py -v

# Test complete L2O framework (158 tests)
uv run pytest tests/optimization/l2o/ -v

# Test MLOps integration (11 tests)
uv run pytest tests/mlops/ -v

# Test benchmarking system (25 tests)
uv run pytest tests/benchmarking/ -v
```

## 🔧 Troubleshooting

### Common Issues

1. **GPU Memory Issues**: Some examples (like the comprehensive neural operators demo) may encounter GPU memory issues. Try reducing batch sizes or using CPU-only mode.

2. **Environment Not Activated**: Always run `source ./activate.sh` before running examples.

3. **Missing Dependencies**: If you encounter import errors, ensure you've run `./setup.sh` and activated the environment.

### GPU Memory Management

```bash
# For GPU memory issues, try CPU-only mode
JAX_PLATFORM_NAME=cpu python your_example.py

# Or use smaller batch sizes in the examples
```

### Getting Help

- Check the [main documentation](../docs/)
- Review the [API reference](../docs/api/)
- Look at the [development guide](../docs/development/)
- Run tests to verify your installation: `uv run pytest tests/ -v`

## 📊 Performance Notes

- **DISCO Convolutions**: Provide 6x+ speedup on regular grids
- **Grid Embeddings**: Add minimal computational overhead while enhancing spatial awareness
- **FNO**: Efficient for PDE solving with spectral methods
- **Training Infrastructure**: Supports both CPU and GPU with automatic optimization

## 🎯 Next Steps

After running these examples, you can:

1. **Explore the API**: Check out the detailed API documentation
2. **Build Custom Models**: Use the framework components to build your own models
3. **Deploy to Production**: Use the deployment guides for production setup
4. **Contribute**: See the development guide for contributing to the framework

---

**Ready to explore?** Start with the basic examples and work your way up to the more complex demonstrations!

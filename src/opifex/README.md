# Opifex Package Structure

This directory contains the Opifex framework implementation organized into modular components. Opifex is a JAX-native platform for scientific machine learning, supporting neural operators, physics-informed neural networks, equation discovery, uncertainty quantification, and more.

## Package Organization

### Core Components

- **`core/`**: Mathematical abstractions, problem definitions, physics losses, training infrastructure, and spectral methods
- **`geometry/`**: Domain handling, constructive solid geometry, Lie groups, and Riemannian/hyperbolic manifolds
- **`physics/`**: Physics solvers (Burgers, Darcy, Navier-Stokes, diffusion-advection, shallow water)
- **`data/`**: Data loaders, sources, and preprocessing (Darcy, Burgers, Navier-Stokes, PDEBench)

### Neural Components

- **`neural/`**: Neural operators (FNO, DeepONet, GNO, SFNO, U-FNO, UNO, TFNO, PINO, and specialized variants), PINNs (standard and domain-decomposition), Bayesian layers, and quantum chemistry modules
- **`training/`**: Training infrastructure with basic trainer, modular trainer, and checkpoint management
- **`fields/`**: JAX-native field abstractions for structured grids -- differential operators, advection, and pressure projection

### Optimization & Discovery

- **`optimization/`**: Learn-to-optimize engine, meta-optimization (MAML/Reptile), second-order methods, control systems, and production optimization
- **`discovery/`**: Equation discovery framework -- SINDy, ensemble SINDy, weak SINDy, and UDE distillation
- **`solvers/`**: Unified solver interfaces -- PINNSolver, EnsembleWrapper, ControlPINNSolver, and Artifex adapter

### Infrastructure

- **`benchmarking/`**: Benchmark registry, evaluation engine, PDEBench integration, profiling, and report generation
- **`deployment/`**: Cloud deployment (AWS, GCP), Kubernetes orchestration, monitoring, and resource management
- **`visualization/`**: Field plotting, animation, and performance visualization
- **`diagnostics/`**: NTK computation and model diagnostics
- **`distributed/`**: Distributed training manager
- **`platform/`**: Model registry, search, and validation
- **`mlops/`**: Experiment tracking with MLflow backend
- **`scalability/`**: Scalability search and analysis
- **`education/`**: Educational tools and tutorials

## Architecture Layers

1. **Foundation**: JAX / Flax NNX core primitives
2. **Mathematical** (`core/`, `geometry/`): Abstractions, domains, and molecular systems
3. **Algorithms** (`neural/`, `training/`): Operators, PINNs, and training loops
4. **Optimization** (`optimization/`, `discovery/`): Meta-learning, L2O, and equation discovery
5. **Infrastructure** (`benchmarking/`, `deployment/`, `visualization/`): Evaluation and production tools

## Getting Started

```python
import opifex

# Neural operators
from opifex.neural.operators.fno import FourierNeuralOperator
from opifex.neural.operators.deeponet import DeepONet

# Problem definitions
from opifex.core.problems import create_pde_problem
from opifex.geometry import Rectangle, Circle

# Training
from opifex.core.training.trainer import Trainer
from opifex.core.training.config import TrainingConfig

# Discovery
from opifex.discovery.sindy import SINDy, SINDyConfig

# Field operations
from opifex.fields import CenteredGrid, gradient, laplacian
```

## Development Guidelines

- **JAX-native**: All implementations use JAX / Flax NNX
- **Type annotations**: Full type coverage with `jax.Array` and `jaxtyping`
- **Modular design**: Clear separation of concerns between packages
- **Physics compliance**: Built-in validation of conservation laws

## Package Documentation

See individual package README files:

- [Core](core/README.md) -- mathematical abstractions and problem definitions
- [Geometry](geometry/README.md) -- geometric framework with CSG, Lie groups, manifolds
- [Neural](neural/README.md) -- neural network implementations
- [Training](training/README.md) -- training infrastructure
- [Optimization](optimization/README.md) -- L2O engine and advanced solvers
- [Benchmarking](benchmarking/README.md) -- evaluation framework

For user-facing documentation, see the [docs/](../../docs/) directory.

# Opifex

<div align="center">

## A unified scientific machine learning framework built on JAX/Flax NNX

*From Latin "opifex" — worker, skilled maker*

[Documentation](docs/) • [Getting Started](docs/getting-started/installation.md) • [Examples](examples/) • [Contributing](CONTRIBUTING.md)

</div>

---

<!-- CI/CD Status Badges -->
[![CI](https://github.com/mahdi-shafiei/opifex/actions/workflows/ci.yml/badge.svg)](https://github.com/mahdi-shafiei/opifex/actions/workflows/ci.yml)
[![Documentation](https://github.com/mahdi-shafiei/opifex/actions/workflows/docs.yml/badge.svg)](https://github.com/mahdi-shafiei/opifex/actions/workflows/docs.yml)
[![Security](https://github.com/mahdi-shafiei/opifex/actions/workflows/security.yml/badge.svg)](https://github.com/mahdi-shafiei/opifex/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/mahdi-shafiei/opifex/branch/main/graph/badge.svg)](https://codecov.io/gh/mahdi-shafiei/opifex)

<!-- Package & Distribution Badges -->
[![PyPI version](https://img.shields.io/pypi/v/opifex?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/opifex/)
[![Python versions](https://img.shields.io/pypi/pyversions/opifex?logo=python&logoColor=white)](https://pypi.org/project/opifex/)
[![Downloads](https://img.shields.io/pypi/dm/opifex?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/opifex/)

<!-- Social & Code Quality Badges -->
[![GitHub stars](https://img.shields.io/github/stars/mahdi-shafiei/opifex?style=social)](https://github.com/mahdi-shafiei/opifex/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/mahdi-shafiei/opifex?style=social)](https://github.com/mahdi-shafiei/opifex/network/members)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<!-- Project Info Badges -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-native-orange.svg)](https://jax.readthedocs.io/)
[![FLAX NNX](https://img.shields.io/badge/FLAX-NNX-green.svg)](https://flax.readthedocs.io/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

---

> **⚠️ Early Development - API Unstable**
>
> Opifex is currently in early development and undergoing rapid iteration. Please be aware of the following implications:
>
> | Area | Status | Impact |
> |------|--------|--------|
> | **API** | 🔄 Unstable | Breaking changes are expected. Public interfaces may change without deprecation warnings. Pin to specific commits if stability is required. |
> | **Tests** | 🔄 In Flux | Test suite is being expanded. Some tests may fail or be skipped. Coverage metrics are improving but not yet comprehensive. |
> | **Documentation** | 🔄 Evolving | Docs may not reflect current implementation. Code examples might be outdated. Refer to source code and tests for accurate usage. |
>
> We recommend waiting for a stable release (v1.0) before using Opifex in production. For research and experimentation, proceed with the understanding that APIs will evolve.

---

A **JAX-native platform** for scientific machine learning, built for unified excellence, probabilistic-first design, and high performance.

## 🎯 Core Vision

- **🔬 Unified Excellence**: Single platform supporting all major Opifex paradigms with mathematical clarity
- **📊 Probabilistic-First**: Built-in uncertainty quantification treating all computation as Bayesian inference
- **⚡ High Performance**: Optimized for speed with JAX transformations and GPU acceleration
- **🏗️ Production-Oriented**: Designed with benchmarking and deployment tools for future production use
- **🤝 Community-Driven**: Open patterns for education, research collaboration, and industrial adoption

## ✨ Key Features

- **Unified SciML Solvers**: Standardized protocol for PINNs, Neural Operators, and Hybrid solvers
- **Advanced Uncertainty Quantification**: Ensemble methods, Conformal Prediction, and Generative UQ
- **Artifex Integration**: Seamless bridge to top-tier generative models (Diffusion, Flows)
- **Neural Operators**: Fourier Neural Operators (FNO), DeepONet, and specialized variants
- **Physics-Informed Neural Networks**: Standard PINNs and advanced variants with multi-physics composition
- **Neural Density Functional Theory**: Quantum chemistry with chemical accuracy
- **Learn-to-Optimize**: Meta-optimization and adaptive algorithms
- **MLOps Integration**: Deployment infrastructure and MLOps tooling (in development)

For detailed feature documentation, see [Features](docs/features.md).

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (optional but recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/mahdi-shafiei/opifex.git
cd opifex

# Set up development environment
./setup.sh

# Activate environment
source ./activate.sh

# Run tests to verify installation
uv run pytest tests/ -v
```

For detailed installation instructions, see [Installation Guide](docs/getting-started/installation.md).

## 📚 Basic Usage

### Fourier Neural Operator (FNO)

```python
import jax
from flax import nnx
from opifex.neural.operators.fno import FourierNeuralOperator

# Create FNO for learning PDE solution operators
rngs = nnx.Rngs(jax.random.PRNGKey(0))

fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    modes=12,
    num_layers=4,
    rngs=rngs,
)

# Input: (batch, channels, *spatial_dims)
x = jax.random.normal(jax.random.PRNGKey(1), (4, 1, 64, 64))
y = fno(x)
print(f"FNO: {x.shape} -> {y.shape}")  # (4, 1, 64, 64) -> (4, 1, 64, 64)
```

### Deep Operator Network (DeepONet)

```python
import jax
from flax import nnx
from opifex.neural.operators.deeponet import DeepONet

# Create DeepONet for function-to-function mapping
rngs = nnx.Rngs(jax.random.PRNGKey(0))

deeponet = DeepONet(
    branch_sizes=[100, 64, 64, 32],  # 100 sensor locations
    trunk_sizes=[2, 64, 64, 32],     # 2D output coordinates
    activation="gelu",
    rngs=rngs,
)

# Branch input: function values at sensors (batch, num_sensors)
# Trunk input: evaluation coordinates (batch, n_locations, coord_dim)
branch_input = jax.random.normal(jax.random.PRNGKey(1), (8, 100))
trunk_input = jax.random.uniform(jax.random.PRNGKey(2), (8, 50, 2))  # 50 eval points

output = deeponet(branch_input, trunk_input)
print(f"DeepONet output: {output.shape}")  # (8, 50)
```

### Unified Solver & Uncertainty Quantification

```python
from opifex.solvers import PINNSolver, EnsembleWrapper
from opifex.solvers.adapters import ArtifexSolverAdapter

# 1. Standard PINN Solver
# solver = PINNSolver(model=pinn_model, optimizer=opt)

# 2. Ensemble UQ Wrapper (wraps multiple solver instances)
# ensemble = EnsembleWrapper(solvers=[solver1, solver2, solver3])
# solution = ensemble.solve(problem)
# print(f"Mean: {solution.fields['u_mean'].shape}, Std: {solution.fields['u_std'].shape}")

# 3. Artifex Generative Integration (Diffusion/Flows)
# adapter = ArtifexSolverAdapter(artifex_model)
# gen_solution = adapter.solve(problem)
```

For comprehensive examples and tutorials, see the [Examples](examples/) directory and [Documentation](docs/).

## 🔧 Development

```bash
# Run tests
uv run pytest tests/ -v

# Code quality checks
uv run pre-commit run --all-files
```

For detailed development guidelines, see [Development Guide](docs/development/).

## 📖 Documentation

- **[Getting Started](docs/getting-started/)**: Installation and basic usage
- **[Features](docs/features.md)**: Complete feature overview
- **[Architecture](docs/architecture.md)**: Framework design and structure
- **[API Reference](docs/api/)**: Complete API documentation
- **[Examples](examples/)**: Working examples and tutorials
- **[Development](docs/development/)**: Contributing and development setup

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Ready to get started?** Check out our [Quick Start Guide](docs/getting-started/quickstart.md) or explore the [Examples](examples/) directory!

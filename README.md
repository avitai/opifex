# Opifex

<div align="center">

## A unified scientific machine learning framework built on JAX/Flax NNX

*From Latin "opifex" — worker, skilled maker*

[Documentation](docs/) • [Getting Started](docs/getting-started/installation.md) • [Examples](examples/) • [Contributing](docs/development/contributing.md)

</div>

---

<!-- CI/CD Status Badges -->
[![CI](https://github.com/avitai/opifex/actions/workflows/ci.yml/badge.svg)](https://github.com/avitai/opifex/actions/workflows/ci.yml)
[![Documentation](https://github.com/avitai/opifex/actions/workflows/docs.yml/badge.svg)](https://github.com/avitai/opifex/actions/workflows/docs.yml)
[![Security](https://github.com/avitai/opifex/actions/workflows/security.yml/badge.svg)](https://github.com/avitai/opifex/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/avitai/opifex/branch/main/graph/badge.svg)](https://codecov.io/gh/avitai/opifex)

<!-- Package & Distribution Badges -->
[![PyPI version](https://img.shields.io/pypi/v/opifex?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/opifex/)
[![Python versions](https://img.shields.io/pypi/pyversions/opifex?logo=python&logoColor=white)](https://pypi.org/project/opifex/)
[![Downloads](https://img.shields.io/pypi/dm/opifex?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/opifex/)

<!-- Social & Code Quality Badges -->
[![GitHub stars](https://img.shields.io/github/stars/avitai/opifex?style=social)](https://github.com/avitai/opifex/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/avitai/opifex?style=social)](https://github.com/avitai/opifex/network/members)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
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
> | **Tests** | 🔄 In Flux | Test suite is being expanded. Some tests may fail or be skipped. Coverage metrics are improving but not yet full. |
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

- **Neural Operators**: FNO, DeepONet, SFNO, U-FNO, UNO, TFNO, GNO, PINO, Local FNO, and more (26 architectures)
- **Physics-Informed Neural Networks**: Standard PINNs plus domain decomposition (FBPINN, XPINN, CPINN)
- **Equation Discovery**: SINDy, Ensemble SINDy, and Weak SINDy for recovering governing equations from data
- **Field Operations**: JAX-native differential operators, advection, and pressure projection on structured grids
- **Uncertainty Quantification**: Bayesian FNO, UQNO, conformal prediction, and ensemble methods
- **Advanced Training**: NTK analysis, GradNorm loss balancing, adaptive sampling (RAR-D)
- **Optimization**: Learn-to-optimize, meta-optimization (MAML/Reptile), and second-order methods
- **Unified SciML Solvers**: Standardized protocol for PINNs, Neural Operators, and Hybrid solvers
- **Quantum Chemistry**: Neural DFT and neural exchange-correlation functionals
- **51 Working Examples**: Full coverage from getting started to advanced research workflows

For detailed feature documentation, see [Features](docs/features.md).

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (optional but recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/avitai/opifex.git
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

### Equation Discovery (SINDy)

```python
import jax.numpy as jnp
from opifex.discovery.sindy import SINDy, SINDyConfig

# Generate Lorenz trajectory (σ=10, ρ=28, β=8/3) with RK4
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

# Discover governing equations from data
model = SINDy(SINDyConfig(polynomial_degree=2, threshold=0.3))
model.fit(jnp.stack(trajectory), jnp.stack(derivatives))

for eq in model.equations(["x", "y", "z"]):
    print(eq)
# dx/dt = -9.999 x + 10.000 y            (true: -10 x + 10 y)
# dy/dt = 28.000 x + -1.000 y + -1.000 x z  (true: 28 x - y - x z)
# dz/dt = -2.667 z + 1.000 x y            (true: -8/3 z + x y)
```

For full examples and tutorials, see the [Examples](examples/) directory and [Documentation](docs/).

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

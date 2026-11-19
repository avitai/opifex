# Opifex

<div align="center">

## A unified scientific machine learning framework built on JAX/Flax NNX

*From Latin "opifex" - worker, skilled maker*

[Documentation](https://opifex.readthedocs.io/en/latest/) • [Getting Started](docs/getting-started/installation.md) • [Examples](examples/) • [Contributing](docs/development/contributing.md)

</div>

---

<!-- CI/CD Status Badges -->
[![CI](https://github.com/avitai/opifex/actions/workflows/ci.yml/badge.svg)](https://github.com/avitai/opifex/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/opifex/badge/?version=latest)](https://opifex.readthedocs.io/en/latest/)
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
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-native-orange.svg)](https://jax.readthedocs.io/)
[![FLAX NNX](https://img.shields.io/badge/FLAX-NNX-green.svg)](https://flax.readthedocs.io/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

---

> **Research preview.** Opifex is under rapid iteration and the API will change while we iterate
> toward v1.0. What that means concretely:
>
> | Area | Status | Impact |
> |------|--------|--------|
> | **API** | 🔄 Unstable | Breaking changes are expected. Public interfaces may change without deprecation warnings. Pin to a specific release or commit if stability is required. |
> | **Tests** | 🔄 In Flux | Test suite is being expanded. Some tests may fail or be skipped. Coverage metrics are improving but not yet full. |
> | **Documentation** | 🔄 Evolving | Docs may not reflect current implementation. Code examples might be outdated. Refer to source code and tests for accurate usage. |
>
> Pin a version if you need stability, and do not put it in production yet. For research and
> experimentation it is ready to use today, with the understanding that APIs will evolve.
>
> This is public this early on purpose. Issues, questions and pull requests genuinely steer
> what gets built next, and a star tells us which layer to push on.

---

A **JAX-native platform** for scientific machine learning, built for unified excellence, probabilistic-first design, and high performance.

## 🎯 Core Vision

- **🔬 Unified Excellence**: Single JAX/Flax NNX platform spanning neural operators, physics-informed learning, differentiable solvers, atomistic & quantum modeling, and uncertainty quantification
- **📊 Probabilistic-First**: Built-in calibration, conformal prediction, Bayesian inference, and active learning across scientific workflows
- **⚡ High Performance**: End-to-end `jit`, `vmap`, and reverse/implicit `grad` compatibility with GPU acceleration
- **🏗️ Production-Oriented**: Reproducible benchmarking, roofline profiling, experiment tracking, and serving primitives
- **🤝 Community-Driven**: Open patterns for education, research collaboration, and industrial adoption

## ✨ Key Features

- **Neural Operators**: FNO, DeepONet, PINO, TFNO, UFNO, SFNO, LocalFNO, AM-FNO, MS-FNO, UNO, FourierDeepONet, AdaptiveDeepONet, MultiPhysicsDeepONet, GINO, MGNO, UQNO, LNO, WNO, GNO, OperatorNet (20 registered architectures; `opifex.neural.operators.OPERATOR_REGISTRY`)
  - **Spectral, Multi-Scale & Clifford Families**: Resolution-invariant Fourier (`FNO`, `TFNO`, `UFNO`, `UNO`, `MS-FNO`, `AM-FNO`, `LocalFNO`), spherical (`SFNO`), wavelet (`WNO`), latent attention (`LNO`), and multivector Clifford (`CliffordFNO`) operators
  - **Branch-Trunk, Geometry & Physics-Informed Families**: `DeepONet` variants (`FourierDeepONet`, `AdaptiveDeepONet`, `MultiPhysicsDeepONet`), irregular-mesh graph and geometry-informed operators (`GNO`, `GINO`, `MGNO`), physics-informed `PINO`, and uncertainty-aware `UQNO`
- **Physics-Informed Neural Networks & KANs**: Standard PINNs, variational/fractional/spacetime/GP variants (`VPINN`, `fPINN`, `GST-PINN`, `GP-PINN`, multi-scale PINNs), domain decomposition (`FBPINN`, `XPINN`, `CPINN`), and physics-informed Kolmogorov-Arnold networks (`PIKAN`)
- **Differentiable PDE, Stiff ODE & Nonlinear Solvers**: Typed `Problem` / `Method` / `Solution` protocol (`opifex.core.solver`) spanning staggered-grid Navier-Stokes, lid-driven cavity, Burgers, Darcy, shallow-water, and spectral `ETDRK4` PDE solvers, pure-JAX stiff Rosenbrock/ESDIRK ODE integrators (`Rodas4`, `Ros34PW2`, `Kvaerno3`), `LevenbergMarquardt` nonlinear least squares with implicit differentiation, `UDE`, and `additive_hybrid` classical-neural solvers
- **Atomistic Potentials**: E(3)-equivariant `SchNet`, `PaiNN`, and `NequIP` backbones (with MACE-style higher body-order symmetric contraction), energy/forces/stress heads, and an ASE calculator
- **Quantum Chemistry**: Differentiable Kohn-Sham DFT (`SCFSolver` with Anderson/DIIS mixing and implicit fixed-point differentiation), neural exchange-correlation functionals, variational Monte Carlo, and equivariant Fock/Hamiltonian prediction (`QH9`)
- **Equivariant Algebra, Geometry & Fields**: Native E(3) irreps, Clebsch-Gordan, Wigner-D, and spherical harmonics; spherical and Poincaré hyperbolic manifolds, constructive solid geometry (`CSG`), point-cloud radius graphs, and staggered-grid MAC differential operators with exact DFT/DCT-II pressure projection
- **Uncertainty Quantification & Probabilistic Numerics**: Split, grouped, time-series, risk-controlling, and right-censored IPCW/doubly-robust conformal prediction (`CensoredSplitConformalRegressor`); censored Gaussian/Tobit likelihoods and correlated contrasts; Gaussian processes, state-space & computation-aware Kalman filters (`cakf_filter`), Bayesian quadrature, data assimilation, simulation-based inference (NPE/NLE/NRE), PAC-Bayes, Sobol/Morris sensitivity, FORM/SORM reliability, BlackJAX MCMC, and UQ adapters (`opifex.capabilities.UQ_CAPABILITIES`)
- **Active Learning, Bayesian Optimization & Multi-Fidelity**: Traced joint mutual-information batch acquisition (`batch_bald`), constrained multi-objective `qEHVI`, Thompson sampling, REMBO, trust-region & asynchronous BO, experimental design, and linear/`NARGP`/`MUMBO` multi-fidelity surrogates
- **Equation Discovery**: Sparse Identification of Nonlinear Dynamics (`SINDy`, `EnsembleSINDy`, `WeakSINDy`, `BayesianSINDy`) and symbolic regression
- **Training, Meta-Optimization & Control**: NNX-native `Trainer` with `GradNorm` physics/boundary loss balancing, NTK spectral diagnostics, adaptive residual sampling (`RAR-D`), learn-to-optimize (`L2O`) & meta-learning (`MAML`/`Reptile`), second-order hybrid optimizers, and model predictive control (`MPC`)
- **Scientific Data, Benchmarking & MLOps**: `datarax` `Source`/`Pipeline` loaders (`PDEBench`, `VTK` unstructured meshes, `QH9`, spectral PDE generators), `calibrax`-backed operator benchmarking & roofline profiling (`opifex-benchmark` CLI), MLflow/W&B/Neptune experiment tracking, and 59 runnable example scripts and notebooks

For detailed feature documentation, see [Features](docs/features.md).

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (optional but recommended)

### Installation

opifex is on PyPI:

```bash
uv add opifex            # or: pip install opifex
uv add "opifex[cuda12]"  # Linux NVIDIA GPU wheels (CUDA 12)
uv add "opifex[mlflow]"  # with the MLflow experiment backend
```

To work on opifex itself, clone the repository and use the managed environment:

```bash
# Clone the repository
git clone https://github.com/avitai/opifex.git
cd opifex

# Set up development environment (auto-detects CUDA / Metal / CPU)
./setup.sh

# Activate environment
source ./activate.sh

# Run tests to verify installation
uv run pytest
```

For detailed installation instructions, see [Installation Guide](docs/getting-started/installation.md).

## 📚 Basic Usage

### Fourier Neural Operator (FNO)

```python
import jax
from flax import nnx
from opifex.neural.operators.fno import FourierNeuralOperator

# Create FNO for learning PDE solution operators
rngs = nnx.Rngs(jax.random.key(0))

fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    modes=12,
    num_layers=4,
    rngs=rngs,
)

# Input: (batch, channels, *spatial_dims)
x = jax.random.normal(jax.random.key(1), (4, 1, 64, 64))
y = fno(x)
print(f"FNO: {x.shape} -> {y.shape}")  # (4, 1, 64, 64) -> (4, 1, 64, 64)
```

### Deep Operator Network (DeepONet)

```python
import jax
from flax import nnx
from opifex.neural.operators.deeponet import DeepONet

# Create DeepONet for function-to-function mapping
rngs = nnx.Rngs(jax.random.key(0))

deeponet = DeepONet(
    branch_sizes=[100, 64, 64, 32],  # 100 sensor locations
    trunk_sizes=[2, 64, 64, 32],     # 2D output coordinates
    activation="gelu",
    rngs=rngs,
)

# Branch input: function values at sensors (batch, num_sensors)
# Trunk input: evaluation coordinates (batch, n_locations, coord_dim)
branch_input = jax.random.normal(jax.random.key(1), (8, 100))
trunk_input = jax.random.uniform(jax.random.key(2), (8, 50, 2))  # 50 eval points

output = deeponet(branch_input, trunk_input)
print(f"DeepONet output: {output.shape}")  # (8, 50)
```

### Equation Discovery (SINDy)

```python
import jax
import jax.numpy as jnp
from opifex.discovery.sindy import SINDy, SINDyConfig

# Generate Lorenz trajectory (σ=10, ρ=28, β=8/3) with scanned RK4
def lorenz(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

dt, state0 = 0.001, jnp.array([1.0, 1.0, 1.0])

def rk4_step(state, _):
    k1 = lorenz(state)
    k2 = lorenz(state + 0.5 * dt * k1)
    k3 = lorenz(state + 0.5 * dt * k2)
    k4 = lorenz(state + dt * k3)
    nxt = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return nxt, (nxt, lorenz(nxt))

_, (traj, derivs) = jax.lax.scan(rk4_step, state0, None, length=10000)
trajectory = jnp.concatenate([state0[None], traj], axis=0)
derivatives = jnp.concatenate([lorenz(state0)[None], derivs], axis=0)

# Discover governing equations from data
model = SINDy(SINDyConfig(polynomial_degree=2, threshold=0.3))
model.fit(trajectory, derivatives)

for eq in model.equations(["x", "y", "z"]):
    print(eq)
# dx/dt = -10.000 x + 10.000 y            (true: -10 x + 10 y)
# dy/dt = 27.999 x + -1.000 y + -1.000 x z  (true: 28 x - y - x z)
# dz/dt = -2.667 z + 1.000 x y            (true: -8/3 z + x y)
```

### Conformal Uncertainty Quantification

```python
import jax
import jax.numpy as jnp
from opifex.uncertainty.conformal import SplitConformalRegressor

x_cal = jnp.linspace(0.0, 1.0, 100)
y_cal = jnp.sin(2.0 * jnp.pi * x_cal) + 0.1 * jax.random.normal(jax.random.key(0), (100,))
pred_cal = jnp.sin(2.0 * jnp.pi * x_cal)

calibrator = SplitConformalRegressor(alpha=0.1)
state = calibrator.fit(predictions=pred_cal, targets=y_cal)

x_test = jnp.array([0.25, 0.5, 0.75])
interval = calibrator.with_state(state).predict(predictions=jnp.sin(2.0 * jnp.pi * x_test))
print(f"90% interval shape: {interval.lower.shape} -> {interval.upper.shape}")  # (3,) -> (3,)
```

For full examples and tutorials, see the [Examples](examples/) directory and [Documentation](docs/).

## 🔧 Development

```bash
# Run tests
uv run pytest

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

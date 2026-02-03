# Opifex: Unified Scientific Machine Learning Framework

A **JAX-native platform** for scientific machine learning, built for unified excellence, probabilistic-first design, and high performance.

## 🎯 Core Vision

- **🔬 Unified Excellence**: Single platform supporting all major Opifex paradigms with mathematical clarity
- **📊 Probabilistic-First**: Built-in uncertainty quantification treating all computation as Bayesian inference
- **⚡ High Performance**: Optimized for speed with JAX transformations and GPU acceleration
- **🏗️ Research Infrastructure**: Flexible design with integrated benchmarking and experimental tools
- **🤝 Community-Driven**: Open patterns for education, research collaboration, and industrial adoption

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (optional but recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/opifex-org/opifex.git
cd opifex

# Set up unified development environment (auto-detects GPU/CPU)
./setup.sh

# Activate environment
source ./activate.sh

# Run tests to verify installation
uv run pytest tests/ -v
```

**For detailed setup instructions, troubleshooting, and configuration options, see the [Environment Setup Guide](getting-started/environment-setup.md).**

### Basic Example

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
x = jax.random.normal(key, (4, 1, 64, 64))  # (batch, channels, height, width)
y = fno(x)
print(f"FNO: {x.shape} -> {y.shape}")  # (4, 1, 64, 64) -> (4, 1, 64, 64)
```

## 📚 Documentation

### Getting Started

- **[Installation Guide](getting-started/installation.md)** - Setup instructions and configuration
- **[Quick Start Tutorial](getting-started/quickstart.md)** - Get up and running quickly
- **[Environment Setup](getting-started/environment-setup.md)** - Development environment configuration
- **[GPU Setup](getting-started/gpu-setup.md)** - GPU acceleration setup

### Core Documentation

- **[Features](features.md)** - Overview of Opifex paradigms and capabilities
- **[Architecture](architecture.md)** - Framework design and 6-layer architecture
- **[Technology Stack](tech-stack.md)** - Dependencies, tools, and infrastructure

### User Guides

- **[Concepts](user-guide/concepts.md)** - Core concepts and terminology
- **[Neural Networks](user-guide/neural-networks.md)** - Neural network implementations
- **[Geometry](user-guide/geometry.md)** - Geometric operations and representations
- **[Training](user-guide/training.md)** - Training infrastructure and workflows
- **[Problems](user-guide/problems.md)** - Problem definition and setup

### Methods & Tutorials

- **[Neural Operators](methods/neural-operators.md)** - FNO, DeepONet, and advanced architectures
- **[Physics-Informed Networks](methods/pinns.md)** - PINNs and physics-constrained learning
- **[Neural DFT](methods/neural-dft.md)** - Quantum chemistry with neural networks
- **[Learn-to-Optimize](methods/l2o.md)** - Meta-optimization and adaptive algorithms
- **[Probabilistic Methods](methods/probabilistic.md)** - Uncertainty quantification
- **[Advanced Benchmarking](methods/advanced-benchmarking.md)** - Evaluation and validation

### API Reference

- **[Core Package](api/core.md)** - Mathematical abstractions and numerical framework
- **[Neural Package](api/neural.md)** - Neural operators and physics-informed networks
- **[Geometry Package](api/geometry.md)** - Geometric operations and manifolds
- **[Training Package](api/training.md)** - Training infrastructure and optimization
- **[Optimization Package](api/optimization.md)** - Meta-optimization and advanced algorithms
- **[Benchmarking Package](api/benchmarking.md)** - Evaluation and validation tools
- **[Bayesian Package](api/bayesian.md)** - Uncertainty quantification and Bayesian methods

### Examples

- **[Examples Overview](examples/index.md)** - Runnable examples and demonstrations
- **[Neural Operators](examples/neural-operators/fno-darcy.md)** - FNO, TFNO, DeepONet, and more
- **[PINNs](examples/pinns/heat-equation.md)** - Physics-informed neural networks
- **[Quantum Chemistry](examples/quantum-chemistry/neural-dft.md)** - Neural DFT and molecular examples

### Development

- **[Contributing](development/contributing.md)** - How to contribute to Opifex
- **[Development Setup](development/architecture.md)** - Development environment and guidelines
- **[Code Quality](development/code-quality.md)** - Standards and best practices
- **[Testing](development/testing.md)** - Testing framework and guidelines
- **[GPU Development](development/gpu-development.md)** - GPU development guidelines

### Deployment

- **[Local Development](deployment/local-development.md)** - Local deployment setup
- **[AWS Deployment](deployment/aws-deployment.md)** - Amazon Web Services deployment
- **[GCP Deployment](deployment/gcp-deployment.md)** - Google Cloud Platform deployment
- **[Troubleshooting](deployment/troubleshooting.md)** - Common deployment issues

## 🤝 Community

### Getting Help

- **[FAQ](faq.md)** - Frequently asked questions
- **GitHub Issues** - Report bugs and request features
- **Discussions** - Community Q&A and collaboration
- **Research Partnerships** - Academic collaboration opportunities

### Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details on:

- Code style and standards
- Testing requirements
- Documentation guidelines
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/opifex-org/opifex/blob/main/LICENSE) file for details.

---

**Ready to get started?** Check out our [Quick Start Guide](getting-started/quickstart.md) or explore the [Features](features.md) to learn about Opifex's capabilities!

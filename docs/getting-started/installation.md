# Installation Guide

## Prerequisites

- **Python 3.11+** (Required for JAX ecosystem compatibility)
- **CUDA-compatible GPU** (Optional but recommended for performance)
- **Git** for repository management
- **uv** package manager (will be installed automatically)

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/opifex-org/opifex.git
cd opifex
```

### 2. Set up Development Environment

The repository includes an automated setup script that handles all dependencies:

```bash
./setup.sh
source ./activate.sh
```

This script automatically:

- Detects GPU/CPU configuration
- Installs all dependencies
- Configures the environment
- Verifies the installation

**📖 For detailed setup options, troubleshooting, and advanced configuration, see the [Environment Setup Guide](environment-setup.md).**

## Technology Stack Verification

After installation, verify that all core dependencies are working:

```python
import jax
import flax.nnx as nnx
import optax
import diffrax
import blackjax
import distrax
import optimistix
import lineax
import orbax.checkpoint
import opifex

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("FLAX NNX available:", hasattr(nnx, 'Module'))
print("Opifex framework ready")
```

Expected output:

```text
JAX version: 0.8.0
JAX devices: [cuda(id=0)] (or [cpu(id=0)] without GPU)
FLAX NNX available: True
Opifex framework ready
```

## Validated Dependencies

The following JAX ecosystem dependencies are validated and operational:

### Core JAX Ecosystem

- **JAX 0.8.0**: Core framework with CUDA support
- **FLAX 0.12.0**: NNX neural network framework (exclusive)
- **Optax 0.2.6+**: Optimization algorithms
- **Optimistix 0.0.10+**: Root finding & minimization
- **Lineax 0.0.8+**: Linear solvers
- **BlackJAX 1.2.5+**: MCMC sampling
- **Distrax 0.1.0**: Probabilistic programming
- **Diffrax 0.4.0+**: Differential equations
- **Orbax 0.11.13+**: Checkpointing system

### Development Tools

- **uv**: Package management (exclusive)
- **ruff + pyright**: Code quality (exclusive)
- **pytest**: Testing framework
- **MkDocs**: Documentation system (exclusive)
- **pre-commit**: Code quality hooks

## GPU Support

### CUDA Installation

For GPU acceleration, ensure CUDA is properly installed:

```bash
# Check CUDA availability
nvidia-smi

# Verify JAX can see GPU
python -c "import jax; print('GPU available:', len(jax.devices('gpu')) > 0)"
```

### CPU-Only Installation

JAX works perfectly on CPU-only systems. The framework automatically detects available hardware and optimizes accordingly.

## Development Environment

### Code Quality Tools

The framework uses exclusive tools for code quality:

```bash
# Run code formatting
uv run ruff format .

# Run linting
uv run ruff check .

# Run type checking
uv run pyright

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

### Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=opifex --cov-report=html

# Comprehensive test reporting with JSON output and detailed coverage
uv run pytest -vv --json-report --json-report-file=temp/test-results.json --json-report-indent=2 --json-report-verbosity=2 --cov=opifex --cov-report=json:temp/coverage.json --cov-report=term-missing
```

### Documentation

```bash
# Serve documentation locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure virtual environment is activated
2. **CUDA issues**: Check NVIDIA drivers and CUDA installation
3. **Memory errors**: Reduce batch sizes or use CPU backend
4. **Package conflicts**: Use `uv sync --force-reinstall`

### Getting Help

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community Q&A and collaboration
- **Documentation**: Comprehensive guides and tutorials

## Next Steps

After successful installation:

1. **Quick Start**: Follow the [Quick Start Guide](quickstart.md)
2. **Development**: Read the [Development Guide](../development/contributing.md)
3. **Examples**: Explore [Working Examples](../examples/working-examples.md)
4. **API Reference**: Check the [Core API](../api/core.md)

## Verification Checklist

- [ ] Python 3.11+ installed
- [ ] Repository cloned successfully
- [ ] Virtual environment created and activated
- [ ] All dependencies installed via `uv sync`
- [ ] Pre-commit hooks installed and passing
- [ ] Tests passing with `uv run pytest tests/ -v`
- [ ] JAX can detect available hardware
- [ ] Opifex package imports successfully
- [ ] Documentation builds with `uv run mkdocs build`

Once all items are checked, you're ready to start using the Opifex framework!

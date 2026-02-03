# Environment Setup Guide

This guide explains the Opifex environment setup system, including the `setup.sh` script, activation process, and troubleshooting common issues.

## Overview

The Opifex framework uses a unified environment setup system designed to handle both CPU and GPU configurations automatically. The system consists of three main components:

1. **`setup.sh`** - Unified setup script that creates and configures the development environment
2. **`activate.sh`** - Environment activation script (created by setup.sh)
3. **`.env`** - Environment configuration file (created by setup.sh)

## Why We Use This System

### 🎯 **Unified Configuration**

- **Single command setup**: `./setup.sh` handles everything from virtual environment creation to CUDA configuration
- **Automatic detection**: Detects GPU availability and configures accordingly
- **Consistent environments**: Ensures all developers have identical setups

### 🚀 **JAX + CUDA Complexity**

- **CUDA library management**: JAX requires specific CUDA library paths and configurations
- **Version compatibility**: Ensures JAX, CUDA, and GPU drivers work together
- **Memory management**: Optimizes GPU memory allocation for scientific computing

### 🔧 **Development Efficiency**

- **One-time setup**: Run once, activate many times
- **Isolated dependencies**: Virtual environment prevents conflicts
- **Easy troubleshooting**: Comprehensive error detection and reporting

## Setup Process

### 1. Initial Setup

```bash
# Clone the repository
git clone https://github.com/opifex-org/opifex.git
cd opifex

# Run unified setup (auto-detects GPU/CPU)
./setup.sh
```

### 2. Environment Activation

**⚠️ Important: Always use `source`**

```bash
# ✅ CORRECT: Use source to activate
source ./activate.sh

# ❌ INCORRECT: Don't run directly
./activate.sh  # This won't work!
```

**Why `source` is required:**

- Environment variables must be set in the current shell
- Virtual environment activation modifies the current shell's PATH
- Running directly creates a subprocess that exits immediately

## Setup Script Options

### Basic Usage

```bash
./setup.sh [OPTIONS]
```

### Available Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `--help`, `-h` | Show help message | Get usage information |
| `--deep-clean` | Comprehensive cleaning | Clear all caches and start fresh |
| `--cpu-only` | Force CPU-only setup | Skip GPU detection/configuration |
| `--force` | Force reinstallation | Overwrite existing environment |
| `--verbose`, `-v` | Show detailed output | Debug setup issues |

### Example Commands

```bash
# Standard setup with auto GPU detection
./setup.sh

# Clean setup with cache clearing
./setup.sh --deep-clean

# Force CPU-only development setup
./setup.sh --cpu-only

# Verbose forced reinstallation
./setup.sh --force --verbose

# Get help and see all options
./setup.sh --help
```

## Files Created by Setup

### Directory Structure

```text
opifex/
├── .venv/                 # Virtual environment
├── .env                   # Environment configuration
├── activate.sh            # Activation script
├── uv.lock               # Dependency lock file
├── setup.sh              # Setup script (existing)
└── dot_env_template      # Environment template (existing)
```

### File Descriptions

#### `.venv/` - Virtual Environment

- **Purpose**: Isolated Python environment with all dependencies
- **Contents**: Python interpreter, packages, CUDA libraries
- **Size**: ~2-4GB depending on GPU/CPU configuration

#### `.env` - Environment Configuration

- **Purpose**: Sets environment variables for JAX, CUDA, and Opifex
- **GPU Version**: Configures CUDA paths, JAX GPU settings
- **CPU Version**: Configures CPU-only JAX settings

#### `activate.sh` - Activation Script

- **Purpose**: Activates virtual environment and loads configuration
- **Features**: Process detection, GPU testing, status reporting
- **Usage**: `source ./activate.sh`

#### `uv.lock` - Dependency Lock File

- **Purpose**: Locks exact dependency versions for reproducibility
- **Generated**: Automatically created during first setup
- **Benefits**: Ensures consistent installations across environments

## Environment Configuration Details

### GPU Configuration (`.env` for CUDA)

```bash
# CUDA Library Paths
export LD_LIBRARY_PATH="..."  # Points to venv CUDA libraries

# JAX CUDA Settings
export JAX_PLATFORMS="cuda,cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.8"

# Performance Optimization
export JAX_ENABLE_X64="0"
export CUDA_MODULE_LOADING="LAZY"

# Development Settings
export PYTHONPATH="$(pwd)"
export PYTEST_CUDA_ENABLED="true"
```

### CPU Configuration (`.env` for CPU-only)

```bash
# JAX CPU Settings
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64="0"

# Development Settings
export PYTHONPATH="$(pwd)"
export PYTEST_CUDA_ENABLED="false"
```

### Opifex-Specific Variables

```bash
# Directory Configuration
export OPIFEX_BENCHMARK_CACHE_DIR="./benchmark_results"
export OPIFEX_DATA_DIR="./data"
export OPIFEX_CHECKPOINT_DIR="./checkpoints"

# Development Mode
export OPIFEX_DEV_MODE="1"
export OPIFEX_LOG_LEVEL="INFO"
```

## Activation Process Details

### What Happens During Activation

1. **Environment Check**: Detects if another virtual environment is active
2. **Process Detection**: Checks for processes using the current environment
3. **Deactivation**: Safely deactivates existing environments
4. **Virtual Environment**: Activates the `.venv` environment
5. **Configuration Loading**: Sources the `.env` file
6. **System Verification**: Tests JAX and GPU functionality
7. **Status Display**: Shows environment status and available commands

### Activation Output Example

```bash
$ source ./activate.sh

🚀 Activating Opifex Development Environment
=============================================
✅ Virtual environment activated
✅ Environment configuration loaded
   🎮 GPU Mode: CUDA enabled
   📍 CUDA_HOME: /path/to/opifex/.venv/lib/python3.11/site-packages/nvidia

🔍 Environment Status:
   Python: Python 3.11.5
   Working Directory: /path/to/opifex
   Virtual Environment: /path/to/opifex/.venv

🧪 JAX Configuration:
   JAX version: 0.6.2
   Default backend: gpu
   Available devices: 2 total
   🎉 GPU devices: 1 ([cuda:0])
   ✅ CUDA acceleration ready!
   🧮 GPU test successful: 14.0
   ✅ JAX functionality verified

🚀 Ready for Development!
=========================

📝 Common Commands:
   uv run pytest tests/ -v              # Run all tests
   uv run python your_script.py         # Run your code
   uv run jupyter lab                   # Start Jupyter lab

💡 To deactivate: deactivate
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Setup Fails to Detect GPU

**Symptoms:**

```bash
ℹ️  No NVIDIA GPU detected - setting up CPU-only environment
```

**Solutions:**

```bash
# Check GPU availability
nvidia-smi

# Check NVIDIA drivers
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Force GPU setup if drivers are installed
./setup.sh --force --verbose

# Check CUDA installation
ls /usr/local/cuda/
echo $CUDA_HOME
```

#### 2. JAX CUDA Import Errors

**Symptoms:**

```bash
❌ JAX not installed properly: No module named 'jax'
```

**Solutions:**

```bash
# Reinstall with verbose output
./setup.sh --force --verbose

# Check virtual environment
source ./activate.sh
python -c "import sys; print(sys.executable)"

# Manual dependency check
uv sync --extra all
```

#### 3. CUDA Library Path Issues

**Symptoms:**

```bash
⚠️  GPU test warning: CUDA library not found
```

**Solutions:**

```bash
# Check CUDA library paths
source ./activate.sh
echo $LD_LIBRARY_PATH

# Verify CUDA libraries exist
ls .venv/lib/python*/site-packages/nvidia/*/lib/

# Reinstall with deep clean
./setup.sh --deep-clean --force
```

#### 4. Virtual Environment Already Exists

**Symptoms:**

```bash
⚠️  Virtual environment already exists
Use --force to reinstall or source ./activate.sh to use existing environment
```

**Solutions:**

```bash
# Use existing environment
source ./activate.sh

# Or force reinstall
./setup.sh --force

# Or clean reinstall
./setup.sh --deep-clean
```

#### 5. Activation Hangs or Fails

**Symptoms:**

- Activation script appears to hang
- Process detection shows running processes

**Solutions:**

```bash
# Check for blocking processes
ps aux | grep python
ps aux | grep jupyter

# Kill blocking processes
pkill -f pytest
pkill -f jupyter

# Force activation
source ./activate.sh

# Manual environment setup
source .venv/bin/activate
source .env
```

#### 6. Permission Errors

**Symptoms:**

```bash
❌ Permission denied: ./setup.sh
```

**Solutions:**

```bash
# Make setup script executable
chmod +x setup.sh

# Check file permissions
ls -la setup.sh

# Run with explicit bash
bash setup.sh
```

#### 7. uv Package Manager Issues

**Symptoms:**

```bash
❌ Failed to install uv
```

**Solutions:**

```bash
# Manual uv installation
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Verify installation
uv --version

# Alternative installation methods
pip install uv
conda install uv
```

### Advanced Troubleshooting

#### Debug Mode Setup

```bash
# Run setup with maximum verbosity
./setup.sh --verbose --force

# Check environment variables
source ./activate.sh
env | grep -E "(JAX|CUDA|LD_LIBRARY)"

# Test JAX manually
python -c "
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('Default backend:', jax.default_backend())
"
```

#### Clean Slate Reinstallation

```bash
# Complete environment reset
rm -rf .venv .env activate.sh uv.lock
./setup.sh --deep-clean --verbose

# Verify clean installation
source ./activate.sh
uv run pytest tests/unit/test_core/ -v
```

#### Manual Environment Verification

```bash
# Check Python environment
which python
python --version

# Check package installations
python -c "import jax, flax, optax; print('All packages imported successfully')"

# Check CUDA functionality
python -c "
import jax.numpy as jnp
x = jnp.array([1., 2., 3.])
print('CUDA test:', jnp.sum(x**2))
"
```

## Best Practices

### 1. Environment Management

```bash
# Always use source for activation
source ./activate.sh

# Deactivate when switching projects
deactivate

# Regular environment updates
./setup.sh --force  # When dependencies change
```

### 2. Development Workflow

```bash
# Start development session
cd /path/to/opifex
source ./activate.sh

# Run tests to verify environment
uv run pytest tests/ -v

# Development work
uv run python your_script.py

# End session
deactivate
```

### 3. Troubleshooting Workflow

```bash
# 1. Try standard activation
source ./activate.sh

# 2. If issues, check status
./setup.sh --help
nvidia-smi

# 3. Force reinstall if needed
./setup.sh --force --verbose

# 4. Deep clean for persistent issues
./setup.sh --deep-clean --force
```



## Integration with Development Tools

### IDE Configuration

**VS Code:**

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.terminal.activateEnvironment": false
}
```

**PyCharm:**

- Set interpreter to `.venv/bin/python`
- Configure environment variables from `.env`

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Setup Opifex Environment
  run: |
    ./setup.sh --cpu-only
    source ./activate.sh
    uv run pytest tests/
```

## Performance Optimization

### GPU Memory Management

The setup automatically configures optimal GPU memory settings:

```bash
# Memory fraction (80% of GPU memory)
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.8"

# Disable preallocation for flexibility
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

# Lazy CUDA module loading
export CUDA_MODULE_LOADING="LAZY"
```

### Compilation Caching

```bash
# JAX compilation cache
export JAX_COMPILATION_CACHE_DIR="./cache/jax"

# XLA cache directory
export XLA_CACHE_DIR="./cache/xla"
```

## Security Considerations

### Environment Isolation

- Virtual environment prevents system-wide package conflicts
- Local CUDA libraries avoid system CUDA conflicts
- Project-specific environment variables

### Dependency Management

- Locked dependency versions in `uv.lock`
- Reproducible environments across systems
- Secure package installation through `uv`

## Getting Help

### Documentation Resources

- [Development Guide](../development/contributing.md) - Development workflow
- [Installation Guide](installation.md) - Basic installation
- [GPU Setup Guide](gpu-setup.md) - GPU-specific configuration

### Community Support

- [GitHub Issues](https://github.com/opifex-org/opifex/issues) - Bug reports
- [Discussions](https://github.com/opifex-org/opifex/discussions) - Q&A
- [Contributing Guide](../development/contributing.md) - Contribution help

### Quick Reference

```bash
# Setup commands
./setup.sh                    # Standard setup
./setup.sh --help            # Show options
./setup.sh --deep-clean      # Clean setup

# Activation
source ./activate.sh         # Activate environment

# Troubleshooting
./setup.sh --force --verbose # Debug setup
nvidia-smi                   # Check GPU
uv run pytest tests/ -v     # Verify installation
```

The Opifex environment setup system is designed to provide a robust, reproducible development environment that handles the complexities of JAX, CUDA, and scientific computing dependencies automatically.

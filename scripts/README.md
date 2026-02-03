# Opifex Scripts Directory

This directory contains utility scripts for the Opifex framework testing, verification, and environment management.

## 🚀 Quick Start

**New users should use the unified setup at the project root:**

```bash
# At project root (not in scripts/)
./setup.sh                    # Standard setup with auto GPU detection
./setup.sh --help             # Show all options
source ./activate.sh          # Activate environment after setup
```

The unified `setup.sh` at the project root provides:
- Automatic GPU/CPU detection
- Unified activation script (`activate.sh`)
- Comprehensive environment verification
- Cross-platform compatibility

## 📁 Available Scripts

### Environment & Setup

#### `setup_env.py`
Python-based environment configuration utility.

```bash
uv run python scripts/setup_env.py
uv run python scripts/setup_env.py --cuda
```

#### `clean_cache.sh`
Clean various caches (JAX, Python, pytest, etc.)

```bash
./scripts/clean_cache.sh
./scripts/clean_cache.sh --deep
```

### Testing & Verification

#### `run_tests_reliably.sh`
Run tests with retry logic and better error handling.

```bash
./scripts/run_tests_reliably.sh tests/neural/ -v
./scripts/run_tests_reliably.sh tests/benchmarks/ --maxfail=1
```

#### `run_tests_with_cuda.sh`
GPU-specific test runner with CUDA configuration.

```bash
./scripts/run_tests_with_cuda.sh tests/ -v
./scripts/run_tests_with_cuda.sh tests/neural/ -m gpu
```

#### `verify_opifex_gpu.py`
Comprehensive GPU verification and diagnostics.

```bash
uv run python scripts/verify_opifex_gpu.py
uv run python scripts/verify_opifex_gpu.py --detailed
```

#### `verify_opifex_examples.py`
Verify example notebooks and scripts work correctly.

```bash
uv run python scripts/verify_opifex_examples.py
uv run python scripts/verify_opifex_examples.py --verbose
```

#### `verify_jit_compatibility.py`
Verify JIT compilation compatibility across the codebase.

```bash
uv run python scripts/verify_jit_compatibility.py
```

### Utilities

#### `check_backend.py`
Quick utility to check which backend (CPU/CUDA) JAX is using.

```bash
uv run python scripts/check_backend.py
```

#### `check_opifex_compatibility.py`
Check system compatibility and dependencies.

```bash
uv run python scripts/check_opifex_compatibility.py
uv run python scripts/check_opifex_compatibility.py --full
```

#### `gpu_test_manager.py`
Advanced GPU testing and management utilities.

```bash
uv run python scripts/gpu_test_manager.py --test-basic
uv run python scripts/gpu_test_manager.py --benchmark
uv run python scripts/gpu_test_manager.py --monitor
```

#### `gpu_utils.py`
GPU utility functions (used by other scripts).

```bash
uv run python scripts/gpu_utils.py  # Shows available functions
```

## 🏗️ Common Workflows

### Initial Setup

1. **Setup Environment** (at project root):
   ```bash
   ./setup.sh
   source ./activate.sh
   ```

2. **Verify Installation**:
   ```bash
   uv run pytest tests/ -v
   uv run python scripts/verify_opifex_gpu.py
   ```

### Daily Development

1. **Activate Environment**:
   ```bash
   source ./activate.sh  # At project root
   ```

2. **Run Tests**:
   ```bash
   ./scripts/run_tests_reliably.sh tests/
   ```

3. **Clean Environment** (if needed):
   ```bash
   ./scripts/clean_cache.sh
   ```

### GPU Development

1. **GPU Verification**:
   ```bash
   uv run python scripts/verify_opifex_gpu.py --detailed
   ```

2. **Check Backend**:
   ```bash
   uv run python scripts/check_backend.py
   ```

3. **GPU Monitoring During Development**:
   ```bash
   uv run python scripts/gpu_test_manager.py --monitor
   ```

4. **GPU Benchmarking**:
   ```bash
   uv run python scripts/gpu_test_manager.py --benchmark
   ```

## 🔍 Troubleshooting

### Environment Issues

- **Environment not found**: Run `./setup.sh` at project root
- **GPU not detected**: Use `./setup.sh --force` to reinstall
- **JAX/CUDA issues**: Try `./setup.sh --deep-clean --force`

### Testing Issues

- **Flaky tests**: Use `./scripts/run_tests_reliably.sh` instead of pytest directly
- **GPU test failures**: Run `scripts/verify_opifex_gpu.py` for diagnosis
- **Memory issues**: Use `scripts/clean_cache.sh --deep`

### Performance Issues

- **Slow tests**: Use `scripts/gpu_test_manager.py --monitor` to check GPU usage
- **Compilation issues**: Clear JAX cache with `scripts/clean_cache.sh`

## 📚 Script Dependencies

All scripts assume:
- Project root activation environment (`source ./activate.sh`)
- `uv` package manager available
- Python 3.10+ environment

For GPU scripts additionally:
- NVIDIA drivers installed
- CUDA toolkit available
- JAX with CUDA support

## 💡 Best Practices

1. **Always use the unified setup**: `./setup.sh` at project root
2. **Source activation script**: `source ./activate.sh` (not `./activate.sh`)
3. **Use script utilities**: Leverage `scripts/` for specific tasks
4. **Clean when needed**: Use `scripts/clean_cache.sh` for issues
5. **Verify GPU setup**: Run `scripts/verify_opifex_gpu.py` after changes

For more information, see the main project documentation and the unified setup help: `./setup.sh --help`.

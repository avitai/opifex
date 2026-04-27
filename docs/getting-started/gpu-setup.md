# GPU Testing Solution - No CPU Fallback

## Overview

This document explains the full solution implemented to ensure that Opifex tests **fail appropriately** when GPU is not available, rather than silently falling back to CPU. This approach ensures that GPU-dependent functionality is properly tested and that users understand when their environment doesn't meet the requirements.

## Problem Statement

### Original Issue

- Tests were passing in local terminal but failing in chat environment
- Root cause: GPU memory issues and CUDA/cuBLAS errors causing segmentation faults
- Previous solution: Silent CPU fallback, which masked GPU requirements

### Why CPU Fallback is Problematic

1. **Masked Requirements**: Tests appeared to pass when GPU was actually required
2. **False Confidence**: Users thought their code worked when it actually failed on GPU
3. **Inconsistent Behavior**: Different environments produced different results
4. **Hidden Bugs**: GPU-specific issues were not caught during testing

## Solution Architecture

### 1. GPU Requirement Enforcement

#### GPU Testing Infrastructure (`opifex/core/testing_infrastructure.py`)

The framework includes full GPU testing infrastructure that handles device detection, stability testing, and environment classification:

```python
class EnvironmentType(Enum):
    """GPU environment classification."""
    GPU_SAFE = "gpu_safe"              # GPU available and stable
    GPU_AVAILABLE_UNSAFE = "gpu_unsafe" # GPU present but causes segfaults
    CPU_ONLY = "cpu_only"              # No GPU or GPU libraries unavailable
    UNKNOWN = "unknown"                # Environment not yet assessed

def _test_gpu_basic_stability(self) -> bool:
    """Test basic GPU stability without triggering JAX compilation."""
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == "gpu"]

        if not gpu_devices:
            return False

        # Very basic operation without JIT compilation
        with jax.default_device(gpu_devices[0]):
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.array([4.0, 5.0, 6.0])
            _ = x + y  # Simple operation, no compilation

        return True
    except Exception as e:
        self.logger.warning(f"GPU stability test failed: {e}")
        return False
```

### 2. Test Configuration Management

#### Pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"') - automatically detected via conftest.py",
    "gpu_required: marks tests as requiring GPU (will fail if GPU unavailable)",
    "cuda: marks tests that specifically test CUDA functionality",
    "cpu: marks tests that should run on CPU only",
    "integration: marks tests as integration tests",
]

# Default test configuration
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "--disable-warnings",
    "--timeout=300",
]
```

**Configuration Options:**

- `gpu_required`: Tests that require GPU (will fail if GPU unavailable)
- `gpu_safe`: GPU with single worker to avoid memory conflicts
- `sequential`: No parallel execution for maximum stability

### 3. Test Collection and Execution

#### Conftest Hooks (`tests/conftest.py`)

The actual conftest.py uses the following hooks:

- **`pytest_runtest_setup`**: Clears JAX caches before each test. For tests marked `gpu` or `cuda`, runs garbage collection and checks for GPU device availability, skipping the test if no GPU is accessible.
- **`pytest_collection_modifyitems`**: Skips tests marked `requires_prometheus` or `requires_psutil` when those optional dependencies are not installed (checked via `DependencyManager`). Slow, benchmark, integration, and end-to-end markers categorize tests for selection and reporting, but are not skipped by default.
- **`pytest_runtest_teardown`**: Clears JAX caches and runs garbage collection after each test. For GPU/CUDA tests, performs a dummy JAX operation to ensure the device is in a clean state.
- **`pytest_configure`**: Registers custom markers (`gpu_required`, `gpu_preferred`, `cpu_safe`, `slow`, `integration`, `cuda_local`, etc.) and suppresses JAX/CUDA warnings.

```python
# Example: GPU tests are skipped automatically when no GPU is present
@pytest.mark.gpu
def test_gpu_operation():
    """This test will be skipped if GPU is not available."""
    ...
```

**Key behaviors:**

- Tests marked `gpu` or `cuda` are automatically skipped when no GPU device is found
- Slow, benchmark, integration, and end-to-end tests run by default unless explicitly deselected with pytest marker selection
- Optional dependency tests are skipped gracefully when packages are missing

### 4. GPU-Safe Operations

#### Enhanced CSG Module (`opifex/geometry/csg.py`)

```python
class PeriodicCell:
    """Periodic cell for materials science calculations."""

    def __init__(self, lattice_vectors: Float[jax.Array, "3 3"]):
        """Initialize periodic cell."""
        self.lattice_vectors = jnp.asarray(lattice_vectors, dtype=jnp.float64)
        self.volume = jnp.abs(jnp.linalg.det(self.lattice_vectors))

        # Compute reciprocal lattice vectors using double precision for stability
        try:
            condition_number = jnp.linalg.cond(self.lattice_vectors)
        except Exception as e:
            # GPU operation failed - raise error instead of falling back to CPU
            raise RuntimeError(
                f"GPU operation failed during condition number computation: {e}. "
                "This may indicate GPU memory issues or CUDA driver problems. "
                "Please ensure GPU is properly configured and has sufficient memory."
            ) from e
```

**Key Changes:**

- Removed all CPU fallback mechanisms
- Clear error messages for GPU failures
- Proper exception handling with context

### 5. Running GPU Tests

## Usage Examples

### Pytest with GPU Markers

```bash
# Run GPU-required tests only
uv run pytest -m gpu_required

# Run GPU tests (automatically detected)
uv run pytest -m gpu

# Run CUDA-specific tests
uv run pytest -m cuda

# Skip GPU tests (CPU only)
uv run pytest -m "not gpu"

# Run with single worker for GPU safety
uv run pytest -n 1

# Full test reporting with JSON output and detailed coverage
uv run pytest -vv --json-report --json-report-file=temp/test-results.json --json-report-indent=2 --json-report-verbosity=2 --cov=src/opifex --cov-report=json:temp/coverage.json --cov-report=term-missing
```

### Marking Tests for GPU Requirements

```python
import pytest
import jax

@pytest.mark.gpu_required
def test_gpu_specific_operation():
    """Test that requires GPU."""
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if not gpu_devices:
        pytest.fail("GPU is required but not available")
    # ... test implementation

@pytest.mark.gpu
def test_gpu_operation():
    """Test that uses GPU but can be skipped."""
    # Automatically detected by conftest.py
    # ... test implementation

@pytest.mark.cuda
def test_cuda_specific_feature():
    """Test CUDA-specific functionality."""
    # ... test implementation
```

## Error Handling and Troubleshooting

### Common Error Messages

#### GPU Not Available

```text
FAILED: GPU is required but not available
Available devices: [cpu(id=0)]
```

**Solution:**

1. Install NVIDIA GPU drivers
2. Install CUDA toolkit
3. Reinstall with GPU support: `./setup.sh --force`

#### GPU Test Failed

```text
RuntimeError: GPU operation failed during computation: CUDA error: out of memory.
This may indicate GPU memory issues or CUDA driver problems.
```

**Solution:**

1. Check GPU memory usage: `nvidia-smi`
2. Close other GPU applications
3. Reduce batch sizes in tests
4. Use `uv run pytest -p no:xdist` for single-worker execution

#### CUDA Driver Issues

```text
RuntimeError: GPU operation failed during matrix inversion: CUDA driver version is insufficient for CUDA runtime version
```

**Solution:**

1. Update NVIDIA drivers
2. Ensure CUDA toolkit version matches driver version
3. Reinstall JAX with compatible CUDA version

### Environment Variables

**Note:** The automated `setup.sh` script configures all necessary environment variables automatically. The variables below are for manual configuration or debugging only.

```python
from enum import Enum
```
**📖 For complete environment configuration, see the [Environment Setup Guide](environment-setup.md).**

#### For GPU Testing (Manual Configuration)

```bash
# JAX Configuration for CUDA (matches setup.sh)
export JAX_PLATFORMS="cuda,cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.8"
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

# JAX CUDA Plugin Configuration
export JAX_CUDA_PLUGIN_VERIFY="false"
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK="1"

# Reduce CUDA warnings
export TF_CPP_MIN_LOG_LEVEL="1"
```

#### For Debugging

```bash
export JAX_DEBUG_NANS=true
export JAX_DEBUG_INFS=true
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
```

## Benefits of This Approach

### 1. Clear Requirements

- Tests explicitly indicate when GPU is required
- Users understand environment requirements upfront
- No hidden dependencies or silent failures

### 2. Consistent Behavior

- Same test behavior across all environments
- Predictable failure modes
- Reproducible test results

### 3. Better Error Messages

- Clear guidance on how to fix GPU issues
- Specific troubleshooting steps
- Environment-specific recommendations

### 4. Proper Testing

- GPU functionality is actually tested
- GPU-specific bugs are caught early
- Performance characteristics are validated

### 5. CI/CD Integration

- Clear pass/fail criteria for CI systems
- Proper environment validation
- Automated GPU requirement checking

## Migration Guide

### For Existing Tests

1. **Identify GPU-Dependent Tests**

   ```python
   # Look for tests that use:
   # - Neural operators
   # - GPU-specific operations
   # - Large matrix operations
   # - CUDA-specific functionality
   ```

2. **Add GPU Markers**

   ```python
   @pytest.mark.gpu_required  # For tests that require GPU
   def test_neural_operator():
       # ... test implementation
   ```

3. **Remove CPU Fallbacks**

   ```python
   # Before (remove this):
   try:
       gpu_operation()
   except Exception:
       cpu_fallback()

   # After (use this):
   gpu_operation()  # Will fail clearly if GPU unavailable
   ```

### For New Tests

1. **Always Mark GPU Requirements**

   ```python
   @pytest.mark.gpu_required
   def test_new_gpu_feature():
       # conftest.py automatically skips gpu-marked tests when no GPU is available
       # ... test implementation
   ```

2. **Use Clear Error Messages**

   ```python
   def gpu_operation():
       try:
           return jax_operation()
       except Exception as e:
           raise RuntimeError(
               f"GPU operation failed: {e}. "
               "Please ensure GPU is available and properly configured."
           ) from e
   ```

## Conclusion

This solution ensures that Opifex tests fail appropriately when GPU is not available, providing clear error messages and guidance for users. By removing CPU fallbacks, we ensure that:

1. **GPU requirements are explicit and enforced**
2. **Tests provide consistent behavior across environments**
3. **Users receive clear guidance on fixing GPU issues**
4. **GPU functionality is properly tested and validated**

The implementation provides multiple configuration options for different testing scenarios while maintaining the core principle that GPU-dependent tests should fail clearly when GPU is not available, rather than silently falling back to CPU.

# GPU Testing Solution - No CPU Fallback

## Overview

This document explains the comprehensive solution implemented to ensure that Opifex tests **fail appropriately** when GPU is not available, rather than silently falling back to CPU. This approach ensures that GPU-dependent functionality is properly tested and that users understand when their environment doesn't meet the requirements.

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

#### Core JAX Configuration (`opifex/core/jax_config.py`)

The framework uses device-agnostic JAX configuration that automatically detects and configures available hardware:

```python
def configure_jax_for_reliability() -> dict[str, str | bool | int | list[str]]:
    """Configure JAX runtime settings for maximum reliability across all platforms."""
    # Detect available devices
    devices = jax.devices()
    backend = jax.default_backend()

    # Enable 64-bit precision for reliability
    jax.config.update("jax_enable_x64", True)

    # Set memory management for optimal performance
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
    if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

**Key Features:**

- Automatic device detection and configuration
- Device-agnostic operation (CPU/GPU/TPU)
- Proper memory management settings
- 64-bit precision for numerical accuracy

#### GPU Testing Infrastructure (`opifex/core/testing_infrastructure.py`)

The framework includes comprehensive GPU testing infrastructure that handles device detection, stability testing, and environment classification:

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

#### Enhanced Conftest (`tests/conftest.py`)

```python
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle GPU requirements."""
    # Check if we're running GPU-required tests
    addopts = config.getoption("addopts", [])
    gpu_required = (
        "--config=gpu_required" in addopts or
        ("-m" in addopts and "gpu_required" in str(addopts))
    )

    if gpu_required:
        # Verify GPU is available before running tests
        try:
            require_gpu()
            print("✅ GPU is available - running GPU-required tests")
        except GPUUnavailableError as e:
            pytest.fail(f"GPU is required but not available: {e}")

    # Mark tests based on their content and GPU requirements
    for item in items:
        test_content = str(item.function)

        # Mark tests that use GPU-specific operations
        gpu_keywords = [
            "gpu", "cuda", "gpu_", "neural_operator", "manifold_neural",
            "periodic_cell", "lattice_vectors", "reciprocal_vectors"
        ]
        if any(keyword in test_content.lower() for keyword in gpu_keywords):
            item.add_marker(pytest.mark.gpu)
```

**Features:**

- Automatic test marking based on content analysis
- GPU availability verification before test execution
- Clear error messages when GPU is required but unavailable

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

### 5. Test Runner Script

#### Enhanced Test Runner (`scripts/run_tests_reliably.sh`)

```bash
#!/bin/bash
# Opifex Test Runner - GPU-Required Testing
# This script provides test configurations that require GPU and fail appropriately when GPU is not available

# Function to check GPU availability
check_gpu_availability() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

# Verify GPU is available before running tests
if ! check_gpu_availability; then
    print_error "GPU is not available. This test suite requires GPU."
    print_error "Please ensure:"
    print_error "  1. NVIDIA GPU is installed and working"
    print_error "  2. NVIDIA drivers are properly installed"
    print_error "  3. CUDA toolkit is installed"
    print_error "  4. JAX with CUDA support is installed"
    exit 1
fi
```

**Features:**

- GPU availability verification before test execution
- Clear error messages and troubleshooting guidance
- Multiple test configurations for different scenarios
- No CPU fallback options

## Usage Examples

### Running GPU-Required Tests

```bash
# Require GPU (tests will fail if GPU unavailable)
./scripts/run_tests_reliably.sh --gpu-required

# GPU-safe configuration (single worker)
./scripts/run_tests_reliably.sh --gpu-safe

# Sequential execution (no parallel)
./scripts/run_tests_reliably.sh --sequential

# Auto-detect best configuration
./scripts/run_tests_reliably.sh
```

### Direct Pytest Usage

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

# Comprehensive test reporting with JSON output and detailed coverage
uv run pytest -vv --json-report --json-report-file=temp/test-results.json --json-report-indent=2 --json-report-verbosity=2 --cov=opifex --cov-report=json:temp/coverage.json --cov-report=term-missing
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
4. Use `./scripts/run_tests_reliably.sh --gpu-safe` for single-worker execution

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
       require_gpu()
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

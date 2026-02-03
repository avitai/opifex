# Development Environment Setup

## UV Package Manager Configuration

### Cross-Filesystem Warning Resolution

If you encounter the following warning during development:

```
warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
```

This occurs when UV's cache directory and the project directory are on different filesystems. To resolve this:

#### Option 1: Set Environment Variable (Recommended)

```bash
export UV_LINK_MODE=copy
```

#### Option 2: Use Command Flag

```bash
uv sync --link-mode=copy
uv run --link-mode=copy <command>
```

#### Option 3: Configure in Shell Profile

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# UV Configuration for cross-filesystem setups
export UV_LINK_MODE=copy
```

### Performance Impact

- **Hardlinking**: ~50-100ms for dependency installation
- **File Copying**: ~200-500ms for dependency installation
- **Impact**: Minimal for development workflow, no impact on application performance

### Other Development Environment Variables

```bash
# JAX Configuration (already set)
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1

# UV Configuration
export UV_LINK_MODE=copy
```

## Pre-commit Configuration

All pre-commit hooks are configured to pass consistently. The following quality gates are enforced:

- **Code Formatting**: ruff format
- **Linting**: ruff check
- **Type Checking**: pyright
- **Security**: bandit
- **Documentation**: pydocstyle
- **File Formatting**: Various pre-commit hooks

### Running Pre-commit

```bash
# Run all hooks
uv run pre-commit run --all-files

# Install hooks (one-time setup)
uv run pre-commit install
```

### Quality Metrics Maintained

- **Type Safety**: 0 type errors across codebase
- **Code Quality**: 0 linting violations
- **Security**: 0 security issues identified
- **Documentation**: Complete docstring coverage for public APIs
- **Test Coverage**: Comprehensive test suite with 3500+ tests

# Opifex Development Setup Guide

This guide covers the development setup for the Opifex framework, including environment configuration, code quality standards, and best practices.

## 🚀 Quick Setup

```bash
# 1. Clone and setup environment
git clone https://github.com/mahdi-shafiei/opifex.git
cd opifex
./setup.sh

# 2. Activate environment
source ./activate.sh

# 3. Install pre-commit hooks
uv run pre-commit install
```

## 🔧 **Code Quality & Pre-commit Standards**

### **Overview**

The Opifex framework maintains production-grade code quality through comprehensive pre-commit hooks and automated tooling. This ensures consistency, security, and maintainability across the codebase.

### **Pre-commit Hook Configuration**

Our pre-commit setup includes:

- ✅ **TOML Formatting**: `sort_pyproject` - Maintains organized dependencies
- ✅ **Python Linting**: `ruff` - 88-character line limit, comprehensive rule set
- ✅ **Type Checking**: `pyright` - JAX-compatible type validation
- ✅ **Security Scanning**: `bandit` - Production security standards
- ✅ **Documentation**: `pydocstyle` - Google-style docstring enforcement
- ✅ **Shell Scripts**: `shellcheck` - Bash script quality validation

### **Common Pre-commit Issues & Solutions**

#### **1. Line Length Violations (E501)**

**Issue**: Lines exceeding 88 characters
**Solution**: Break long comments and function calls across multiple lines

```python
# ❌ Bad - Long comment line
# Shape: (batch, in_channels, spectral_size) @ (in_channels, out_channels, spectral_size) -> (batch, out_channels, spectral_size)

# ✅ Good - Split across multiple lines
# Shape: (batch, in_channels, spectral_size) @ (in_channels, out_channels,
# spectral_size) -> (batch, out_channels, spectral_size)
```

#### **2. Unnecessary Assignments (RET504)**

**Issue**: Variable assignment immediately before return
**Solution**: Return expression directly

```python
<!-- skip -->
```python
# ❌ Bad - Unnecessary assignment
result = computation()
return result

# ✅ Good - Direct return
return computation()
```

#### **3. Complex Functions**

**Issue**: Functions exceeding complexity limits
**Solution**: Extract helper methods

<!-- skip -->
```python
# ❌ Bad - Complex function
def complex_function(self, inputs):
    # 50+ lines of logic
    pass
```

<!-- skip -->
```python
# ✅ Good - Extracted helpers
def complex_function(self, inputs):
    prepared = self._prepare_inputs(inputs)
    processed = self._process_data(prepared)
    return self._finalize_output(processed)
```

### **Long-term Code Quality Strategy**

#### **1. Preventive Measures**

- **IDE Integration**: Configure your IDE with ruff and pyright extensions
- **Editor Settings**: Set line length to 88 characters with visual guides
- **Auto-formatting**: Enable format-on-save for consistent style
- **Pre-commit Installation**: Always run `uv run pre-commit install` in new clones

#### **2. Development Workflow**

```bash
# Daily development workflow
git checkout -b feature/my-feature
# ... make changes ...
uv run pre-commit run --all-files  # Check before commit
git add -A
git commit -m "feat: descriptive commit message"
```

#### **3. Handling Complex Scientific Code**

**Scientific computing often requires flexibility in certain rules:**

- **Mathematical Variables**: Single-letter variables (x, y, k) are acceptable in mathematical contexts
- **Long Parameter Lists**: Neural network constructors may have many parameters
- **Complex Algorithms**: Numerical algorithms may have higher complexity
- **Constants**: Magic numbers are acceptable for physical/mathematical constants

**Our configuration already accounts for these patterns through targeted rule exclusions.**

#### **4. Dependency Management**

**TOML Sorting**: The `sort_pyproject` hook automatically organizes dependencies:

- Alphabetical ordering within groups
- Consistent table key sorting
- Inline table formatting

**This ensures**:

- ✅ Easier dependency management
- ✅ Reduced merge conflicts
- ✅ Professional configuration appearance

#### **5. Documentation Standards**

**Google-style docstrings** are enforced for:

- Public modules, classes, and functions
- Complex algorithms requiring explanation
- API interfaces

**Relaxed requirements** for:

- Test files
- Internal helper methods
- Magic methods (`__init__`, `__call__`)

### **Emergency Fixes**

#### **Quick Fix Commands**

```bash
# Fix most formatting issues automatically
uv run pre-commit run --all-files

# Fix specific file
uv run ruff check --fix path/to/file.py

# Check types only
uv run pyright

# Skip pre-commit for emergency commits (use sparingly!)
git commit -m "emergency fix" --no-verify
```

#### **Configuration Updates**

If patterns emerge that need rule adjustments:

1. **Analyze the pattern**: Is it legitimate scientific computing usage?
2. **Update pyproject.toml**: Add targeted rule exclusions
3. **Document the decision**: Update this guide with rationale
4. **Test comprehensively**: Ensure no quality degradation

### **Best Practices for Scientific Computing**

#### **Code Organization**

<!-- skip -->
```python
# ✅ Good - Clear separation of concerns
class NeuralOperator(nnx.Module):
    def __init__(self, ...):
        # Initialization logic
        pass

    def _validate_inputs(self, x):
        # Input validation helper
        pass

    def _compute_spectral_transform(self, x):
        # Core mathematical computation
        pass

    def __call__(self, x):
        # Main interface - compose helpers
        x = self._validate_inputs(x)
        return self._compute_spectral_transform(x)
```

#### **Mathematical Comments**

```python
# ✅ Good - Clear mathematical explanation
# Fourier transform: F[f](k) = ∫ f(x) e^(-2πikx) dx
# Discretized: F[f]_k = Σ_j f_j e^(-2πijk/N)
fourier_coeffs = jnp.fft.fft(input_signal)
```

#### **Performance-Critical Code**

```python
# ✅ Good - Document performance considerations
@jax.jit  # JIT compilation for GPU acceleration
def spectral_convolution(x, weights):
    """Spectral convolution with GPU optimization.

    Note: Uses static shapes for XLA compatibility.
    """
    # Implementation with XLA-friendly operations
    pass
```

### **Quality Metrics & Monitoring**

#### **Automated Tracking**

- **Pre-commit success rate**: Monitor hook pass rates
- **Code coverage**: Maintain >80% for core algorithms
- **Type coverage**: Track type annotation completeness
- **Documentation coverage**: Ensure public APIs are documented

#### **Manual Review Points**

- **Algorithm correctness**: Peer review for mathematical implementations
- **Performance impact**: Benchmark critical paths
- **API consistency**: Maintain interface standards
- **Error handling**: Comprehensive error paths with informative messages

### **IDE Configuration Recommendations**

#### **VS Code Settings**

```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "none",
    "editor.formatOnSave": true,
    "editor.rulers": [88],
    "python.analysis.typeCheckingMode": "basic"
}
```

#### **PyCharm Settings**

- **Code Style**: Set line length to 88
- **Inspections**: Enable Ruff and Pyright
- **Format on Save**: Enable with Ruff formatter
- **Type Hints**: Enable type checking

This comprehensive approach ensures the Opifex framework maintains world-class code quality while accommodating the unique requirements of scientific computing and machine learning development.

## UV Configuration & Performance Optimization

### Hardlinking Performance Warning

You may see this warning during `uv` operations:

```
warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
```

**Root Cause**: This occurs when UV's cache directory and the project directory are on different filesystems (e.g., cache on SSD, project on HDD).

**Long-term Solution**: The Opifex framework now includes comprehensive UV configuration to optimize performance:

#### Manual Configuration

If you experience performance issues with UV hardlinking, manually set:

```bash
# Temporary (current session)
export UV_LINK_MODE=copy

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
```

### Performance Impact Analysis

| Configuration | Installation Time | Impact |
|---------------|------------------|---------|
| **Default (hardlink)** | ~50-100ms | ✅ Optimal when same filesystem |
| **Copy mode** | ~200-500ms | ✅ Consistent across all setups |
| **Cross-filesystem** | ~800ms-2s | ❌ Degraded without copy mode |

**Recommendation**: Use copy mode for consistent, predictable performance.

## 🔧 Troubleshooting & Verification

### Manual Troubleshooting Commands

If you prefer manual verification or need to debug specific issues:

```bash
# Check environment configuration
echo "UV_LINK_MODE: $UV_LINK_MODE"
echo "JAX_SKIP_CUDA_CONSTRAINTS_CHECK: $JAX_SKIP_CUDA_CONSTRAINTS_CHECK"

# Test UV performance
time uv sync --quiet  # Should complete in 200-500ms

# Verify pre-commit without warnings
uv run pre-commit run --all-files 2>&1 | grep -i "warning\|failed\|error"

```

### Common Issues & Solutions

#### Issue: "Failed to hardlink files" Warning

```
warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
```

**Solution**: Set the UV_LINK_MODE environment variable:

1. Set `export UV_LINK_MODE=copy` in your shell
2. Or add it permanently to your `~/.bashrc` or `~/.zshrc`

#### Issue: Slow Pre-commit Execution (>10 seconds)

**Symptoms**: Pre-commit takes significantly longer than expected

**Diagnosis**:

```bash
# Time the execution
time uv run pre-commit run --all-files

# Check for package reinstallation
uv run pre-commit run --all-files --verbose | grep "Installing"
```

**Solutions**:

1. Clear and rebuild pre-commit cache: `uv run pre-commit clean && uv run pre-commit install`
2. Check for package reinstallation in verbose output

#### Issue: Pre-commit Hooks Failing

**Symptoms**: Individual hooks return non-zero exit codes

**Diagnosis**:

```bash
# Run specific hook for detailed output
uv run pre-commit run ruff --verbose
uv run pre-commit run pyright --verbose
```

**Solutions**:

1. Fix code issues identified by the hooks
2. Update hook dependencies if needed
3. Check for configuration conflicts in `pyproject.toml`

### Performance Benchmarks

**Expected Performance (after optimization):**

| Operation | Time Range | Notes |
|-----------|------------|-------|
| **UV Sync** | 200-500ms | Initial dependency installation |
| **Pre-commit (all hooks)** | 5-8 seconds | All 15 quality gates |
| **Individual hooks** | 0.1-2 seconds | Varies by hook complexity |
| **Environment activation** | <100ms | Including UV configuration |

**If you're seeing significantly slower performance**, check the troubleshooting steps above.

## 🚀 Advanced Setup Options

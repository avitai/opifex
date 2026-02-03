# Code Quality Infrastructure

This document describes the comprehensive code quality infrastructure implemented in the Opifex framework, including pre-commit hooks, static analysis, testing, and enterprise-grade standards.

## 🎯 Overview

The Opifex framework maintains enterprise-grade code quality through a comprehensive infrastructure that ensures:

- **Perfect Compliance**: 19/19 pre-commit hooks passing with zero errors/warnings
- **Static Analysis**: Complete type safety with pyright and code quality with ruff
- **Security**: Automated security scanning with bandit
- **Documentation**: Consistent documentation standards with pydocstyle
- **Testing**: 1800+ tests with 99.8%+ pass rate
- **Dependency Management**: SQLAlchemy integration with type safety

## 🔧 Pre-commit Infrastructure

### Pre-commit Hooks Configuration

The framework uses a comprehensive `.pre-commit-config.yaml` with the following hooks:

#### 1. File Management & Formatting

```yaml
# Trim trailing whitespace
- repo: https://github.com/pre-commit/pre-commit-hooks
  hooks:
    - id: trailing-whitespace
    - id: end-of-file-fixer
    - id: check-merge-conflicts
    - id: check-added-large-files
```

#### 2. Configuration Validation

```yaml
# YAML, TOML, and JSON validation
- repo: https://github.com/pre-commit/pre-commit-hooks
  hooks:
    - id: check-yaml
    - id: check-toml
    - id: check-json
```

#### 3. Python Code Quality

```yaml
# Ruff for linting and formatting
- repo: https://github.com/astral-sh/ruff-pre-commit
  hooks:
    - id: ruff
      args: [--fix, --exit-non-zero-on-fix]
    - id: ruff-format

# Pyright for type checking
- repo: https://github.com/RobertCraigie/pyright-python
  hooks:
    - id: pyright
```

#### 4. Security Analysis

```yaml
# Bandit for security scanning
- repo: https://github.com/PyCQA/bandit
  hooks:
    - id: bandit
      args: ['-c', 'pyproject.toml']
```

#### 5. Documentation Standards

```yaml
# pydocstyle for documentation compliance
- repo: https://github.com/PyCQA/pydocstyle
  hooks:
    - id: pydocstyle
```

#### 6. Jupyter Notebook Quality

```yaml
# nbqa-ruff for notebook linting
- repo: https://github.com/nbQA-dev/nbQA
  hooks:
    - id: nbqa-ruff
```

#### 7. Shell Script Validation

```yaml
# shellcheck for shell script quality
- repo: https://github.com/shellcheck-py/shellcheck-py
  hooks:
    - id: shellcheck
```

### Running Pre-commit Hooks

#### Manual Execution

```bash
# Run all hooks on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff --all-files
uv run pre-commit run pyright --all-files

# Run hooks on staged files only
uv run pre-commit run
```

#### Automatic Execution

Pre-commit hooks run automatically on every commit:

```bash
# Install pre-commit hooks (done during setup)
uv run pre-commit install

# Hooks will run automatically on git commit
git commit -m "Your commit message"
```

## 🔍 Static Analysis

### Pyright Type Checking

#### Configuration

The framework uses a comprehensive `pyproject.toml` configuration for pyright:

```toml
[tool.pyright]
include = ["opifex", "tests", "examples"]
exclude = ["**/__pycache__"]
pythonVersion = "3.10"
pythonPlatform = "All"
typeCheckingMode = "strict"
reportMissingImports = true
reportMissingTypeStubs = false
reportUnusedImport = true
reportUnusedClass = true
reportUnusedFunction = true
reportDuplicateImport = true
reportOptionalSubscript = true
reportOptionalMemberAccess = true
reportOptionalCall = true
reportOptionalIterable = true
reportOptionalContextManager = true
reportOptionalOperand = true
reportTypedDictNotRequiredAccess = false
```

#### Key Features

- **Strict Type Checking**: Complete type safety across the entire codebase
- **JAX Integration**: Native support for JAX arrays and transformations
- **FLAX NNX Compatibility**: Full type coverage for neural network components
- **SQLAlchemy Integration**: Type-safe database operations
- **Scientific Computing Types**: Specialized type annotations for scientific computing

#### Type Safety Achievements

```python
# Example of comprehensive type annotations
from jax import Array
from jaxtyping import Float, Complex
from flax import nnx

def neural_operator_forward(
    model: nnx.Module,
    x: Float[Array, "batch spatial_dims channels"],
    training: bool = False
) -> Float[Array, "batch spatial_dims output_channels"]:
    """Type-safe neural operator forward pass."""
    return model(x, training=training)
```

### Ruff Code Quality

#### Configuration

```toml
[tool.ruff]
target-version = "py310"
line-length = 88
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "PL",  # pylint
    "SIM", # flake8-simplify
]
ignore = [
    "E501",   # line too long (handled by formatter)
    "PLR0913", # too many arguments
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

#### Key Features

- **Code Formatting**: Consistent code style across the entire codebase
- **Import Sorting**: Automatic import organization with isort
- **Complexity Analysis**: Detection of overly complex functions and classes
- **Bug Detection**: Identification of common Python bugs and anti-patterns
- **Performance Optimization**: Suggestions for performance improvements

#### Recent Fixes Applied

```python
```

<!-- skip -->
```python
# Before: Line length violation
very_long_variable_name = some_function_with_many_parameters(param1, param2, param3, param4, param5)

# After: Proper line breaking
very_long_variable_name = some_function_with_many_parameters(
    param1, param2, param3, param4, param5
)
```

<!-- skip -->
```python
# Before: Complex function with 21 branches
def _check_boundary_conditions(self, ...):
    # 51 statements with 21 branches
```

<!-- skip -->
```python
# After: Refactored into helper methods
def _check_boundary_conditions(self, ...):
    # 10 statements with 2 branches
    return self._check_dirichlet_boundary_condition(...)

def _check_dirichlet_boundary_condition(self, ...):
    # Extracted helper method
```

## 🔒 Security Analysis

### Bandit Security Scanning

#### Configuration

```toml
[tool.bandit]
exclude_dirs = ["tests", "examples"]
skips = ["B101", "B601"]  # Skip assert and shell usage in tests
```

#### Security Features

- **SQL Injection Prevention**: Detection of potential SQL injection vulnerabilities
- **Command Injection Prevention**: Identification of unsafe shell command usage
- **Cryptography Best Practices**: Validation of cryptographic implementations
- **File System Security**: Detection of unsafe file operations
- **Network Security**: Identification of insecure network operations

## 📚 Documentation Standards

### pydocstyle Compliance

#### Configuration

```toml
[tool.pydocstyle]
convention = "google"
add-ignore = ["D100", "D104", "D105", "D107"]
```

#### Documentation Requirements

- **Google Style Docstrings**: Consistent documentation format
- **Type Annotations**: Complete type information in function signatures
- **Examples**: Code examples in docstrings where appropriate
- **Mathematical Notation**: LaTeX formatting for mathematical expressions

#### Example Documentation

```python
def fourier_neural_operator(
    x: Float[Array, "batch spatial_dims channels"],
    modes: int = 12,
    width: int = 64
) -> Float[Array, "batch spatial_dims output_channels"]:
    """Fourier Neural Operator for learning solution operators.

    This function implements the Fourier Neural Operator (FNO) architecture
    for learning mappings between function spaces. The FNO uses spectral
    convolutions in the Fourier domain to capture global dependencies.

    Args:
        x: Input tensor with spatial dimensions and channels.
        modes: Number of Fourier modes to retain in spectral convolution.
        width: Hidden dimension width for the neural network layers.

    Returns:
        Output tensor with the same spatial dimensions but potentially
        different number of channels.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((32, 64, 64, 3))  # batch=32, spatial=64x64, channels=3
        >>> y = fourier_neural_operator(x, modes=12, width=64)
        >>> print(y.shape)  # (32, 64, 64, 1)

    References:
        Li, Z., et al. "Fourier Neural Operator for Parametric Partial
        Differential Equations." ICLR 2021.
    """
    # Implementation here...
```

## 🧪 Testing Infrastructure

### Test Coverage

#### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Component interaction testing
3. **End-to-End Tests**: Complete workflow testing
4. **Performance Tests**: Benchmarking and optimization validation
5. **Regression Tests**: Prevention of functionality regressions

#### Test Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=opifex",
    "--cov-report=term-missing",
    "--cov-report=html",
]
markers = [
    "slow: marks tests as slow",
    "gpu: marks tests as requiring GPU",
    "integration: marks tests as integration tests",
]
```

#### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=opifex --cov-report=html

# Run specific test categories
uv run pytest tests/unit/ -v          # Unit tests
uv run pytest tests/integration/ -v   # Integration tests
uv run pytest -m "not slow" -v       # Skip slow tests
uv run pytest -m gpu -v              # GPU tests only
```

## 🔄 Continuous Integration

### GitHub Actions Integration

The pre-commit configuration is designed to match GitHub Actions CI/CD pipeline:

```yaml
# .github/workflows/ci.yml (example)
name: CI
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run pre-commit
        run: uv run pre-commit run --all-files
      - name: Run tests
        run: uv run pytest tests/ -v
```

### Local Development Workflow

1. **Setup**: Install pre-commit hooks during environment setup
2. **Development**: Write code with type annotations and documentation
3. **Commit**: Pre-commit hooks run automatically, fixing issues
4. **Push**: Code passes all quality checks before reaching CI/CD
5. **Review**: Code review focuses on logic and design, not style

## 📊 Quality Metrics

## 🛠️ Maintenance

### Regular Quality Checks

```bash
# Daily quality check
uv run pre-commit run --all-files

# Weekly comprehensive check
uv run pytest tests/ --cov=opifex
uv run pre-commit autoupdate

# Monthly dependency update
uv sync --upgrade
uv run pre-commit run --all-files
```

### Quality Standards Enforcement

1. **No Commits Without Passing Hooks**: Pre-commit hooks prevent commits with quality issues
2. **Type Safety Required**: All new code must include proper type annotations
3. **Documentation Required**: All public functions must have docstrings
4. **Security Validation**: All code must pass security scanning
5. **Test Coverage**: New features must include comprehensive tests

## 🎯 Best Practices

### Development Guidelines

1. **Type Annotations**: Use comprehensive type annotations with jaxtyping
2. **Documentation**: Write clear docstrings with examples
3. **Testing**: Include unit and integration tests for new features
4. **Error Handling**: Implement robust error handling and validation
5. **Performance**: Consider performance implications and optimize when necessary

### Code Style Guidelines

1. **Line Length**: Maximum 88 characters (enforced by ruff)
2. **Import Organization**: Automatic sorting with isort
3. **Function Complexity**: Keep functions simple and focused
4. **Variable Naming**: Use descriptive names following PEP 8
5. **Comments**: Use comments sparingly, prefer self-documenting code

### Security Guidelines

1. **Input Validation**: Validate all external inputs
2. **SQL Safety**: Use parameterized queries for database operations
3. **File Operations**: Validate file paths and permissions
4. **Network Security**: Use secure protocols and validate connections
5. **Dependency Management**: Keep dependencies updated and secure

## 📚 Additional Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pyright Documentation](https://microsoft.github.io/pyright/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [pydocstyle Documentation](http://www.pydocstyle.org/)
- [JAX Type Annotations](https://jax.readthedocs.io/en/latest/type_promotion.html)
- [FLAX NNX Documentation](https://flax.readthedocs.io/en/latest/nnx/index.html)

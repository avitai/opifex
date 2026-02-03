# Testing

## Overview

Testing strategies and tools for Opifex development.

## Running Tests

### Basic Test Suite

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/neural/
pytest tests/training/

# Run with coverage
pytest --cov=opifex --cov-report=html

# Comprehensive test reporting with JSON output and detailed coverage
uv run pytest -vv --json-report --json-report-file=temp/test-results.json --json-report-indent=2 --json-report-verbosity=2 --cov=opifex --cov-report=json:temp/coverage.json --cov-report=term-missing
```

### Comprehensive Test Reporting

The comprehensive test command above generates detailed reports for CI/CD integration and analysis:

**Output Files:**

- `temp/test-results.json`: Detailed test results in JSON format with full verbosity
- `temp/coverage.json`: Coverage data in JSON format for programmatic analysis
- Terminal output: Coverage report with missing lines highlighted

**Note:** The `temp/` directory is automatically created if it doesn't exist. Output files are suitable for CI/CD integration and automated analysis tools.

**Use Cases:**

- **CI/CD Integration**: Machine-readable test and coverage data
- **Quality Analysis**: Detailed test metrics and coverage tracking
- **Debugging**: Verbose output for investigating test failures
- **Reporting**: Structured data for dashboard integration

**Command Breakdown:**

- `-vv`: Very verbose output for detailed test information
- `--json-report`: Enable JSON test result reporting
- `--json-report-file=temp/test-results.json`: Output location for test results
- `--json-report-indent=2`: Pretty-print JSON with 2-space indentation
- `--json-report-verbosity=2`: Maximum verbosity in JSON output
- `--cov=opifex`: Enable coverage for the opifex package
- `--cov-report=json:temp/coverage.json`: Output coverage data as JSON
- `--cov-report=term-missing`: Show missing coverage in terminal

### GPU Tests

```bash
# Run GPU-specific tests
pytest tests/ -k "gpu"

# Verify GPU functionality
python scripts/verify_opifex_gpu.py
```

## Test Structure

### Unit Tests

- Individual component testing
- Isolated functionality verification
- Fast execution

### Integration Tests

- Component interaction testing
- End-to-end workflows
- Realistic scenarios

### Performance Tests

- Benchmarking critical paths
- Memory usage validation
- Scalability testing

## Writing Tests

### Test Guidelines

```python
import pytest
import jax.numpy as jnp
from opifex.neural import PINN

def test_pinn_creation():
    """Test PINN model creation."""
    pinn = PINN(layers=[2, 10, 1])
    assert pinn is not None

def test_pinn_forward_pass():
    """Test PINN forward computation."""
    pinn = PINN(layers=[2, 10, 1])
    x = jnp.array([[0.5, 0.5]])
    output = pinn(x)
    assert output.shape == (1, 1)
```

### Fixtures

```python
@pytest.fixture
def sample_pde_problem():
    """Fixture for test PDE problems."""
    return PDEProblem(
        equation=heat_equation,
        domain=unit_square,
        boundary_conditions=dirichlet_bcs
    )
```

## Continuous Integration

Tests run automatically on:

- Pull requests
- Main branch commits
- Scheduled nightly builds

### Pre-commit Hooks

```bash
# Setup pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

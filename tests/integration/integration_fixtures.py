"""Integration test fixtures for Opifex framework."""

import jax
import pytest
from flax import nnx

from .framework import OpifexTestFramework


@pytest.fixture(scope="session")
def integration_framework():
    """Provide shared integration test framework."""
    return OpifexTestFramework()


@pytest.fixture
def fluid_problem_simple(integration_framework):
    """Simple fluid dynamics problem for testing."""
    return integration_framework.create_test_problem("fluid", "simple")


@pytest.fixture
def fluid_problem_medium(integration_framework):
    """Medium complexity fluid dynamics problem for testing."""
    return integration_framework.create_test_problem("fluid", "medium")


@pytest.fixture
def quantum_problem_simple(integration_framework):
    """Simple quantum mechanics problem for testing."""
    return integration_framework.create_test_problem("quantum", "simple")


@pytest.fixture
def quantum_problem_medium(integration_framework):
    """Medium complexity quantum mechanics problem for testing."""
    return integration_framework.create_test_problem("quantum", "medium")


@pytest.fixture
def test_rngs():
    """Provide consistent random number generators for tests."""
    return nnx.Rngs(42)


@pytest.fixture
def integration_device():
    """Provide appropriate device for integration testing."""
    devices = jax.devices()
    return devices[0] if devices else None


@pytest.fixture
def performance_benchmark():
    """Provide performance benchmarking utility."""

    def _benchmark(operation, expected_time=None):
        framework = OpifexTestFramework()
        expected = {"max_execution_time": expected_time} if expected_time else None
        return framework.benchmark_performance(operation, expected)

    return _benchmark

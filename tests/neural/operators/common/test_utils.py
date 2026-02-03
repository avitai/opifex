"""Test utilities for neural operator tests.

Provides utilities for consistent testing across JAX backends.
"""

import jax
import pytest


class TestEnvironmentManager:
    """Manages test environment configuration for universal compatibility."""

    @staticmethod
    def get_current_platform():
        """Get the current platform name for display purposes.

        Returns:
            str: Current JAX backend name (e.g., 'gpu', 'cpu', 'tpu')
        """
        return jax.default_backend()

    @staticmethod
    def get_backend_info():
        """Get information about the current JAX backend.

        Returns:
            dict: Backend information including name and available devices
        """
        backend = jax.default_backend()
        devices = jax.devices()

        return {
            "backend": backend,
            "num_devices": len(devices),
            "device_types": {device.device_kind for device in devices},
            "devices": devices,
        }

    @staticmethod
    def get_test_tolerances():
        """Get consistent test tolerances for all backends.

        Returns:
            dict: Universal tolerance values for test comparisons
        """
        return {"rtol": 1e-6, "atol": 1e-6, "grad_rtol": 1e-5, "grad_atol": 1e-5}


@pytest.fixture
def rng_key():
    """Provide a JAX random key for testing."""
    return jax.random.PRNGKey(42)

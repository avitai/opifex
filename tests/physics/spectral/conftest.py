"""Double-precision fixture for the spectral-solver accuracy tests.

The ETDRK4 properties under test -- a machine-exact linear propagator, fourth-order
temporal convergence, and the KdV invariants -- are only verifiable in float64;
single precision floors the error near ``1e-5`` and hides the ``dt^4`` scaling.
The root ``reset_jax_config`` fixture pins ``jax_enable_x64=False`` per test, so this
fixture declares a dependency on it to run *after* the reset and re-enable x64.
"""

import jax
import pytest


@pytest.fixture(autouse=True)
def _enable_float64(reset_jax_config: None) -> None:  # pyright: ignore[reportUnusedFunction]
    jax.config.update("jax_enable_x64", True)

"""Double-precision fixture for the quasiseparable CARMA equivalence tests.

``test_quasisep_carma`` compares the scalable O(n) Kalman CARMA GP against the
direct dense exact GP and asserts agreement at the ``1e-6`` level -- only reachable
in float64. Its arrays are written with ``dtype=jnp.float64``, but the suite-wide
``reset_jax_config`` fixture pins ``jax_enable_x64=False`` per test, so float64
silently degrades to float32, whose BLAS noise floor differs across platforms
(passes on Linux/OpenBLAS, fails on macOS/Accelerate). Re-enable x64 for this module
only -- after the reset -- so the comparison is exact and platform-independent
(mirrors ``tests/physics/spectral/conftest.py``). Other GP test modules are float32
and keep the default configuration.
"""

import jax
import pytest


@pytest.fixture(autouse=True)
def _enable_float64_for_carma(  # pyright: ignore[reportUnusedFunction]
    request: pytest.FixtureRequest, reset_jax_config: None
) -> None:
    if request.module.__name__.rsplit(".", 1)[-1] == "test_quasisep_carma":
        jax.config.update("jax_enable_x64", True)

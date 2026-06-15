"""Shared JAX numerical pinning for the neural-wavefunction / VMC tests.

VMC energies are validated against *exact* reference values to chemical
accuracy (~1 mHa), so float64 is enabled for the whole package and the matmul
precision is pinned ``high``. On GPU, float32 matmuls otherwise fall back to
TF32 (~1e-3 error), which is two orders of magnitude larger than the milli-
Hartree tolerances the energy assertions require (the TF32 guard documented in
the root conftest). ``high`` is the error-corrected 3xTF32 path -- full fp32
accuracy at tensor-core speed.
"""

from __future__ import annotations

from collections.abc import Iterator  # noqa: TC003

import jax
import pytest


@pytest.fixture(scope="function", autouse=True)
def vmc_x64_and_precision() -> Iterator[None]:
    """Enable x64 and pin matmul precision for every VMC test."""
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "high")
    yield
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "high")

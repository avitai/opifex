"""Shared fixtures and JAX numerical pinning for the quantum integral tests.

The Gaussian-integral engine is validated at float64 against an eager
cross-check and PySCF, so x64 is enabled for the whole package and the matmul
precision is pinned ``high`` (the TF32 guard documented in the root conftest --
GPU float32 matmuls otherwise drift ~1e-3 and break the tight integral
assertions).
"""

from __future__ import annotations

from collections.abc import Iterator  # noqa: TC003

import jax
import pytest


@pytest.fixture(scope="function", autouse=True)
def quantum_x64_and_precision() -> Iterator[None]:
    """Enable x64 and pin matmul precision for every integral test."""
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "high")
    yield
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "high")

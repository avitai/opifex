"""Timing helpers for JAX computations."""

from __future__ import annotations

from typing import Any

import jax


def block_until_ready(value: Any) -> Any:
    """Synchronize any JAX arrays contained in ``value`` and return ``value``."""
    return jax.block_until_ready(value)

"""JAX runtime settings.

Device identity (platform, kind, count) is ``substrax.devices.detect_devices()``.
"""

from __future__ import annotations

import jax


def configure_jax_precision(enable_x64: bool = True) -> None:
    """Configure JAX precision settings.

    Args:
        enable_x64: Whether to enable 64-bit precision (default: True)
    """
    jax.config.update("jax_enable_x64", enable_x64)

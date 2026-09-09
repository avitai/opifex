"""Device helpers: one-release wrappers over substrax's device identity.

``get_device_info``, ``get_platform`` and ``is_gpu_available`` read
``substrax.devices.detect_devices()``; each call emits a ``DeprecationWarning``
and the names are removed in 0.2.3. ``configure_jax_precision`` stays.
"""

from __future__ import annotations

import logging

import jax
from substrax.devices import detect_devices, DeviceKind

from opifex._deprecated import warn_deprecated


_logger = logging.getLogger(__name__)
_HOME = "substrax.devices.detect_devices"


def get_device_info() -> dict[str, str | bool | int | list[str]]:
    """The visible devices as substrax reports them, in the historical dictionary shape.

    Returns:
        ``available_devices``, ``default_backend``, ``device_count``,
        ``gpu_available`` and ``cpu_available``.
    """
    warn_deprecated("opifex.core.get_device_info", _HOME)
    info = detect_devices()
    return {
        "available_devices": list(info.device_kinds),
        "default_backend": info.platform,
        "device_count": info.count,
        "gpu_available": info.kind is DeviceKind.GPU,
        "cpu_available": True,
    }


def get_platform() -> str:
    """The default JAX backend name: ``substrax.devices.detect_devices().platform``."""
    warn_deprecated("opifex.core.get_platform", _HOME)
    return detect_devices().platform


def is_gpu_available() -> bool:
    """Whether the default backend is a GPU: ``detect_devices().kind is DeviceKind.GPU``."""
    warn_deprecated("opifex.core.is_gpu_available", _HOME)
    return detect_devices().kind is DeviceKind.GPU


def configure_jax_precision(enable_x64: bool = True) -> None:
    """Configure JAX precision settings.

    Args:
        enable_x64: Whether to enable 64-bit precision (default: True)
    """
    jax.config.update("jax_enable_x64", enable_x64)

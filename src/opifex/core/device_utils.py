"""Device utilities for JAX device detection and management.

This module provides utilities for detecting and managing JAX devices
in a consistent way across the Opifex framework.
"""


def get_device_info() -> dict[str, str | bool | int | list[str]]:
    """Get comprehensive information about available JAX devices.

    Returns:
        Dictionary containing device information with the following keys:
        - available_devices: List of available device strings
        - default_backend: Default JAX backend (cpu, gpu, tpu)
        - device_count: Number of available devices
        - gpu_available: Whether GPU backend is available
        - cpu_available: Whether CPU backend is available

    Example:
        >>> device_info = get_device_info()
        >>> print(f"Backend: {device_info['default_backend']}")
        >>> print(f"GPU available: {device_info['gpu_available']}")
    """
    try:
        import jax

        devices = jax.devices()
        backend = jax.default_backend()

        return {
            "available_devices": [str(d) for d in devices],
            "default_backend": backend,
            "device_count": len(devices),
            "gpu_available": backend == "gpu",
            "cpu_available": True,  # CPU is always available as fallback
        }
    except Exception:
        # Fallback configuration if JAX is not available or fails
        return {
            "available_devices": ["cpu:0"],
            "default_backend": "cpu",
            "device_count": 1,
            "gpu_available": False,
            "cpu_available": True,
        }


def get_platform() -> str:
    """Get the current JAX platform/backend.

    Returns:
        String indicating the current JAX backend (cpu, gpu, tpu)

    Example:
        >>> platform = get_platform()
        >>> print(f"Running on: {platform}")
    """
    try:
        import jax

        return jax.default_backend()
    except Exception:
        return "cpu"


def is_gpu_available() -> bool:
    """Check if GPU backend is available.

    Returns:
        True if GPU backend is available, False otherwise

    Example:
        >>> if is_gpu_available():
        ...     print("GPU acceleration available")
    """
    device_info = get_device_info()
    gpu_available = device_info.get("gpu_available", False)
    # Ensure we return a boolean even if the value is not what we expect
    return gpu_available is True


def configure_jax_precision(enable_x64: bool = True) -> None:
    """Configure JAX precision settings.

    Args:
        enable_x64: Whether to enable 64-bit precision (default: True)

    Example:
        >>> configure_jax_precision(enable_x64=True)
        >>> # JAX will now use 64-bit precision
    """
    try:
        import jax

        jax.config.update("jax_enable_x64", enable_x64)
    except Exception:
        # Silently fail if JAX is not available
        pass

"""Hardware identity is substrax's and hardware specs are calibrax's.

``RooflineMemoryManager`` reads its roofline numbers from
``calibrax.profiling.detect_hardware_specs``; ``MixedPrecisionOptimizer`` reads the
accelerator class from ``substrax.devices.detect_devices``.
"""

from __future__ import annotations

import importlib.util

import jax.numpy as jnp
from calibrax.profiling import detect_hardware_specs
from substrax.devices import detect_devices, DeviceKind

import opifex
import opifex.core
from opifex.core.gpu_acceleration import MixedPrecisionOptimizer, RooflineMemoryManager


def test_roofline_specs_are_calibrax_specs_plus_memory_and_platform() -> None:
    specs = RooflineMemoryManager().hw_specs
    expected = detect_hardware_specs()
    info = detect_devices()

    for key in ("peak_flops", "memory_bandwidth", "critical_intensity"):
        assert specs[key] == expected[key]
    assert specs["platform"] == info.platform
    assert specs["supports_tensorcore"] == bool(expected.get("tensor_core_shapes"))
    assert specs["memory_gb"] > 0


def test_mixed_precision_hardware_config_reads_substrax_and_calibrax() -> None:
    config = MixedPrecisionOptimizer().hardware_config
    info = detect_devices()
    shapes = detect_hardware_specs().get("tensor_core_shapes", [])

    assert config["supports_tensorcore"] == bool(shapes)
    assert config["tensor_shapes"] == list(shapes)
    if info.kind is DeviceKind.CPU:
        assert config["optimal_dtype"] == jnp.float32
        assert config["alignment"] == 1
    else:
        assert config["optimal_dtype"] == jnp.bfloat16
        assert config["alignment"] > 1


def test_jax_runtime_helpers_are_gone() -> None:
    """JAX runtime settings are ``substrax.runtime``'s: opifex keeps no setup helper."""
    assert importlib.util.find_spec("opifex.core.device_utils") is None
    for name in ("get_device_info", "get_platform", "is_gpu_available", "configure_jax_precision"):
        assert not hasattr(opifex.core, name), name
        assert name not in opifex.core.__all__, name
    assert not hasattr(opifex, "setup_jax_optimization")

"""Hardware identity is substrax's and hardware specs are calibrax's.

``RooflineMemoryManager`` reads its roofline numbers from
``calibrax.profiling.detect_hardware_specs``; ``MixedPrecisionOptimizer`` reads the
accelerator class from ``substrax.devices.detect_devices``; the published
``opifex.core`` device helpers are one-release wrappers over substrax that warn
and are removed in 0.2.3.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from calibrax.profiling import detect_hardware_specs
from substrax.devices import detect_devices, DeviceKind

from opifex.core import device_utils
from opifex.core.gpu_acceleration import MixedPrecisionOptimizer, RooflineMemoryManager


if TYPE_CHECKING:
    from collections.abc import Callable


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


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("get_platform", lambda: detect_devices().platform),
        ("is_gpu_available", lambda: detect_devices().kind is DeviceKind.GPU),
    ],
)
def test_device_helpers_are_deprecated_wrappers_over_substrax(
    name: str, expected: Callable[[], object]
) -> None:
    with pytest.warns(DeprecationWarning, match=rf"{name}.*substrax"):
        value = getattr(device_utils, name)()

    assert value == expected()


def test_get_device_info_is_substrax_device_info_as_a_dict() -> None:
    info = detect_devices()

    with pytest.warns(DeprecationWarning, match="get_device_info.*substrax"):
        payload = device_utils.get_device_info()

    assert payload == {
        "available_devices": list(info.device_kinds),
        "default_backend": info.platform,
        "device_count": info.count,
        "gpu_available": info.kind is DeviceKind.GPU,
        "cpu_available": True,
    }


def test_configure_jax_precision_still_flips_x64() -> None:
    import jax

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        device_utils.configure_jax_precision(enable_x64=False)
    assert jax.config.jax_enable_x64 is False

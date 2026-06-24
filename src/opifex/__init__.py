"""
Opifex: Unified Scientific Machine Learning Framework

A JAX-native platform for scientific machine learning with probabilistic-first design,
high performance, and production-ready architecture.
"""

import logging
import os
from email.utils import parseaddr
from importlib.metadata import metadata, PackageNotFoundError
from pathlib import Path

import jax


try:
    # Single source of truth: project metadata declared in pyproject.toml, read
    # from the installed package rather than duplicated here. ``Author-email`` is
    # the PEP 621 combined ``"Name <email>"`` form, split via ``parseaddr``.
    _metadata = metadata("opifex")
    __version__ = _metadata["Version"]
    __author__, __email__ = parseaddr(_metadata["Author-email"] or "")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
    __author__ = ""
    __email__ = ""


def _append_xla_flags(flags: list[str]) -> None:
    """Append XLA flags to the ``XLA_FLAGS`` env var, skipping ones already set."""
    existing_flags = os.environ.get("XLA_FLAGS", "")
    for flag in flags:
        if flag not in existing_flags:
            existing_flags += f" {flag}"
    os.environ["XLA_FLAGS"] = existing_flags.strip()


def setup_jax_optimization() -> None:
    """Setup JAX optimizations for improved performance.

    This function configures:
    - XLA compilation cache for faster startup
    - Platform-specific optimizations
    - Memory management settings
    - Backend-specific configurations

    Based on profiling harness recommendations for optimal performance.
    """
    # XLA Compilation Cache Configuration
    cache_dir = os.environ.get("OPIFEX_XLA_CACHE_DIR", ".cache/jax")

    # Ensure cache directory exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Platform-specific optimizations
    backend = jax.default_backend()

    if backend == "gpu":
        # GPU-specific optimizations for TensorCore utilization
        _append_xla_flags(
            [
                "--xla_gpu_enable_triton_gemm=true",
                "--xla_gpu_enable_latency_hiding_scheduler=true",
                # Hopper/Blackwell speed-of-light latency model -> better scheduling.
                "--xla_gpu_enable_analytical_sol_latency_estimator=true",
            ]
        )
        # Enable high precision matmul for GPU
        jax.config.update("jax_default_matmul_precision", "high")

    elif backend == "tpu":
        # TPU-specific optimizations
        _append_xla_flags(
            [
                "--xla_tpu_enable_async_collective_fusion=true",
                "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
            ]
        )
        # TPU uses default precision
        jax.config.update("jax_default_matmul_precision", "default")

    else:
        # CPU optimizations
        _append_xla_flags(
            [
                "--xla_cpu_enable_fast_math=true",
                "--xla_cpu_fast_math_honor_infs=true",
                "--xla_cpu_fast_math_honor_nans=true",
            ]
        )

    # General performance optimizations
    _append_xla_flags(["--xla_force_host_platform_device_count=1"])

    # Memory management
    jax.config.update("jax_enable_x64", False)  # Use 32-bit by default for performance

    # Enable compilation cache
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    logger = logging.getLogger(__name__)
    logger.info("🚀 Opifex JAX Optimizations Enabled:")
    logger.info("   • Backend: %s", backend)
    logger.info("   • XLA Cache: %s", cache_dir)
    logger.info("   • Device count: %s", jax.device_count())
    logger.info("   • XLA Flags: %s", os.environ.get("XLA_FLAGS", "None"))


# Auto-configuration is opt-in (Rule 13: no hidden side effects at import
# time). Importing ``opifex`` mutates nothing global — no ``os.environ``,
# no ``jax.config``, no cache directory. To enable the JAX performance
# optimisations, call :func:`setup_jax_optimization` explicitly at process
# startup (recommended in application entry points / notebooks):
#
#     import opifex
#     opifex.setup_jax_optimization()
#
# The previous ``OPIFEX_AUTO_CONFIGURE`` environment toggle is removed: an
# explicit function call is the single, discoverable configuration path.

"""
Opifex: Unified Scientific Machine Learning Framework

A JAX-native platform for scientific machine learning with probabilistic-first design,
high performance, and production-ready architecture.
"""

import logging
import os
from pathlib import Path

import jax


__version__ = "0.1.0"
__author__ = "Opifex Team"
__email__ = "team@opifex.io"


def setup_jax_optimization():
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
        gpu_flags = [
            "--xla_gpu_enable_triton_gemm=true",
            "--xla_gpu_enable_latency_hiding_scheduler=true",
        ]

        existing_flags = os.environ.get("XLA_FLAGS", "")
        for flag in gpu_flags:
            if flag not in existing_flags:
                existing_flags += f" {flag}"

        os.environ["XLA_FLAGS"] = existing_flags.strip()

        # Enable high precision matmul for GPU
        jax.config.update("jax_default_matmul_precision", "high")

    elif backend == "tpu":
        # TPU-specific optimizations
        tpu_flags = [
            "--xla_tpu_enable_async_collective_fusion=true",
            "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
        ]

        existing_flags = os.environ.get("XLA_FLAGS", "")
        for flag in tpu_flags:
            if flag not in existing_flags:
                existing_flags += f" {flag}"

        os.environ["XLA_FLAGS"] = existing_flags.strip()

        # TPU uses default precision
        jax.config.update("jax_default_matmul_precision", "default")

    else:
        # CPU optimizations
        cpu_flags = [
            "--xla_cpu_enable_fast_math=true",
            "--xla_cpu_fast_math_honor_infs=true",
            "--xla_cpu_fast_math_honor_nans=true",
        ]

        existing_flags = os.environ.get("XLA_FLAGS", "")
        for flag in cpu_flags:
            if flag not in existing_flags:
                existing_flags += f" {flag}"

        os.environ["XLA_FLAGS"] = existing_flags.strip()

    # General performance optimizations
    general_flags = [
        "--xla_force_host_platform_device_count=1",
    ]

    existing_flags = os.environ.get("XLA_FLAGS", "")
    for flag in general_flags:
        if flag not in existing_flags:
            existing_flags += f" {flag}"

    os.environ["XLA_FLAGS"] = existing_flags.strip()

    # Memory management
    jax.config.update("jax_enable_x64", False)  # Use 32-bit by default for performance

    # Enable compilation cache
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    logger = logging.getLogger(__name__)
    logger.info("🚀 Opifex JAX Optimizations Enabled:")
    logger.info(f"   • Backend: {backend}")
    logger.info(f"   • XLA Cache: {cache_dir}")
    logger.info(f"   • Device count: {jax.device_count()}")
    logger.info(f"   • XLA Flags: {os.environ.get('XLA_FLAGS', 'None')}")


# Automatically setup optimizations on import
setup_jax_optimization()

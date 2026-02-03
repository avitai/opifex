#!/usr/bin/env python
"""
Quick backend checker for Opifex framework.

Run this script to see which backend (CPU/CUDA) JAX is currently using.
"""

import os
import sys
from pathlib import Path


# Add the project root to the path so we can import opifex modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from opifex.core.device_utils import get_device_info


def check_backend():
    """Check and display the current JAX backend being used."""
    try:
        import jax
        import jax.numpy as jnp

        # Direct JAX imports instead of jax_config

        print("\n" + "=" * 50)
        print("🔧 JAX Backend Status Check")
        print("=" * 50)

        # Basic JAX info
        print(f"📱 JAX Version: {jax.__version__}")

        # Get actual JAX backend info
        jax_backend = jax.default_backend()
        test_array = jnp.array([1.0, 2.0, 3.0])
        actual_device = test_array.device
        device_info = get_device_info()

        # Determine backend type clearly
        # Note: JAX reports "gpu" backend when using CUDA
        using_gpu = jax_backend == "gpu"
        backend_type = "🟢 CUDA/GPU" if using_gpu else "🔵 CPU"

        print(f"🎪 Backend Type: {backend_type}")
        print(f"🎯 JAX Backend: {jax_backend}")
        print(f"✅ Active Device: {actual_device}")
        print(f"\n💻 Available Devices: {device_info['available_devices']}")
        print(f"🚀 GPU Available: {device_info['gpu_available']}")
        print(f"🖥️  CPU Available: {device_info['cpu_available']}")

        # Show relevant environment variables
        print("\n🔧 Environment Variables:")
        env_vars = [
            "JAX_PLATFORM_NAME",
            "JAX_PLATFORMS",
            "CUDA_VISIBLE_DEVICES",
            "JAX_ENABLE_X64",
        ]

        for var in env_vars:
            value = os.environ.get(var, "Not set")
            print(f"   {var}: {value}")

        print("=" * 50)

        return {
            "backend": jax.default_backend(),
            "platform": jax_backend,
            "is_gpu": jax_backend == "gpu",
            "device": str(actual_device),
        }

    except ImportError as e:
        print(f"❌ Could not import required modules: {e}")
        return None
    except Exception as e:
        print(f"❌ Error checking backend: {e}")
        return None


if __name__ == "__main__":
    result = check_backend()
    if result:
        print(f"\n🎯 Summary: Using {result['backend']} backend on {result['device']}")
    else:
        print("❌ Failed to determine backend")
        sys.exit(1)

#!/usr/bin/env python
"""Enhanced GPU utilities for Opifex framework with workshop insights."""

import os
import subprocess
import sys


def detect_cuda_version() -> int | None:
    """Detect CUDA version from nvidia-smi with enhanced parsing.

    Returns:
        Optional[int]: CUDA major version (e.g., 11 or 12) or None if not found
    """
    try:
        output = subprocess.check_output(["nvidia-smi"], text=True)
        for line in output.split("\n"):
            if "CUDA Version" in line:
                # Extract version number with better parsing
                version_str = line.split("CUDA Version:")[1].strip().split()[0]
                return int(float(version_str))
    except (subprocess.SubprocessError, FileNotFoundError, IndexError, ValueError):
        pass
    return None


def detect_cuda_version_detailed() -> dict[str, str | None]:
    """Detect detailed CUDA information from multiple sources.

    Returns:
        dict: Detailed CUDA information including driver, runtime, and compute capability
    """
    info: dict[str, str | None] = {
        "driver_version": None,
        "cuda_version": None,
        "gpu_name": None,
        "compute_capability": None,
        "memory_total": None,
        "cuda_runtime": None,
    }

    try:
        # Get driver and CUDA version from nvidia-smi
        output = subprocess.check_output(["nvidia-smi"], text=True)
        lines = output.split("\n")

        for line in lines:
            if "CUDA Version" in line:
                parts = line.split("|")
                for part in parts:
                    if "CUDA Version" in part:
                        info["cuda_version"] = part.split("CUDA Version:")[1].strip()
            elif "Driver Version" in line:
                parts = line.split("|")
                for part in parts:
                    if "Driver Version" in part:
                        info["driver_version"] = (
                            part.split("Driver Version:")[1].strip().split()[0]
                        )

        # Get GPU details
        try:
            gpu_output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )

            if gpu_output.strip():
                parts = [p.strip() for p in gpu_output.strip().split(",")]
                if len(parts) >= 3:
                    info["gpu_name"] = parts[0]
                    info["memory_total"] = f"{parts[1]} MB"
                    info["compute_capability"] = parts[2]
        except subprocess.SubprocessError:
            pass

    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Try to get CUDA runtime version
    try:
        import jax

        # JAX can sometimes provide CUDA runtime info
        devices = jax.devices("cuda")
        if devices:
            # This is a simplified approach - in practice, CUDA runtime detection is complex
            info["cuda_runtime"] = "Available via JAX"
    except Exception:
        pass

    return info


def get_gpu_memory_info() -> dict[str, int | None]:
    """Get detailed GPU memory information.

    Returns:
        dict: Memory information with total, used, and free memory in MB
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )

        parts = [int(p.strip()) for p in output.strip().split(",")]
        return {"total": parts[0], "used": parts[1], "free": parts[2]}
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        return {"total": None, "used": None, "free": None}


def get_jax_cuda_extra() -> str:
    """Get appropriate JAX CUDA extra based on detected CUDA version.

    Returns:
        str: JAX CUDA extra (e.g., "cuda11_pip" or "cuda12_pip") or empty string if no GPU
    """
    cuda_version = detect_cuda_version()
    if cuda_version is None:
        return ""

    if cuda_version < 12:
        return "cuda11_pip"
    return "cuda12_pip"


def has_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available using nvidia-smi.

    Returns:
        bool: True if GPU is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_optimal_jax_env_vars() -> dict[str, str]:
    """Get optimal JAX environment variables for GPU usage with workshop insights.

    Returns:
        dict: Dictionary of environment variables and their values
    """
    return {
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",  # More conservative than 0.8
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_ENABLE_X64": "0",  # Keep 32-bit for better GPU performance
        "TF_CPP_MIN_LOG_LEVEL": "1",  # Reduce TensorFlow logging
        "JAX_PLATFORMS": "cuda,cpu",  # Prefer CUDA but fallback to CPU
        "XLA_FLAGS": "--xla_gpu_strict_conv_algorithm_picker=false",  # Workshop insight
        "JAX_CUDA_PLUGIN_VERIFY": "false",  # Bypass cuSPARSE issues
        "JAX_SKIP_CUDA_CONSTRAINTS_CHECK": "1",  # Skip constraint checks that cause issues
        "CUDA_ROOT": "/usr/local/cuda",
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
    }


def get_cpu_only_env_vars() -> dict[str, str]:
    """Get environment variables for CPU-only execution.

    Returns:
        dict: Dictionary of environment variables for CPU execution
    """
    return {
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "0",
        "TF_CPP_MIN_LOG_LEVEL": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }


def configure_jax_env_vars(force_cpu: bool = False) -> None:
    """Configure environment variables for optimal JAX performance.

    Args:
        force_cpu: If True, configure for CPU-only execution
    """
    if force_cpu or not has_nvidia_gpu():
        env_vars = get_cpu_only_env_vars()
    else:
        env_vars = get_optimal_jax_env_vars()

    for key, value in env_vars.items():
        os.environ[key] = value


def get_jax_installation_command() -> str:
    """Get the appropriate command to install JAX with GPU support if available.

    Returns:
        str: Command to install JAX with proper CUDA support
    """
    cuda_extra = get_jax_cuda_extra()
    if cuda_extra:
        # Use the find-links approach for CUDA installations
        return (
            f"uv pip install --find-links "
            f"https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "
            f'"jax[{cuda_extra}]>=0.6.1" "jaxlib>=0.6.1"'
        )
    return 'uv pip install "jax>=0.6.1" "jaxlib>=0.6.1"'


def check_jax_gpu_status() -> dict[str, str | list | bool | None]:
    """Check JAX GPU status with comprehensive information.

    Returns:
        dict: JAX and GPU status information
    """
    status: dict[str, str | list | bool | None] = {
        "jax_available": False,
        "jax_version": None,
        "default_backend": None,
        "all_devices": [],
        "gpu_devices": [],
        "gpu_available": False,
        "cuda_available": False,
        "error": None,
    }

    try:
        import jax

        status["jax_available"] = True
        status["jax_version"] = jax.__version__
        status["default_backend"] = jax.default_backend()

        devices = jax.devices()
        status["all_devices"] = [str(d) for d in devices]

        try:
            gpu_devices = jax.devices("cuda")
            status["gpu_devices"] = [str(d) for d in gpu_devices]
            status["gpu_available"] = len(gpu_devices) > 0
            status["cuda_available"] = any(
                "cuda" in str(d).lower() for d in gpu_devices
            )
        except Exception as gpu_error:
            status["error"] = f"GPU device check failed: {gpu_error}"

    except ImportError as e:
        status["error"] = f"JAX not available: {e}"
    except Exception as e:
        status["error"] = f"JAX status check failed: {e}"

    return status


def print_comprehensive_gpu_info() -> None:
    """Print comprehensive GPU and JAX information."""
    print("🔍 Opifex GPU & JAX Status Report")
    print("=" * 50)

    # System GPU info
    print("\n📊 System GPU Information:")
    if has_nvidia_gpu():
        print("✅ NVIDIA GPU detected")

        cuda_info = detect_cuda_version_detailed()
        for key, value in cuda_info.items():
            if value:
                formatted_key = key.replace("_", " ").title()
                print(f"   {formatted_key}: {value}")

        memory_info = get_gpu_memory_info()
        if memory_info["total"]:
            print(
                f"   Memory Usage: {memory_info['used']}/{memory_info['total']} MB "
                f"({memory_info['free']} MB free)"
            )
    else:
        print("❌ No NVIDIA GPU detected")

    # JAX status
    print("\n🧪 JAX Status:")
    jax_status = check_jax_gpu_status()
    if jax_status["jax_available"]:
        print(f"✅ JAX {jax_status['jax_version']} available")
        print(f"   Default backend: {jax_status['default_backend']}")
        print(f"   Available devices: {jax_status['all_devices']}")
        if jax_status["gpu_available"]:
            print(f"   GPU devices: {jax_status['gpu_devices']}")
        if jax_status["error"]:
            print(f"   ⚠️  Warning: {jax_status['error']}")
    else:
        print(f"❌ JAX not available: {jax_status['error']}")

    # Recommendation
    print("\n🎯 Recommendation:")
    if has_nvidia_gpu() and jax_status.get("gpu_available"):
        print("✅ System is ready for GPU-accelerated Opifex development")
    elif has_nvidia_gpu():
        print("⚠️  GPU detected but JAX cannot access it")
        print("   Try: uv pip install --upgrade 'jax[cuda12_pip]'")
    else:
        print("INFO: No GPU detected - system configured for CPU-only execution")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    # Enhanced script behavior with comprehensive reporting
    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        print_comprehensive_gpu_info()
    else:
        # Quick status check
        print(f"GPU available: {has_nvidia_gpu()}")
        print(f"CUDA version: {detect_cuda_version()}")
        print(f"Recommended JAX CUDA extra: {get_jax_cuda_extra()}")
        print(f"Recommended installation command: {get_jax_installation_command()}")

        # Quick JAX check
        jax_status = check_jax_gpu_status()
        if jax_status["jax_available"]:
            print(f"\nJAX {jax_status['jax_version']} status:")
            print(f"Available devices: {jax_status['all_devices']}")
            if jax_status["gpu_available"]:
                print(f"🎉 GPU devices: {jax_status['gpu_devices']}")
            else:
                print("📱 CPU-only mode")
        else:
            print(f"\n❌ JAX status: {jax_status.get('error', 'Not available')}")

        print("\nRun with --comprehensive for detailed information")

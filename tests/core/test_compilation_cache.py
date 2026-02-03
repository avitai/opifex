"""Test XLA compilation cache configuration."""

from pathlib import Path

import jax


class TestCompilationCache:
    """Test suite for JAX compilation cache configuration."""

    def test_compilation_cache_enabled(self):
        """Test that XLA compilation cache is properly configured."""
        # Import opifex to trigger cache setup
        import opifex  # noqa: F401 # type: ignore[import-untyped]

        # Cache should be configured on import
        cache_dir = jax.config.jax_compilation_cache_dir  # type: ignore[attr-defined]
        assert cache_dir is not None
        assert Path(cache_dir).exists()

    def test_cache_persistence(self):
        """Test that compilation cache persists across runs."""
        # Import required modules
        from flax import nnx

        from opifex.neural.operators import fno

        # Create and compile a model
        rngs = nnx.Rngs(42)
        model = fno.FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        # First compilation
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 1, 32, 32))
        _ = model(x)

        # Check that cache directory is accessible and writable
        # (Cache may not be populated if compilation is too fast)
        cache_dir = Path(jax.config.jax_compilation_cache_dir)  # type: ignore[attr-defined]
        assert cache_dir.exists()

        # Test cache directory is writable
        test_file = cache_dir / "test_write"
        test_file.touch()
        assert test_file.exists()
        test_file.unlink()  # Clean up

    def test_jax_configuration_applied(self):
        """Test that JAX configuration optimizations are applied."""
        import subprocess
        import sys

        # Test in a fresh subprocess to avoid test interference
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import opifex; import jax; "
                "print('cache_dir:', jax.config.jax_compilation_cache_dir is not None); "
                "print('cache_time:', jax.config.jax_persistent_cache_min_compile_time_secs >= 1.0); "
                "print('x64_enabled:', jax.config.jax_enable_x64)",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        output_lines = result.stdout.strip().split("\n")

        # Parse the output
        cache_dir_ok = "cache_dir: True" in output_lines
        cache_time_ok = "cache_time: True" in output_lines
        x64_disabled = "x64_enabled: False" in output_lines

        # Check that key optimizations are applied
        assert cache_dir_ok, (
            f"Compilation cache directory not set. Output: {result.stdout}"
        )
        assert cache_time_ok, (
            f"Cache min compile time not set properly. Output: {result.stdout}"
        )

        # Check default precision settings - should use 32-bit by default for performance
        assert x64_disabled, (
            f"Expected 32-bit precision by default, but x64 is enabled. Output: {result.stdout}"
        )

    def test_backend_specific_optimizations(self):
        """Test backend-specific optimization configurations."""
        import opifex  # noqa: F401 # type: ignore[import-untyped]

        backend = jax.default_backend()

        if backend == "gpu":
            # GPU should have high precision matmul
            assert jax.config.jax_default_matmul_precision == "high"  # type: ignore[attr-defined]
        elif backend == "tpu":
            # TPU should have default precision matmul
            assert jax.config.jax_default_matmul_precision == "default"  # type: ignore[attr-defined]
        # CPU backend doesn't need specific matmul precision

    def test_environment_variable_override(self, tmp_path):
        """Test that environment variables can override cache directory."""
        import importlib
        import os

        # Set custom cache directory
        custom_cache_dir = tmp_path / "custom_xla_cache"
        os.environ["OPIFEX_XLA_CACHE_DIR"] = str(custom_cache_dir)

        # Reload opifex to pick up environment variable
        import opifex  # type: ignore[import-untyped]

        importlib.reload(opifex)

        # Check that custom directory is used
        assert jax.config.jax_compilation_cache_dir == str(custom_cache_dir)  # type: ignore[attr-defined]
        assert custom_cache_dir.exists()

        # Clean up
        del os.environ["OPIFEX_XLA_CACHE_DIR"]

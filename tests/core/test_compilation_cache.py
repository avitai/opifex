"""Test XLA compilation cache configuration."""

from pathlib import Path

import jax


class TestCompilationCache:
    """Test suite for JAX compilation cache configuration."""

    def test_setup_configures_the_compilation_cache(self):
        """``setup_jax_optimization`` points JAX's compilation cache at an existing directory."""
        from opifex import setup_jax_optimization

        setup_jax_optimization()

        cache_dir = jax.config.jax_compilation_cache_dir  # type: ignore[attr-defined]
        assert cache_dir is not None
        assert Path(cache_dir).exists()

    def test_cache_persistence(self):
        """Test that compilation cache persists across runs."""
        # Import required modules
        from flax import nnx

        from opifex import setup_jax_optimization
        from opifex.neural.operators import fno

        setup_jax_optimization()

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

    def test_import_configures_nothing_and_setup_configures_jax(self, tmp_path):
        """Importing opifex leaves JAX's config alone; ``setup_jax_optimization`` applies it.

        A fresh interpreter, without the ``JAX_COMPILATION_CACHE_DIR`` that CI exports (JAX
        reads it itself, which would hide an import-time setup), reports the cache directory
        before and after the explicit call.
        """
        import os
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if k != "JAX_COMPILATION_CACHE_DIR"}
        env["OPIFEX_XLA_CACHE_DIR"] = str(tmp_path / "xla-cache")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import opifex; import jax; "
                "print('imported:', jax.config.jax_compilation_cache_dir); "
                "opifex.setup_jax_optimization(); "
                "print('cache_dir:', jax.config.jax_compilation_cache_dir); "
                "print('cache_time:', jax.config.jax_persistent_cache_min_compile_time_secs); "
                "print('x64_enabled:', jax.config.jax_enable_x64)",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        output_lines = result.stdout.strip().split("\n")

        assert "imported: None" in output_lines, f"import configured JAX. Output: {result.stdout}"
        assert f"cache_dir: {tmp_path / 'xla-cache'}" in output_lines, result.stdout
        assert "cache_time: 1.0" in output_lines, result.stdout
        assert "x64_enabled: False" in output_lines, result.stdout

    def test_backend_specific_optimizations(self):
        """Test backend-specific optimization configurations.

        ``setup_jax_optimization`` is opt-in (Rule 13: no hidden import-time
        side effects) — tests that exercise the JAX-config side effects must
        invoke it explicitly instead of relying on ``import opifex``.
        """
        from opifex import setup_jax_optimization

        setup_jax_optimization()

        backend = jax.default_backend()

        if backend == "gpu":
            # GPU should have high precision matmul
            assert jax.config.jax_default_matmul_precision == "high"  # type: ignore[attr-defined]
        elif backend == "tpu":
            # TPU should have default precision matmul
            assert jax.config.jax_default_matmul_precision == "default"  # type: ignore[attr-defined]
        # CPU backend doesn't need specific matmul precision

    def test_environment_variable_override(self, tmp_path):
        """Test that environment variables can override cache directory.

        Since ``setup_jax_optimization`` is opt-in (see Rule 13 in
        ``__init__.py``), the test invokes it explicitly after pointing
        ``OPIFEX_XLA_CACHE_DIR`` at the temp dir.
        """
        import os

        # Set custom cache directory
        custom_cache_dir = tmp_path / "custom_xla_cache"
        os.environ["OPIFEX_XLA_CACHE_DIR"] = str(custom_cache_dir)

        try:
            from opifex import setup_jax_optimization

            setup_jax_optimization()

            assert jax.config.jax_compilation_cache_dir == str(custom_cache_dir)  # type: ignore[attr-defined]
            assert custom_cache_dir.exists()
        finally:
            del os.environ["OPIFEX_XLA_CACHE_DIR"]

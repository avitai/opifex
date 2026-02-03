"""
Test suite for PDE Residual Registry.

This module tests the PDEResidualRegistry class, which provides:
- Global registration of custom PDE residual functions
- Thread-safe registration and retrieval
- Built-in PDE residual implementations
- Clear error messages for missing PDEs
- Listing and introspection capabilities

Tests follow TDD principles - they define expected behavior.
"""

import jax.numpy as jnp
import pytest
from flax import nnx


class TestBasicRegistration:
    """Test basic registration and retrieval of PDE residuals."""

    def test_register_and_retrieve_simple_pde(self):
        """Test registering a simple PDE and retrieving it."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        # Clear registry for clean test
        PDEResidualRegistry._clear_registry()

        # Define a simple PDE residual function
        def my_pde(model, x, autodiff_engine):
            """Simple test PDE."""
            return jnp.sum(model(x) ** 2)

        # Register the PDE
        PDEResidualRegistry.register("my_pde", my_pde)

        # Retrieve and verify
        retrieved = PDEResidualRegistry.get("my_pde")
        assert retrieved is my_pde

    def test_register_using_decorator(self):
        """Test registration using decorator syntax."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        # Register using decorator
        @PDEResidualRegistry.register("decorated_pde")
        def decorated_pde(model, x, autodiff_engine):
            """Decorated PDE."""
            return jnp.mean(model(x))

        # Verify registration
        retrieved = PDEResidualRegistry.get("decorated_pde")
        assert retrieved is decorated_pde

    def test_decorator_returns_original_function(self):
        """Test that decorator returns the original function unchanged."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        def original_function(model, x, autodiff_engine):
            return x

        decorated = PDEResidualRegistry.register("test")(original_function)
        assert decorated is original_function

    def test_retrieve_nonexistent_pde_raises_error(self):
        """Test that retrieving non-existent PDE raises clear error."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        with pytest.raises(KeyError) as exc_info:
            PDEResidualRegistry.get("nonexistent_pde")

        assert "nonexistent_pde" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()


class TestRegistryOperations:
    """Test registry listing and introspection operations."""

    def test_list_registered_pdes(self):
        """Test listing all registered PDEs."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        # Register multiple PDEs
        PDEResidualRegistry.register("pde1", lambda m, x, e: x)
        PDEResidualRegistry.register("pde2", lambda m, x, e: x)
        PDEResidualRegistry.register("pde3", lambda m, x, e: x)

        # List should contain all custom PDEs plus built-ins
        names = PDEResidualRegistry.list()
        assert "pde1" in names
        assert "pde2" in names
        assert "pde3" in names
        # Built-ins should also be present
        assert "poisson" in names
        assert "heat" in names

    def test_list_empty_registry(self):
        """Test listing when registry is cleared (only built-ins remain)."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        names = PDEResidualRegistry.list()
        # Built-in PDEs should always be present (5 original + 5 from Batch 5 + 3 from Batch 6)
        builtin_pdes = [
            "poisson",
            "heat",
            "wave",
            "burgers",
            "schrodinger",
            "schrodinger_td",
            "navier_stokes",
            "maxwell",
            "schrodinger_nonlinear",
            "reaction_diffusion",
            "homogenization",
            "two_scale",
            "amr_poisson",
        ]
        assert set(names) == set(builtin_pdes)

    def test_contains_check(self):
        """Test checking if a PDE is registered."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        PDEResidualRegistry.register("existing", lambda m, x, e: x)

        assert PDEResidualRegistry.contains("existing")
        assert not PDEResidualRegistry.contains("nonexistent")

    def test_get_info_returns_metadata(self):
        """Test that get_info returns PDE metadata."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        def my_pde(model, x, autodiff_engine):
            """This is my PDE with a docstring."""
            return x

        PDEResidualRegistry.register("my_pde", my_pde)

        info = PDEResidualRegistry.get_info("my_pde")
        assert info["name"] == "my_pde"
        assert info["function"] is my_pde
        assert "This is my PDE" in info["docstring"]


class TestDuplicateRegistration:
    """Test handling of duplicate registrations."""

    def test_duplicate_registration_raises_warning(self):
        """Test that registering duplicate name raises warning."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        PDEResidualRegistry.register("duplicate", lambda m, x, e: x)

        # Second registration should raise warning but succeed
        with pytest.warns(UserWarning, match="already registered"):
            PDEResidualRegistry.register("duplicate", lambda m, x, e: x * 2)

    def test_duplicate_with_override_succeeds(self):
        """Test that duplicate registration with override flag succeeds."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        func1 = lambda m, x, e: x
        func2 = lambda m, x, e: x * 2

        PDEResidualRegistry.register("pde", func1)
        PDEResidualRegistry.register("pde", func2, override=True)

        # Should get the second function
        retrieved = PDEResidualRegistry.get("pde")
        assert retrieved is func2


class TestBuiltInPDEs:
    """Test built-in PDE residual functions."""

    def test_poisson_pde_registered(self):
        """Test that Poisson equation is pre-registered."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        assert PDEResidualRegistry.contains("poisson")

    def test_heat_pde_registered(self):
        """Test that heat equation is pre-registered."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        assert PDEResidualRegistry.contains("heat")

    def test_wave_pde_registered(self):
        """Test that wave equation is pre-registered."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        assert PDEResidualRegistry.contains("wave")

    def test_burgers_pde_registered(self):
        """Test that Burgers equation is pre-registered."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        assert PDEResidualRegistry.contains("burgers")

    def test_schrodinger_pde_registered(self):
        """Test that Schrödinger equation is pre-registered."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        assert PDEResidualRegistry.contains("schrodinger")


class TestPoissonResidual:
    """Test Poisson equation residual computation."""

    def test_poisson_residual_signature(self):
        """Test Poisson residual has correct signature."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        poisson = PDEResidualRegistry.get("poisson")

        # Should accept model, x, autodiff_engine, and optional source_term
        # Verify it's callable
        assert callable(poisson)

    def test_poisson_residual_with_callable(self):
        """Test Poisson residual with simple callable."""
        from opifex.core.physics import AutoDiffEngine
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        poisson = PDEResidualRegistry.get("poisson")

        # Known solution: u = x² + y²
        # Laplacian: ∇²u = 4
        def analytical_solution(x):
            return x[..., 0] ** 2 + x[..., 1] ** 2

        x = jnp.array([[1.0, 2.0], [0.5, 0.5]])

        # Compute residual with source term = 4
        source_term = jnp.full(x.shape[0], 4.0)
        residual = poisson(
            analytical_solution, x, AutoDiffEngine, source_term=source_term
        )

        # Residual should be ∇²u - f ≈ 0
        assert residual.shape == (2,)
        assert jnp.allclose(residual, 0.0, atol=1e-5)

    def test_poisson_residual_with_nnx_model(self):
        """Test Poisson residual with NNX model."""
        from opifex.core.physics import AutoDiffEngine
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        poisson = PDEResidualRegistry.get("poisson")

        # Simple NNX model
        class SimplePINN(nnx.Module):
            def __init__(self, rngs):
                self.dense = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimplePINN(rngs=nnx.Rngs(0))
        x = jnp.array([[0.5, 0.5]])

        # Should not raise and return correct shape
        source_term = jnp.zeros(1)
        residual = poisson(model, x, AutoDiffEngine, source_term=source_term)
        assert residual.shape == (1,)


class TestHeatResidual:
    """Test heat equation residual computation."""

    def test_heat_residual_signature(self):
        """Test heat residual has correct signature."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        heat = PDEResidualRegistry.get("heat")
        assert callable(heat)

    def test_heat_residual_with_callable(self):
        """Test heat residual with simple callable."""
        from opifex.core.physics import AutoDiffEngine
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        heat = PDEResidualRegistry.get("heat")

        # For steady-state heat: ∇²u = 0
        # Use u = x² - y² (harmonic function)
        def harmonic_solution(x):
            return x[..., 0] ** 2 - x[..., 1] ** 2

        x = jnp.array([[1.0, 1.0], [0.5, 0.5]])

        # Steady-state: alpha = 0 or use laplacian directly
        residual = heat(harmonic_solution, x, AutoDiffEngine, alpha=1.0)

        # For harmonic function, ∇²u = 0
        assert residual.shape == (2,)
        assert jnp.allclose(residual, 0.0, atol=1e-5)


class TestThreadSafety:
    """Test thread-safe registration."""

    def test_concurrent_registration(self):
        """Test that concurrent registrations don't corrupt registry."""
        import threading

        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        def register_pde(i):
            PDEResidualRegistry.register(f"pde_{i}", lambda m, x, e: x * i)

        # Register 10 PDEs concurrently
        threads = [threading.Thread(target=register_pde, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 10 custom PDEs should be registered (plus 13 built-ins)
        all_names = PDEResidualRegistry.list()
        assert len(all_names) == 23  # 10 custom + 13 built-ins
        for i in range(10):
            assert PDEResidualRegistry.contains(f"pde_{i}")


class TestErrorMessages:
    """Test that error messages are clear and helpful."""

    def test_get_nonexistent_suggests_similar(self):
        """Test that error message suggests similar registered names."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()
        PDEResidualRegistry.register("poisson", lambda m, x, e: x)

        with pytest.raises(KeyError) as exc_info:
            PDEResidualRegistry.get("possion")  # Typo

        error_msg = str(exc_info.value)
        # Should mention available PDEs
        assert "available" in error_msg.lower() or "registered" in error_msg.lower()

    def test_register_invalid_name_raises_error(self):
        """Test that invalid PDE names raise errors."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        # Empty name
        with pytest.raises(ValueError, match=r"name.*empty"):
            PDEResidualRegistry.register("", lambda m, x, e: x)

        # Non-string name
        with pytest.raises(TypeError):
            PDEResidualRegistry.register(123, lambda m, x, e: x)  # type: ignore  # noqa: PGH003

    def test_register_non_callable_raises_error(self):
        """Test that registering non-callable raises error."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        with pytest.raises(TypeError, match="callable"):
            PDEResidualRegistry.register("invalid", "not a function")  # type: ignore  # noqa: PGH003


class TestIntegrationWithAutoDiffEngine:
    """Test integration with AutoDiffEngine."""

    def test_custom_pde_uses_autodiff_engine(self):
        """Test custom PDE using AutoDiffEngine utilities."""
        from opifex.core.physics import AutoDiffEngine
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        # Custom Helmholtz equation: ∇²u + k²u = 0
        @PDEResidualRegistry.register("helmholtz")
        def helmholtz(model, x, autodiff_engine, wavenumber=1.0):  # pyright: ignore[reportUnusedFunction]
            """Helmholtz equation residual."""
            # Use autodiff_engine parameter
            laplacian = autodiff_engine.compute_laplacian(model, x)
            u = model(x)
            if u.ndim > 1:
                u = u.squeeze(-1)
            return laplacian + wavenumber**2 * u

        # Test with simple function
        def test_func(x):
            return jnp.sum(x**2, axis=-1)

        x = jnp.array([[1.0, 1.0]])
        helmholtz_fn = PDEResidualRegistry.get("helmholtz")
        residual = helmholtz_fn(test_func, x, AutoDiffEngine, wavenumber=2.0)

        # Should compute without errors
        assert residual.shape == (1,)
        assert jnp.isfinite(residual).all()


class TestDocumentation:
    """Test that PDEs have proper documentation."""

    def test_builtin_pdes_have_docstrings(self):
        """Test that built-in PDEs have docstrings."""
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        builtin_pdes = ["poisson", "heat", "wave", "burgers", "schrodinger"]

        for pde_name in builtin_pdes:
            if PDEResidualRegistry.contains(pde_name):
                pde_func = PDEResidualRegistry.get(pde_name)
                assert pde_func.__doc__ is not None
                assert len(pde_func.__doc__.strip()) > 0

    def test_get_info_includes_signature(self):
        """Test that get_info includes function signature."""

        from opifex.core.physics.pde_registry import PDEResidualRegistry

        PDEResidualRegistry._clear_registry()

        def my_pde(model, x, autodiff_engine, param1=1.0):
            """Test PDE."""
            return x

        PDEResidualRegistry.register("my_pde", my_pde)

        info = PDEResidualRegistry.get_info("my_pde")
        sig = info["signature"]

        # Should contain parameter names
        assert "model" in str(sig)
        assert "x" in str(sig)
        assert "autodiff_engine" in str(sig)

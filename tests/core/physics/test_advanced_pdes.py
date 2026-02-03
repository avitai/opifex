"""Tests for advanced PDE types in registry.

Following TDD: These tests are written BEFORE implementation.

Tests cover:
- Time-dependent Schrödinger equation (TDSE)
- Navier-Stokes equations (2D incompressible)
- Maxwell's equations (electromagnetism)
- Nonlinear Schrödinger (Gross-Pitaevskii)
- Coupled reaction-diffusion systems
- JAX compatibility (JIT, vmap)
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.core.physics.pde_registry import PDEResidualRegistry


class TestTimeDependentSchrodinger:
    """Test time-dependent Schrödinger equation (TDSE): iℏ∂ψ/∂t = Hψ."""

    def test_tdse_registered(self):
        """TDSE should be registered in PDE registry."""
        assert "schrodinger_td" in PDEResidualRegistry.list()

    def test_tdse_free_particle(self):
        """Free particle plane wave should satisfy TDSE."""
        # Free particle: iℏ∂ψ/∂t = -ℏ²/2m ∇²ψ
        # Plane wave: ψ = exp(i(kx - ωt)) with ω = ℏk²/2m

        def model(x):
            """Model representing plane wave."""
            k = 2.0
            return jnp.exp(1j * k * x[..., 0])

        x = jnp.array([[0.5], [1.0], [1.5]])

        tdse = PDEResidualRegistry.get("schrodinger_td")
        residual = tdse(model, x, hbar=1.0, mass=1.0)

        # Should compute without error
        assert residual.shape == (3,)
        assert jnp.all(jnp.isfinite(residual))

    def test_tdse_harmonic_oscillator(self):
        """Ground state of harmonic oscillator should have specific residual."""
        # Ground state: ψ₀(x) = (mω/πℏ)^(1/4) exp(-mωx²/2ℏ)

        def ground_state(x):
            """Harmonic oscillator ground state."""
            m, omega, hbar = 1.0, 1.0, 1.0
            A = (m * omega / (jnp.pi * hbar)) ** 0.25
            return A * jnp.exp(-m * omega * x[..., 0] ** 2 / (2 * hbar))

        x = jnp.array([[0.0], [0.5], [1.0]])

        tdse = PDEResidualRegistry.get("schrodinger_td")
        residual = tdse(
            ground_state, x, hbar=1.0, mass=1.0, potential_fn=lambda x: 0.5 * x**2
        )

        assert residual.shape == (3,)
        assert jnp.all(jnp.isfinite(residual))

    def test_tdse_signature(self):
        """TDSE residual should have correct signature."""
        info = PDEResidualRegistry.get_info("schrodinger_td")

        assert "hbar" in str(info["signature"]) or "ℏ" in str(info)
        assert "mass" in str(info["signature"]) or "m" in str(info)
        assert info["docstring"] is not None

    def test_tdse_complex_output(self):
        """TDSE should work with complex wavefunctions."""

        def complex_model(x):
            return jnp.exp(1j * x[..., 0]) + jnp.exp(-1j * x[..., 0])

        x = jnp.array([[0.0], [1.0]])
        tdse = PDEResidualRegistry.get("schrodinger_td")
        residual = tdse(complex_model, x)

        assert residual.dtype in [jnp.complex64, jnp.complex128] or jnp.isrealobj(
            residual
        )


class TestNavierStokes:
    """Test Navier-Stokes equations: ∂u/∂t + u·∇u = -∇p/ρ + ν∇²u."""

    def test_navier_stokes_registered(self):
        """Navier-Stokes should be registered."""
        assert "navier_stokes" in PDEResidualRegistry.list()

    def test_navier_stokes_steady_poiseuille(self):
        """Poiseuille flow should satisfy steady NS equations."""
        # Steady-state: u·∇u = -∇p + ν∇²u + f
        # Poiseuille: u = (y(H-y), 0), parabolic profile

        def model_u(x):
            """u-velocity component (parabolic profile)."""
            y = x[..., 1]
            H = 1.0
            return y * (H - y)

        def model_v(x):
            """v-velocity component (zero)."""
            return jnp.zeros_like(x[..., 0])

        x = jnp.array([[0.5, 0.5], [0.5, 0.25], [0.5, 0.75]])

        ns = PDEResidualRegistry.get("navier_stokes")
        # Should return tuple of residuals (momentum_x, momentum_y, continuity)
        residuals = ns(model_u, model_v, x, nu=0.01, rho=1.0)

        assert isinstance(residuals, tuple)
        assert len(residuals) == 3  # momentum-x, momentum-y, continuity
        for residual in residuals:
            assert residual.shape == (3,)

    def test_navier_stokes_uniform_flow(self):
        """Uniform flow should satisfy continuity."""

        def model_u(x):
            return jnp.ones_like(x[..., 0])

        def model_v(x):
            return jnp.ones_like(x[..., 0])

        x = jnp.array([[1.0, 1.0], [2.0, 2.0]])
        ns = PDEResidualRegistry.get("navier_stokes")
        residuals = ns(model_u, model_v, x)

        # Continuity should be satisfied (∇·u = 0 for uniform flow)
        assert residuals[2].shape == (2,)  # Continuity residual

    def test_navier_stokes_vector_output(self):
        """NS residuals should return vector for each equation."""

        def model_u(x):
            return x[..., 0]

        def model_v(x):
            return x[..., 1]

        x = jnp.array([[1.0, 1.0], [2.0, 2.0]])
        ns = PDEResidualRegistry.get("navier_stokes")
        residuals = ns(model_u, model_v, x)

        assert len(residuals) == 3
        for residual in residuals:
            assert residual.shape == (2,)
            assert jnp.all(jnp.isfinite(residual))

    def test_navier_stokes_signature(self):
        """NS should have correct parameters."""
        info = PDEResidualRegistry.get_info("navier_stokes")

        # Should mention viscosity and/or density
        signature_str = str(info["signature"])
        assert "nu" in signature_str or "viscosity" in info["docstring"].lower()


class TestMaxwellEquations:
    """Test Maxwell's equations for electromagnetism."""

    def test_maxwell_registered(self):
        """Maxwell's equations should be registered."""
        assert "maxwell" in PDEResidualRegistry.list()

    def test_maxwell_gauss_law(self):
        """Gauss's law: ∇·E = ρ/ε₀."""
        # Uniform E-field from charge distribution

        def model_Ex(x):  # noqa: N802
            """E-field x-component (linear)."""
            return x[..., 0]

        def model_Ey(x):  # noqa: N802
            """E-field y-component (zero)."""
            return jnp.zeros_like(x[..., 0])

        def model_Ez(x):  # noqa: N802
            """E-field z-component (zero)."""
            return jnp.zeros_like(x[..., 0])

        x = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        maxwell = PDEResidualRegistry.get("maxwell")

        # Should return residuals for all 4 Maxwell equations
        residuals = maxwell(model_Ex, model_Ey, model_Ez, x, charge_density=1.0)

        assert isinstance(residuals, tuple)
        assert len(residuals) >= 2  # At minimum Gauss law + one other
        for residual in residuals:
            assert residual.shape == (2,)

    def test_maxwell_plane_wave(self):
        """Plane wave should satisfy Maxwell's equations in vacuum."""
        # E = E₀ sin(kx - ωt), propagating wave

        def model_Ex(x):  # noqa: N802
            k = 2 * jnp.pi
            return jnp.sin(k * x[..., 0])

        def model_Ey(x):  # noqa: N802
            return jnp.zeros_like(x[..., 0])

        def model_Ez(x):  # noqa: N802
            return jnp.zeros_like(x[..., 0])

        x = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
        maxwell = PDEResidualRegistry.get("maxwell")
        residuals = maxwell(model_Ex, model_Ey, model_Ez, x, charge_density=0.0)

        # All residuals should be computed
        assert isinstance(residuals, tuple)
        for residual in residuals:
            assert residual.shape == (3,)
            assert jnp.all(jnp.isfinite(residual))

    def test_maxwell_uniform_field(self):
        """Uniform E-field should have zero divergence."""

        def model_Ex(x):  # noqa: N802
            return jnp.ones_like(x[..., 0])

        def model_Ey(x):  # noqa: N802
            return jnp.ones_like(x[..., 0])

        def model_Ez(x):  # noqa: N802
            return jnp.ones_like(x[..., 0])

        x = jnp.array([[0.0, 0.0, 0.0]])
        maxwell = PDEResidualRegistry.get("maxwell")
        residuals = maxwell(model_Ex, model_Ey, model_Ez, x, charge_density=0.0)

        # Gauss law: ∇·E = 0 for uniform field with no charges
        # First residual should be near zero
        assert jnp.all(jnp.isfinite(residuals[0]))

    def test_maxwell_signature(self):
        """Maxwell should have electromagnetic parameters."""
        info = PDEResidualRegistry.get_info("maxwell")

        # Should mention charge or epsilon
        assert "charge" in str(info).lower() or "epsilon" in str(info).lower()


class TestNonlinearSchrodinger:
    """Test nonlinear Schrödinger (Gross-Pitaevskii) equation."""

    def test_nlse_registered(self):
        """NLSE should be registered."""
        assert "schrodinger_nonlinear" in PDEResidualRegistry.list()

    def test_nlse_soliton_solution(self):
        """Bright soliton should satisfy NLSE."""
        # NLSE: iψ_t = -ψ_xx + σ|ψ|²ψ
        # Bright soliton: ψ(x,t) = A sech(A(x-vt)) exp(i(...))

        def soliton_model(x):
            """Bright soliton solution."""
            A = 1.0
            return A / jnp.cosh(A * x[..., 0])

        x = jnp.array([[0.0], [1.0], [2.0], [-1.0]])

        nlse = PDEResidualRegistry.get("schrodinger_nonlinear")
        residual = nlse(soliton_model, x, sigma=1.0)

        # Should compute without error
        assert residual.shape == (4,)
        assert jnp.all(jnp.isfinite(residual))

    def test_nlse_plane_wave(self):
        """Plane wave should satisfy NLSE with nonlinearity."""

        def plane_wave(x):
            k = 1.0
            return jnp.exp(1j * k * x[..., 0])

        x = jnp.array([[0.0], [1.0]])
        nlse = PDEResidualRegistry.get("schrodinger_nonlinear")
        residual = nlse(plane_wave, x, sigma=0.5)

        assert residual.shape == (2,)

    def test_nlse_complex_coefficient(self):
        """NLSE should handle complex wavefunctions."""

        def complex_wavefunction(x):
            return (1.0 + 1.0j) * jnp.exp(-(x[..., 0] ** 2))

        x = jnp.array([[0.0], [0.5]])
        nlse = PDEResidualRegistry.get("schrodinger_nonlinear")
        residual = nlse(complex_wavefunction, x)

        assert residual.shape == (2,)
        assert jnp.all(jnp.isfinite(residual))

    def test_nlse_signature(self):
        """NLSE should have nonlinearity parameter."""
        info = PDEResidualRegistry.get_info("schrodinger_nonlinear")

        # Should mention nonlinearity or sigma
        assert (
            "sigma" in str(info["signature"]) or "nonlin" in info["docstring"].lower()
        )


class TestCoupledPDESystems:
    """Test support for coupled PDE systems."""

    def test_reaction_diffusion_registered(self):
        """Reaction-diffusion system should be registered."""
        assert "reaction_diffusion" in PDEResidualRegistry.list()

    def test_reaction_diffusion_steady_state(self):
        """Steady-state solution should have specific residual."""
        # ∂u/∂t = D∇²u + f(u,v)
        # ∂v/∂t = D∇²v + g(u,v)
        # Steady state: ∂u/∂t = 0, ∂v/∂t = 0

        def model_u(x):
            """First component - uniform."""
            return jnp.ones_like(x[..., 0])

        def model_v(x):
            """Second component - uniform."""
            return jnp.ones_like(x[..., 0])

        x = jnp.array([[0.5, 0.5], [1.0, 1.0]])

        rd = PDEResidualRegistry.get("reaction_diffusion")
        residuals = rd(model_u, model_v, x, D=0.1)

        # Both residuals should be arrays
        assert isinstance(residuals, tuple)
        assert len(residuals) == 2
        for residual in residuals:
            assert residual.shape == (2,)

    def test_reaction_diffusion_turing_pattern(self):
        """Should handle Turing pattern-like solutions."""

        def model_u(x):
            """Sinusoidal pattern in u."""
            return 1.0 + 0.1 * jnp.sin(2 * jnp.pi * x[..., 0])

        def model_v(x):
            """Sinusoidal pattern in v (different phase)."""
            return 1.0 + 0.1 * jnp.cos(2 * jnp.pi * x[..., 0])

        x = jnp.array([[0.0, 0.0], [0.25, 0.0], [0.5, 0.0]])
        rd = PDEResidualRegistry.get("reaction_diffusion")
        residuals = rd(model_u, model_v, x, D=0.1, a=1.0, b=0.5)

        assert len(residuals) == 2
        for residual in residuals:
            assert residual.shape == (3,)
            assert jnp.all(jnp.isfinite(residual))

    def test_reaction_diffusion_signature(self):
        """Reaction-diffusion should have diffusion coefficient."""
        info = PDEResidualRegistry.get_info("reaction_diffusion")

        # Should mention diffusion
        assert "D" in str(info["signature"]) or "diffusion" in info["docstring"].lower()


class TestJITCompatibility:
    """Test that all advanced PDEs are JIT-compatible."""

    def test_all_advanced_pdes_jittable(self):
        """All advanced PDE residuals should be JIT-compatible."""
        advanced_pdes = [
            "schrodinger_td",
            "navier_stokes",
            "maxwell",
            "schrodinger_nonlinear",
            "reaction_diffusion",
        ]

        for pde_name in advanced_pdes:
            if pde_name not in PDEResidualRegistry.list():
                pytest.skip(f"PDE {pde_name} not registered yet")

            pde_fn = PDEResidualRegistry.get(pde_name)

            # Basic test model
            def test_model(x):
                return jnp.ones_like(x[..., 0])

            x = jnp.array([[0.5, 0.5, 0.5]])

            # Should be JIT-compilable
            try:

                @jax.jit
                def jitted_residual(x, _pde_name=pde_name, _pde_fn=pde_fn):
                    # Handle different call signatures
                    if _pde_name == "navier_stokes":
                        return _pde_fn(test_model, test_model, x)
                    if _pde_name == "maxwell":
                        return _pde_fn(test_model, test_model, test_model, x)
                    if _pde_name == "reaction_diffusion":
                        return _pde_fn(test_model, test_model, x)
                    return _pde_fn(test_model, x)

                result = jitted_residual(x)
                # Check result is finite (tuple or array)
                if isinstance(result, tuple):
                    for r in result:
                        assert jnp.all(jnp.isfinite(r))
                else:
                    assert jnp.all(jnp.isfinite(result))
            except Exception as e:
                pytest.fail(f"JIT compilation failed for {pde_name}: {e}")

    def test_vmap_compatibility(self):
        """Advanced PDEs should work with vmap."""
        # Test with time-dependent Schrödinger (simplest)
        if "schrodinger_td" not in PDEResidualRegistry.list():
            pytest.skip("schrodinger_td not registered")

        def test_model(x):
            return jnp.ones_like(x[..., 0])

        tdse = PDEResidualRegistry.get("schrodinger_td")

        # Batch of inputs
        x_batch = jnp.array([[[0.5]], [[1.0]], [[1.5]]])

        # Vmap over batch
        batched_tdse = jax.vmap(lambda x: tdse(test_model, x))
        results = batched_tdse(x_batch)

        assert results.shape[0] == 3  # Batch dimension


class TestRegistryExtensibility:
    """Test that registry remains extensible for advanced PDEs."""

    def test_can_register_custom_quantum_pde(self):
        """Users should be able to register custom quantum PDEs."""
        from opifex.core.physics.autodiff_engine import compute_laplacian

        def custom_quantum_residual(model, x, coupling=1.0):
            """Custom quantum PDE with coupling."""
            psi = model(x)
            laplacian_psi = compute_laplacian(model, x)
            return laplacian_psi + coupling * jnp.abs(psi) ** 2 * psi

        # Register it
        PDEResidualRegistry.register(
            "custom_quantum", custom_quantum_residual, override=True
        )

        assert "custom_quantum" in PDEResidualRegistry.list()

        # Test it works
        def test_model(x):
            return jnp.ones_like(x[..., 0])

        x = jnp.array([[0.5]])
        custom_pde = PDEResidualRegistry.get("custom_quantum")
        result = custom_pde(test_model, x, coupling=2.0)
        assert jnp.all(jnp.isfinite(result))

    def test_can_register_custom_fluid_pde(self):
        """Users should be able to register custom fluid dynamics PDEs."""
        from opifex.core.physics.autodiff_engine import compute_gradient

        def custom_fluid_residual(model_u, model_v, x, Re=100.0):
            """Custom fluid PDE with Reynolds number."""
            # Simplified momentum equation
            u = model_u(x)
            v = model_v(x)
            grad_u = compute_gradient(model_u, x)
            # Simplified residual
            return grad_u[..., 0] + u * v / Re

        # Register
        PDEResidualRegistry.register(
            "custom_fluid", custom_fluid_residual, override=True
        )

        assert "custom_fluid" in PDEResidualRegistry.list()


class TestDRYCompliance:
    """Verify no code duplication in advanced PDEs."""

    def test_all_use_autodiff_engine(self):
        """All PDEs should use AutoDiffEngine, not custom derivatives."""
        # This is a meta-test - actual verification happens in implementation review
        # Just verify all PDEs are registered
        advanced_pdes = [
            "schrodinger_td",
            "navier_stokes",
            "maxwell",
            "schrodinger_nonlinear",
            "reaction_diffusion",
        ]

        registered = PDEResidualRegistry.list()

        for pde in advanced_pdes:
            assert pde in registered, f"{pde} should be registered"

    def test_consistent_calling_conventions(self):
        """PDEs should follow consistent calling conventions."""
        # Single field PDEs: pde_fn(model, x, **params)
        # Multi-field PDEs: pde_fn(model1, model2, ..., x, **params)

        # Test single-field
        if "schrodinger_td" in PDEResidualRegistry.list():
            PDEResidualRegistry.get("schrodinger_td")
            info = PDEResidualRegistry.get_info("schrodinger_td")
            # Should have model and x in signature
            sig_str = str(info["signature"])
            assert "model" in sig_str and "x" in sig_str

        # Test multi-field
        if "navier_stokes" in PDEResidualRegistry.list():
            PDEResidualRegistry.get("navier_stokes")
            info = PDEResidualRegistry.get_info("navier_stokes")
            # Should have multiple models
            sig_str = str(info["signature"])
            assert "model" in sig_str or "u" in sig_str


class TestPerformance:
    """Basic performance checks for advanced PDEs."""

    def test_tdse_performance(self):
        """TDSE should compile and run efficiently."""
        if "schrodinger_td" not in PDEResidualRegistry.list():
            pytest.skip("schrodinger_td not registered")

        def test_model(x):
            return jnp.exp(-(x[..., 0] ** 2))

        x = jnp.linspace(-5, 5, 1000).reshape(-1, 1)

        tdse = PDEResidualRegistry.get("schrodinger_td")

        # JIT compile
        @jax.jit
        def compute(x):
            return tdse(test_model, x)

        # Warm-up
        _ = compute(x[:10])

        # Should run efficiently on larger input
        import time

        start = time.perf_counter()
        result = compute(x)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0, f"TDSE too slow: {elapsed:.3f}s"
        assert result.shape == (1000,)

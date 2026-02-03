#!/usr/bin/env python3
"""
Verify JAX JIT compatibility for all PDE residuals in the registry.

This script tests that all 13 registered PDEs can be successfully JIT-compiled
without runtime errors due to Python control flow or try/except blocks.

Usage:
    python scripts/verify_jit_compatibility.py
"""

import sys

import jax
import jax.numpy as jnp

from opifex.core.physics.autodiff_engine import AutoDiffEngine
from opifex.core.physics.pde_registry import PDEResidualRegistry


def create_dummy_model(input_dim: int = 2, output_dim: int = 1):
    """Create a simple MLP model for testing.

    Returns (batch,) for scalar output (matches autodiff_engine expectations).
    """

    def model(x):
        # Simple quadratic model: sum of squares
        # Shape: (batch, dim) -> (batch,)
        return jnp.sum(x**2, axis=-1)

    return model


def test_heat_equation():
    """Test Heat equation JIT compatibility."""
    model = create_dummy_model()
    x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    heat_fn = PDEResidualRegistry.get("heat")

    @jax.jit
    def jit_residual(x):
        return heat_fn(model, x, AutoDiffEngine)

    result = jit_residual(x)
    return result.shape == (2,)


def test_wave_equation():
    """Test Wave equation JIT compatibility."""
    model = create_dummy_model()
    x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    wave_fn = PDEResidualRegistry.get("wave")

    @jax.jit
    def jit_residual(x):
        return wave_fn(model, x, AutoDiffEngine, wave_speed=1.0)

    result = jit_residual(x)
    return result.shape == (2,)


def test_poisson_equation():
    """Test Poisson equation JIT compatibility."""
    model = create_dummy_model()
    x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    source = jnp.array([0.0, 0.0])
    poisson_fn = PDEResidualRegistry.get("poisson")

    @jax.jit
    def jit_residual(x, source):
        return poisson_fn(model, x, AutoDiffEngine, source_term=source)

    result = jit_residual(x, source)
    return result.shape == (2,)


def test_burgers_equation():
    """Test Burgers equation JIT compatibility."""
    model = create_dummy_model()
    x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    burgers_fn = PDEResidualRegistry.get("burgers")

    @jax.jit
    def jit_residual(x):
        return burgers_fn(model, x, AutoDiffEngine, nu=0.01)

    result = jit_residual(x)
    return result.shape == (2,)


def test_schrodinger():
    """Test Schrödinger equation JIT compatibility."""
    model = create_dummy_model()
    x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    schrodinger_fn = PDEResidualRegistry.get("schrodinger")

    @jax.jit
    def jit_residual(x):
        return schrodinger_fn(model, x, AutoDiffEngine)

    result = jit_residual(x)
    return result.shape == (2,)


def test_schrodinger_td():
    """Test Time-dependent Schrödinger equation JIT compatibility."""
    model = create_dummy_model()
    x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    schrodinger_td_fn = PDEResidualRegistry.get("schrodinger_td")

    @jax.jit
    def jit_residual(x):
        return schrodinger_td_fn(model, x)

    result = jit_residual(x)
    return result.shape == (2,)


def test_navier_stokes():
    """Test Navier-Stokes equation JIT compatibility."""
    model_u = create_dummy_model()
    model_v = create_dummy_model()
    x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    ns_fn = PDEResidualRegistry.get("navier_stokes")

    @jax.jit
    def jit_residual(x):
        return ns_fn(model_u, model_v, x, nu=0.01)

    result = jit_residual(x)
    # Returns tuple of (momentum_x, momentum_y, continuity)
    return len(result) == 3 and all(r.shape == (2,) for r in result)


def test_maxwell():
    """Test Maxwell's equations JIT compatibility."""
    model_Ex = create_dummy_model(input_dim=3)
    model_Ey = create_dummy_model(input_dim=3)
    model_Ez = create_dummy_model(input_dim=3)
    x = jnp.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    maxwell_fn = PDEResidualRegistry.get("maxwell")

    @jax.jit
    def jit_residual(x):
        return maxwell_fn(model_Ex, model_Ey, model_Ez, x)

    result = jit_residual(x)
    # Returns tuple of (gauss_law, curl_E)
    return len(result) == 2 and all(r.shape == (2,) for r in result)


def test_schrodinger_nonlinear():
    """Test Nonlinear Schrödinger (Gross-Pitaevskii) JIT compatibility."""
    model = create_dummy_model()
    x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    nl_fn = PDEResidualRegistry.get("schrodinger_nonlinear")

    @jax.jit
    def jit_residual(x):
        return nl_fn(model, x, sigma=1.0)

    result = jit_residual(x)
    return result.shape == (2,)


def test_reaction_diffusion():
    """Test Reaction-Diffusion system JIT compatibility."""
    model_u = create_dummy_model()
    model_v = create_dummy_model()
    x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    rd_fn = PDEResidualRegistry.get("reaction_diffusion")

    @jax.jit
    def jit_residual(x):
        return rd_fn(model_u, model_v, x)

    result = jit_residual(x)
    # Returns tuple of (residual_u, residual_v)
    return len(result) == 2 and all(r.shape == (2,) for r in result)


def test_homogenization():
    """Test Homogenization PDE JIT compatibility."""
    model = create_dummy_model()
    x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    homog_fn = PDEResidualRegistry.get("homogenization")

    @jax.jit
    def jit_residual(x):
        return homog_fn(model, x, AutoDiffEngine)

    result = jit_residual(x)
    return result.shape == (2,)


def test_two_scale():
    """Test Two-scale expansion PDE JIT compatibility."""
    model_macro = create_dummy_model()
    model_micro = create_dummy_model()
    x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    two_scale_fn = PDEResidualRegistry.get("two_scale")

    @jax.jit
    def jit_residual(x):
        return two_scale_fn(model_macro, model_micro, x, AutoDiffEngine, epsilon=0.1)

    result = jit_residual(x)
    # Returns tuple of (macro_residual, micro_residual)
    return len(result) == 2 and all(r.shape == (2,) for r in result)


def test_amr_poisson():
    """Test AMR Poisson PDE JIT compatibility."""
    model = create_dummy_model()
    x = jnp.array([[0.5, 0.5], [1.0, 1.0]])
    amr_fn = PDEResidualRegistry.get("amr_poisson")

    @jax.jit
    def jit_residual(x):
        return amr_fn(model, x, AutoDiffEngine)

    result = jit_residual(x)
    # Returns tuple of (residual, error_indicator)
    return len(result) == 2 and result[0].shape == (2,) and result[1].shape == (2,)


def main():
    """Run all JIT compatibility tests."""
    print("=" * 70)
    print("JAX JIT Compatibility Verification for PDE Registry")
    print("=" * 70)
    print()

    tests = [
        ("Heat Equation", test_heat_equation),
        ("Wave Equation", test_wave_equation),
        ("Poisson Equation", test_poisson_equation),
        ("Burgers Equation", test_burgers_equation),
        ("Schrödinger Equation", test_schrodinger),
        ("Time-Dependent Schrödinger", test_schrodinger_td),
        ("Navier-Stokes", test_navier_stokes),
        ("Maxwell's Equations", test_maxwell),
        ("Nonlinear Schrödinger", test_schrodinger_nonlinear),
        ("Reaction-Diffusion", test_reaction_diffusion),
        ("Homogenization", test_homogenization),
        ("Two-Scale Expansion", test_two_scale),
        ("AMR Poisson", test_amr_poisson),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            success = test_fn()
            if success:
                print(f"✅ {name:<35} PASSED")
                passed += 1
            else:
                print(f"❌ {name:<35} FAILED (output shape mismatch)")
                failed += 1
                errors.append((name, "Output shape mismatch"))
        except Exception as e:
            print(f"❌ {name:<35} FAILED (exception)")
            failed += 1
            errors.append((name, str(e)))

    print()
    print("=" * 70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 70)

    if failed > 0:
        print()
        print("FAILURES:")
        for name, error in errors:
            print(f"  {name}:")
            print(f"    {error}")
        print()
        return 1
    print()
    print("🎉 All PDEs are JIT-compatible!")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

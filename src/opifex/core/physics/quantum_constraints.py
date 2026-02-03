"""Quantum constraint functions for physics-informed neural networks.

This module provides constraint functions specific to quantum mechanical systems:
- Wavefunction normalization: ∫|ψ|²dx = 1
- Density positivity: ρ(x) ≥ 0
- Hermiticity verification: H = H†
- Probability conservation: ||ψ(t)||² = constant

All functions are JAX-compatible (JIT, vmap, grad) and designed for use in
physics-informed training loops.

Design Philosophy:
- Pure functions (no side effects)
- JIT-compiled for performance
- DRY: No duplication with other modules
- Simple, clear APIs
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float


def wavefunction_normalization(
    psi: Complex[Array, "..."] | Float[Array, "..."],
    dx: float | Float[Array, ""],
    tolerance: float = 1e-6,
) -> Float[Array, ""]:
    """Compute wavefunction normalization violation.

    Checks if ∫|ψ|²dx = 1 (normalized wavefunction constraint).
    Returns the absolute deviation from unit norm.

    Args:
        psi: Wavefunction array (can be complex or real)
        dx: Spatial step size for integration
        tolerance: Violations below this are set to zero

    Returns:
        Violation magnitude: |∫|ψ|²dx - 1|

    Examples:
        >>> psi = jnp.array([1.0, 0.0, 0.0])
        >>> dx = 1.0
        >>> wavefunction_normalization(psi, dx)
        Array(0., dtype=float32)

        >>> psi = jnp.ones(10) * 2.0  # Unnormalized
        >>> dx = 0.1
        >>> violation = wavefunction_normalization(psi, dx)
        >>> violation > 0  # Should detect violation
        Array(True, dtype=bool)
    """
    # Compute norm: ∫|ψ|²dx
    norm_squared = jnp.sum(jnp.abs(psi) ** 2) * dx

    # Violation is deviation from 1
    violation = jnp.abs(norm_squared - 1.0)

    # Filter small violations within tolerance
    return jnp.where(violation < tolerance, 0.0, violation)


def density_positivity_violation(
    rho: Float[Array, "..."],
    tolerance: float = 1e-8,
) -> Float[Array, ""]:
    """Compute density positivity constraint violation.

    Physical densities must be non-negative: ρ(x) ≥ 0.
    Returns the sum of negative density values (how much negativity exists).

    Args:
        rho: Density array
        tolerance: Negative values within tolerance are ignored

    Returns:
        Sum of negative parts: Σ max(0, -ρ)

    Examples:
        >>> rho = jnp.array([1.0, 2.0, 0.5])
        >>> density_positivity_violation(rho)
        Array(0., dtype=float32)

        >>> rho = jnp.array([1.0, -0.5, 2.0, -1.0])
        >>> violation = density_positivity_violation(rho)
        >>> violation  # Should be 0.5 + 1.0 = 1.5
        Array(1.5, dtype=float32)
    """
    # Identify negative values
    negative_parts = jnp.where(rho < -tolerance, -rho, 0.0)

    # Sum up all negativity
    return jnp.sum(negative_parts)


def hermiticity_violation(
    H: Complex[Array, "N N"] | Float[Array, "N N"],
    tolerance: float = 1e-8,
) -> Float[Array, ""]:
    """Verify operator Hermiticity: H = H†.

    Physical observables must be Hermitian operators.
    Returns the Frobenius norm of the deviation from Hermiticity.

    Args:
        H: Operator matrix
        tolerance: Deviations below this are set to zero

    Returns:
        ||H - H†||_F (Frobenius norm of anti-Hermitian part)

    Examples:
        >>> H = jnp.array([[1.0, 2.0], [2.0, 3.0]])  # Symmetric (Hermitian)
        >>> hermiticity_violation(H)
        Array(0., dtype=float32)

        >>> H = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # Not Hermitian
        >>> violation = hermiticity_violation(H)
        >>> violation > 0
        Array(True, dtype=bool)
    """
    # Compute H†(conjugate transpose)
    H_dagger = jnp.conj(H.T)

    # Deviation from Hermiticity
    deviation = H - H_dagger

    # Frobenius norm: sqrt(Σ|H_ij - H†_ij|²)
    violation = jnp.linalg.norm(deviation, ord="fro")

    # Filter small violations
    return jnp.asarray(jnp.where(violation < tolerance, 0.0, violation))


def probability_conservation(
    psi_t0: Complex[Array, "..."] | Float[Array, "..."],
    psi_t1: Complex[Array, "..."] | Float[Array, "..."],
    tolerance: float = 1e-6,
) -> Float[Array, ""]:
    """Check probability conservation in time evolution.

    Total probability must be conserved: ||ψ(t₁)||² = ||ψ(t₀)||².
    Returns the absolute difference in total probability.

    Args:
        psi_t0: Wavefunction at time t₀
        psi_t1: Wavefunction at time t₁
        tolerance: Violations below this are set to zero

    Returns:
        | ||ψ(t₁)||² - ||ψ(t₀)||² |

    Examples:
        >>> psi_t0 = jnp.array([1.0, 0.0])
        >>> psi_t1 = jnp.array([0.0, 1.0])  # Rotated, same norm
        >>> probability_conservation(psi_t0, psi_t1)
        Array(0., dtype=float32)

        >>> psi_t0 = jnp.array([1.0, 0.0])
        >>> psi_t1 = jnp.array([0.5, 0.0])  # Lost probability
        >>> violation = probability_conservation(psi_t0, psi_t1)
        >>> violation > 0
        Array(True, dtype=bool)
    """
    # Compute norms at both times
    norm_t0 = jnp.sum(jnp.abs(psi_t0) ** 2)
    norm_t1 = jnp.sum(jnp.abs(psi_t1) ** 2)

    # Probability conservation violation
    violation = jnp.abs(norm_t1 - norm_t0)

    # Filter small violations
    return jnp.where(violation < tolerance, 0.0, violation)


# JIT-compile all functions for performance
wavefunction_normalization = jax.jit(wavefunction_normalization)
density_positivity_violation = jax.jit(density_positivity_violation)
hermiticity_violation = jax.jit(hermiticity_violation)
probability_conservation = jax.jit(probability_conservation)


# Public API
__all__ = [
    "density_positivity_violation",
    "hermiticity_violation",
    "probability_conservation",
    "wavefunction_normalization",
]

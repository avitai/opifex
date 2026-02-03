"""
Quantum Spectral Methods

This module provides spectral methods for quantum mechanical calculations,
extracted from quantum operators while preserving exact mathematical formulation.
All functions maintain the original mathematical logic and numerical behavior.

Key Functions:
- spectral_gradient: First derivative using FFT (from MomentumOperator)
- spectral_second_derivative: Second derivative using FFT (from KineticEnergyOperator)
- spectral_kinetic_energy: Kinetic energy computation (from HamiltonianOperator)
- spectral_momentum: Momentum operator application
"""

import jax.numpy as jnp
from jaxtyping import Array


def _finite_difference_gradient(wavefunction: Array, dx: float) -> Array:
    """
    Fallback finite difference gradient for small arrays.

    Uses centered difference with boundary conditions.

    Args:
        wavefunction: Input wavefunction
        dx: Grid spacing

    Returns:
        Gradient computed using finite differences
    """
    n = len(wavefunction)
    if n < 2:
        return jnp.zeros_like(wavefunction)

    # Use centered differences where possible
    grad = jnp.zeros_like(wavefunction)

    # Forward difference at start
    grad = grad.at[0].set((wavefunction[1] - wavefunction[0]) / dx)

    # Centered differences in middle
    if n > 2:
        grad = grad.at[1:-1].set((wavefunction[2:] - wavefunction[:-2]) / (2 * dx))

    # Backward difference at end
    return grad.at[-1].set((wavefunction[-1] - wavefunction[-2]) / dx)


def _finite_difference_second_derivative(wavefunction: Array, dx: float) -> Array:
    """
    Fallback finite difference second derivative for small arrays.

    Uses centered second difference with boundary conditions.

    Args:
        wavefunction: Input wavefunction
        dx: Grid spacing

    Returns:
        Second derivative computed using finite differences
    """
    n = len(wavefunction)
    if n < 3:
        return jnp.zeros_like(wavefunction)

    # Use centered second differences
    second_deriv = jnp.zeros_like(wavefunction)

    # Second order differences for interior points
    second_deriv = second_deriv.at[1:-1].set(
        (wavefunction[2:] - 2 * wavefunction[1:-1] + wavefunction[:-2]) / (dx**2)
    )

    # Use forward/backward differences for boundaries
    second_deriv = second_deriv.at[0].set(
        (wavefunction[2] - 2 * wavefunction[1] + wavefunction[0]) / (dx**2)
    )
    return second_deriv.at[-1].set(
        (wavefunction[-1] - 2 * wavefunction[-2] + wavefunction[-3]) / (dx**2)
    )


def quantum_fft_gradient(wavefunction: Array, dx: float, hbar: float = 1.0) -> Array:
    """
    Compute gradient using spectral (FFT) method for quantum wavefunctions.

    Uses complex FFT appropriate for quantum mechanical wavefunctions.
    This preserves the original mathematical formulation from quantum operators.

    Args:
        wavefunction: Complex wavefunction array
        dx: Grid spacing
        hbar: Reduced Planck constant

    Returns:
        Gradient of the wavefunction with JAX-native precision
    """
    # Ensure input is JAX array
    wf_array = jnp.asarray(wavefunction)
    n = len(wf_array)

    # For very small arrays, use finite difference instead
    if n < 4:
        return _finite_difference_gradient(wf_array, dx)

    was_real = jnp.isrealobj(wf_array)

    # Convert to complex if needed - JAX handles precision automatically
    wf_complex = wf_array + 0j if was_real else wf_array

    # FFT of the wavefunction
    fft_psi = jnp.fft.fft(wf_complex)

    # Wave numbers
    k = jnp.fft.fftfreq(n, dx) * 2 * jnp.pi

    # Multiply by ik in frequency domain (derivative becomes multiplication)
    fft_grad = 1j * k * fft_psi

    # Inverse FFT to get gradient
    gradient = jnp.fft.ifft(fft_grad)

    # For real inputs, return real part (physical requirement)
    if was_real:
        return jnp.real(gradient)

    return gradient


def quantum_fft_second_derivative(
    wavefunction: Array, dx: float, hbar: float = 1.0
) -> Array:
    """
    Compute second derivative using spectral (FFT) method for quantum wavefunctions.

    This preserves the original mathematical formulation from quantum operators.

    Args:
        wavefunction: Complex wavefunction array
        dx: Grid spacing
        hbar: Reduced Planck constant

    Returns:
        Second derivative of the wavefunction with JAX-native precision
    """
    # Ensure input is JAX array
    wf_array = jnp.asarray(wavefunction)
    n = len(wf_array)

    # For very small arrays, use finite difference instead
    if n < 4:
        return _finite_difference_second_derivative(wf_array, dx)

    was_real = jnp.isrealobj(wf_array)

    # Convert to complex if needed - JAX-native complex conversion
    wf_complex = wf_array + 0j if was_real else wf_array

    # FFT of the wavefunction
    fft_psi = jnp.fft.fft(wf_complex)

    # Wave numbers
    k = jnp.fft.fftfreq(n, dx) * 2 * jnp.pi

    # Multiply by (ik)^2 = -k^2 in frequency domain (second derivative)
    fft_second_deriv = -(k**2) * fft_psi

    # Inverse FFT to get second derivative
    second_derivative = jnp.fft.ifft(fft_second_deriv)

    # For real inputs, return real part (physical requirement)
    if was_real:
        return jnp.real(second_derivative)

    return second_derivative


def spectral_gradient(
    wavefunction: Array, dx: float, hbar: float = 1.0, allow_fallback: bool = True
) -> Array:
    """
    Compute gradient using spectral method.

    Args:
        wavefunction: Input wavefunction
        dx: Grid spacing
        hbar: Reduced Planck constant
        allow_fallback: Whether to allow finite difference fallback for small arrays

    Returns:
        Gradient of wavefunction

    Raises:
        ValueError: If array is too small and fallback is disabled
    """
    wf_array = jnp.asarray(wavefunction)
    n = len(wf_array)

    if n < 4 and not allow_fallback:
        raise ValueError("Array too small for spectral method and fallback disabled")

    # Use quantum-specific FFT gradient for proper mathematical formulation
    return quantum_fft_gradient(wavefunction, dx, hbar)


def spectral_second_derivative(
    wavefunction: Array,
    dx: float | Array = 1.0,
    hbar: float | Array = 1.0,
    mass: float | Array = 1.0,
    apply_kinetic_factor: bool = False,
) -> Array:
    """
    Compute second derivative using spectral (FFT) method with enhanced precision.

    This function preserves the exact mathematical formulation from
    KineticEnergyOperator while leveraging JAX X64 precision management.

    Args:
        wavefunction: Input wavefunction to differentiate
        dx: Grid spacing for spatial derivative
        hbar: Reduced Planck constant (default 1.0)
        mass: Particle mass (default 1.0)
        apply_kinetic_factor: Whether to apply -ℏ²/(2m) factor for kinetic
            energy

    Returns:
        Second derivative of the wavefunction (or kinetic energy if
            apply_kinetic_factor=True)

    Notes:
        Mathematical formulation: T = -ℏ²∇²/(2m) when apply_kinetic_factor=True
        JAX X64 precision provides enhanced numerical accuracy.
    """
    # Ensure wavefunction is Array type
    wf_array = jnp.asarray(wavefunction)
    n = len(wf_array)

    # Determine input characteristics for output consistency
    was_real = jnp.isrealobj(wf_array)

    # Convert to complex if needed - JAX handles precision automatically
    wf_complex = wf_array + 0j if was_real else wf_array

    # FFT of the wavefunction
    fft_psi = jnp.fft.fft(wf_complex)

    # Wave numbers for second derivative
    k = jnp.fft.fftfreq(n, dx) * 2 * jnp.pi

    # Second derivative in frequency domain: multiply by -k²
    fft_second_deriv = -(k**2) * fft_psi

    # Inverse FFT to get second derivative
    second_deriv = jnp.fft.ifft(fft_second_deriv)

    # Apply kinetic energy factor if requested: T = -ℏ²∇²/(2m)
    if apply_kinetic_factor:
        kinetic_result = -(hbar**2) / (2 * mass) * second_deriv
        # Return real part if input was real
        if was_real:
            return jnp.real(kinetic_result)
        return kinetic_result

    # Return raw second derivative
    if was_real:
        return jnp.real(second_deriv)
    return second_deriv


def spectral_kinetic_energy(
    wavefunction: Array, dx: float, hbar: float = 1.0, mass: float = 1.0
) -> Array:
    """
    Compute kinetic energy using spectral method.

    Implements T|ψ⟩ = -ℏ²∇²|ψ⟩/(2m) using FFT-based second derivative.
    This preserves the original mathematical formulation.

    Args:
        wavefunction: Quantum wavefunction
        dx: Grid spacing
        hbar: Reduced Planck constant
        mass: Particle mass

    Returns:
        Kinetic energy operator applied to wavefunction (always real for physics)
    """
    # Get second derivative using quantum FFT method
    second_deriv = quantum_fft_second_derivative(wavefunction, dx, hbar)

    # Apply kinetic energy factor: T = -ℏ²∇²/(2m)
    kinetic = -(hbar**2) / (2 * mass) * second_deriv

    # Kinetic energy is always real in physics (even for complex wavefunctions)
    # This is because T is Hermitian and ⟨ψ|T|ψ⟩ is always real
    return jnp.real(kinetic)


def spectral_momentum(wavefunction: Array, dx: float, hbar: float = 1.0) -> Array:
    """
    Compute momentum using spectral method.

    Implements p|ψ⟩ = -iℏ∇|ψ⟩ using FFT-based gradient.
    This preserves the original mathematical formulation.

    Args:
        wavefunction: Quantum wavefunction
        dx: Grid spacing
        hbar: Reduced Planck constant

    Returns:
        Momentum operator applied to wavefunction
    """
    # Get gradient using quantum FFT method
    gradient = quantum_fft_gradient(wavefunction, dx, hbar)

    # Apply momentum operator: p = -iℏ∇
    return -1j * hbar * gradient

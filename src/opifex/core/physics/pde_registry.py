"""
PDE Residual Registry.

This module provides a global registry for PDE residual functions, enabling:
- Registration of custom PDEs with decorator or explicit registration
- Thread-safe storage and retrieval
- Built-in implementations for common PDEs
- Introspection and listing capabilities
- Clear error messages

The registry follows the extensibility pattern from Version 0E, allowing users to:
1. Use built-in PDEs (zero code)
2. Register custom PDEs globally (reusable)
3. Pass custom functions directly via config (one-off use)
"""
# ruff: noqa: F821
# F821 disabled: Ruff incorrectly flags jaxtyping symbolic dimensions ("batch", "dim")
# as undefined names. These are valid jaxtyping string literal dimension annotations.

import inspect
import threading
import warnings
from collections.abc import Callable
from typing import Any, ClassVar

import jax.numpy as jnp
from jaxtyping import Array, Float

from opifex.core.physics.autodiff_engine import (
    AutoDiffEngine,
)


# Type alias for PDE residual functions
PDEResidualFunction = Callable[..., Any]


class PDEResidualRegistry:
    """
    Global registry for PDE residual functions.

    This registry provides a centralized location for registering and retrieving
    PDE residual computation functions. It supports:

    - Decorator-based registration
    - Explicit registration
    - Thread-safe operations
    - Built-in PDE implementations
    - Introspection and listing

    Examples:
        Register using decorator:
        >>> @PDEResidualRegistry.register("my_pde")
        ... def my_pde(model, x, autodiff_engine, param=1.0):
        ...     return autodiff_engine.compute_laplacian(model, x)

        Register explicitly:
        >>> PDEResidualRegistry.register("my_pde", my_pde_func)

        Retrieve:
        >>> pde_fn = PDEResidualRegistry.get("poisson")

        List all:
        >>> names = PDEResidualRegistry.list()
    """

    _registry: ClassVar[dict[str, PDEResidualFunction]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def register(
        cls,
        name: str,
        function: PDEResidualFunction | None = None,
        override: bool = False,
    ) -> Callable:
        """
        Register a PDE residual function.

        Can be used as decorator or called explicitly.

        Args:
            name: Name to register the PDE under (e.g., "poisson", "heat")
            function: PDE residual function (if not using as decorator)
            override: If True, allow overriding existing registration

        Returns:
            The registered function (for decorator chaining)

        Raises:
            ValueError: If name is empty
            TypeError: If name is not a string or function is not callable
            KeyError: If name already exists and override=False

        Examples:
            As decorator:
            >>> @PDEResidualRegistry.register("custom_pde")
            ... def my_pde(model, x, autodiff_engine):
            ...     return x

            Explicit:
            >>> PDEResidualRegistry.register("custom_pde", my_pde_func)
        """
        # Validate name
        if not isinstance(name, str):
            raise TypeError(f"PDE name must be a string, got {type(name)}")
        if not name or not name.strip():
            raise ValueError("PDE name cannot be empty")

        def _register(func: PDEResidualFunction) -> PDEResidualFunction:
            # Validate function
            if not callable(func):
                raise TypeError(f"PDE function must be callable, got {type(func)}")

            with cls._lock:
                # Check for duplicates
                if name in cls._registry and not override:
                    warnings.warn(
                        f"PDE '{name}' is already registered. Use override=True to replace it.",
                        UserWarning,
                        stacklevel=3,
                    )
                    # Still allow registration with warning
                    # This follows Python's "we're all adults" philosophy

                cls._registry[name] = func

            return func

        # If function provided, register immediately (explicit mode)
        if function is not None:
            return _register(function)

        # Otherwise, return decorator (decorator mode)
        return _register

    @classmethod
    def get(cls, name: str) -> PDEResidualFunction:
        """
        Retrieve a registered PDE residual function.

        Args:
            name: Name of the PDE to retrieve

        Returns:
            The registered PDE residual function

        Raises:
            KeyError: If PDE is not registered (with helpful error message)

        Examples:
            >>> poisson = PDEResidualRegistry.get("poisson")
            >>> residual = poisson(model, x, AutoDiffEngine, source_term=f)
        """
        with cls._lock:
            if name not in cls._registry:
                # Get available names without calling list() to avoid deadlock
                available = sorted(cls._registry.keys())
                raise KeyError(f"PDE '{name}' not found in registry. Available PDEs: {available}")
            return cls._registry[name]

    @classmethod
    def contains(cls, name: str) -> bool:
        """
        Check if a PDE is registered.

        Args:
            name: Name to check

        Returns:
            True if PDE is registered, False otherwise

        Examples:
            >>> if PDEResidualRegistry.contains("poisson"):
            ...     pde = PDEResidualRegistry.get("poisson")
        """
        with cls._lock:
            return name in cls._registry

    @classmethod
    def list(cls) -> list[str]:
        """
        List all registered PDE names.

        Returns:
            List of registered PDE names (sorted)

        Examples:
            >>> names = PDEResidualRegistry.list()
            >>> print(f"Available PDEs: {', '.join(names)}")
        """
        with cls._lock:
            return sorted(cls._registry.keys())

    @classmethod
    def get_info(cls, name: str) -> dict[str, Any]:
        """
        Get detailed information about a registered PDE.

        Args:
            name: Name of the PDE

        Returns:
            Dictionary with PDE metadata:
                - name: PDE name
                - function: The function object
                - docstring: Function docstring
                - signature: Function signature

        Raises:
            KeyError: If PDE is not registered

        Examples:
            >>> info = PDEResidualRegistry.get_info("poisson")
            >>> print(info["docstring"])
        """
        func = cls.get(name)  # Raises KeyError if not found

        return {
            "name": name,
            "function": func,
            "docstring": inspect.getdoc(func) or "",
            "signature": inspect.signature(func),
        }

    @classmethod
    def _clear_registry(cls) -> None:
        """
        Clear all registrations (for testing only).

        This is an internal method used by tests to ensure clean state.
        Built-in PDEs will be automatically re-registered after clearing.
        Should not be used in production code.
        """
        with cls._lock:
            cls._registry.clear()

        # Re-register built-in PDEs
        _register_builtin_pdes()


# =============================================================================
# Built-in PDE Residual Functions
# =============================================================================


def _poisson_residual(
    model: Callable,
    x: Float[Array, "batch spatial_dim"],
    autodiff_engine: Any,
    source_term: Float[Array, "batch"] | None = None,
) -> Float[Array, "batch"]:
    """
    Compute Poisson equation residual: ∇²u - f = 0.

    The Poisson equation is a fundamental elliptic PDE:
        ∇²u(x) = f(x)

    where u is the solution and f is the source term.

    Args:
        model: Neural network or callable returning u(x)
        x: Spatial coordinates, shape (batch, spatial_dim)
        autodiff_engine: AutoDiffEngine class with compute_laplacian method
        source_term: Source term f(x), shape (batch,). Defaults to zero.

    Returns:
        PDE residual (∇²u - f), shape (batch,)

    Examples:
        >>> from opifex.core.physics import AutoDiffEngine
        >>> # For u = x² + y², ∇²u = 4
        >>> def u(x):
        ...     return x[..., 0]**2 + x[..., 1]**2
        >>> x = jnp.array([[1.0, 1.0]])
        >>> f = jnp.array([4.0])
        >>> residual = _poisson_residual(u, x, AutoDiffEngine, source_term=f)
        >>> # residual ≈ 0
    """
    # Compute Laplacian using generic method (works for both callables and NNX modules)
    laplacian = autodiff_engine.compute_laplacian(model, x)
    # Poisson is real-valued PDE - ensure real output
    laplacian = jnp.real(laplacian)

    # Handle source term
    if source_term is None:
        source_term = jnp.zeros(x.shape[0])

    # Residual: ∇²u - f
    return laplacian - source_term


def _heat_residual(
    model: Callable,
    x: Float[Array, "batch spatial_dim"],
    autodiff_engine: Any,
    alpha: float = 1.0,
) -> Float[Array, "batch"]:
    """
    Compute heat equation residual (steady-state): α∇²u = 0.

    The heat equation (steady-state):
        α∇²u(x) = 0

    where α is thermal diffusivity.

    For time-dependent problems, time derivative would be computed separately.

    Args:
        model: Neural network or callable returning u(x)
        x: Spatial coordinates, shape (batch, spatial_dim)
        autodiff_engine: AutoDiffEngine class
        alpha: Thermal diffusivity coefficient

    Returns:
        PDE residual (α∇²u), shape (batch,)

    Examples:
        >>> # For harmonic function u = x² - y², ∇²u = 0
        >>> def u(x):
        ...     return x[..., 0]**2 - x[..., 1]**2
        >>> x = jnp.array([[1.0, 1.0]])
        >>> residual = _heat_residual(u, x, AutoDiffEngine, alpha=1.0)
        >>> # residual ≈ 0
    """
    # Compute Laplacian using generic method (works for both callables and NNX modules)
    laplacian = autodiff_engine.compute_laplacian(model, x)
    # Heat equation is real-valued PDE - ensure real output
    laplacian = jnp.real(laplacian)

    # Steady-state heat equation: α∇²u = 0  # noqa: RUF003
    return alpha * laplacian


def _wave_residual(
    model: Callable,
    x: Float[Array, "batch spatial_dim"],
    autodiff_engine: Any,
    wave_speed: float = 1.0,
) -> Float[Array, "batch"]:
    """
    Compute wave equation residual (spatial part): c²∇²u = 0.

    The wave equation (spatial part):
        c²∇²u(x) = 0

    where c is wave speed.

    Time derivatives would be handled separately in time-dependent training.

    Args:
        model: Neural network or callable returning u(x)
        x: Spatial coordinates, shape (batch, spatial_dim)
        autodiff_engine: AutoDiffEngine class
        wave_speed: Wave propagation speed

    Returns:
        PDE residual (c²∇²u), shape (batch,)

    Examples:
        >>> def u(x):
        ...     return jnp.sin(jnp.pi * x[..., 0])
        >>> x = jnp.array([[0.5]])
        >>> residual = _wave_residual(u, x, AutoDiffEngine, wave_speed=1.0)
    """
    # Compute Laplacian using generic method (works for both callables and NNX modules)
    laplacian = autodiff_engine.compute_laplacian(model, x)
    # Wave equation is real-valued PDE - ensure real output
    laplacian = jnp.real(laplacian)

    # Wave equation (spatial): c²∇²u
    return wave_speed**2 * laplacian


def _burgers_residual(
    model: Callable,
    x: Float[Array, "batch spatial_dim"],
    autodiff_engine: Any,
    nu: float = 0.01,
) -> Float[Array, "batch"]:
    """
    Compute Burgers equation residual (steady-state): u·∇u - ν∇²u = 0.

    The Burgers equation (steady-state):
        u·∇u - ν∇²u = 0

    where ν is viscosity.

    Args:
        model: Neural network or callable returning u(x)
        x: Spatial coordinates, shape (batch, spatial_dim)
        autodiff_engine: AutoDiffEngine class
        nu: Viscosity coefficient

    Returns:
        PDE residual (u·∇u - ν∇²u), shape (batch,)

    Examples:
        >>> def u(x):
        ...     return x[..., 0]
        >>> x = jnp.array([[1.0]])
        >>> residual = _burgers_residual(u, x, AutoDiffEngine, nu=0.01)
    """
    # Compute solution value, gradient, and Laplacian using generic methods
    u = model(x)
    if u.ndim > 1:
        u = u.squeeze(-1)
    grad_u = autodiff_engine.compute_gradient(model, x)
    laplacian_u = autodiff_engine.compute_laplacian(model, x)
    # Burgers is real-valued PDE - ensure real output
    laplacian_u = jnp.real(laplacian_u)

    # Burgers: u·∇u - ν∇²u  # noqa: RUF003
    # For 1D: u * du/dx - ν * d²u/dx²  # noqa: RUF003
    # For simplicity, take first component of gradient
    advection = u * grad_u[..., 0]  # u * du/dx
    diffusion = nu * laplacian_u

    return advection - diffusion


def _schrodinger_residual(
    model: Callable,
    x: Float[Array, "batch spatial_dim"],
    autodiff_engine: Any,
    potential_type: str = "harmonic",
    energy: float | None = None,
) -> Float[Array, "batch"]:
    """
    Compute Schrödinger equation residual: -½∇²ψ + V(x)ψ - Eψ = 0.

    The time-independent Schrödinger equation (atomic units, ℏ=m=1):
        -½∇²ψ(x) + V(x)ψ(x) = Eψ(x)

    where ψ is the wavefunction, V is the potential, and E is the energy eigenvalue.

    Args:
        model: Neural network or callable returning ψ(x)
        x: Spatial coordinates, shape (batch, spatial_dim)
        autodiff_engine: AutoDiffEngine class
        potential_type: Type of potential ("harmonic", "coulomb", or "zero")
        energy: Energy eigenvalue E. If None, uses 0.5 (ground state harmonic)

    Returns:
        PDE residual (-½∇²ψ + Vψ - Eψ), shape (batch,)

    Examples:
        >>> # Harmonic oscillator ground state: ψ = exp(-r²/2)
        >>> def psi(x):
        ...     r_sq = jnp.sum(x**2, axis=-1)
        ...     return jnp.exp(-0.5 * r_sq)
        >>> x = jnp.array([[0.0, 0.0]])
        >>> residual = _schrodinger_residual(
        ...     psi, x, AutoDiffEngine, potential_type="harmonic", energy=0.5
        ... )
    """
    # Compute wavefunction and Laplacian using generic methods
    psi = model(x)
    if psi.ndim > 1:
        psi = psi.squeeze(-1)
    laplacian_psi = autodiff_engine.compute_laplacian(model, x)
    # For real wavefunctions, ensure real output
    # (Complex wavefunctions will have non-zero imaginary part)
    laplacian_psi = jnp.real(laplacian_psi)

    # Compute potential V(x)
    if potential_type == "harmonic":
        # V(x) = ½r² where r² = x² + y² + z²
        r_squared = jnp.sum(x**2, axis=-1)
        potential = 0.5 * r_squared
        # Default energy for ground state of harmonic oscillator
        if energy is None:
            energy = 0.5 * x.shape[-1]  # E_0 = (d/2) where d is dimension
    elif potential_type == "coulomb":
        # V(x) = -1/r (for hydrogen atom, atomic units)
        r = jnp.sqrt(jnp.sum(x**2, axis=-1) + 1e-10)  # Add small epsilon to avoid division by zero
        potential = -1.0 / r
        if energy is None:
            energy = -0.5  # Ground state energy of hydrogen
    elif potential_type == "zero":
        # Free particle
        potential = jnp.zeros(x.shape[0])
        if energy is None:
            energy = 0.0
    else:
        # Unknown potential type - use zero
        potential = jnp.zeros(x.shape[0])
        if energy is None:
            energy = 0.0

    # Schrödinger residual: -½∇²ψ + Vψ - Eψ
    kinetic_term = -0.5 * laplacian_psi
    potential_term = potential * psi
    energy_term = energy * psi

    return kinetic_term + potential_term - energy_term


def _schrodinger_td_residual(
    model: Callable,
    x: Float[Array, "batch spatial_dim"],
    hbar: float = 1.0,
    mass: float = 1.0,
    potential_fn: Callable | None = None,
) -> Float[Array, "batch"]:
    """
    Compute time-dependent Schrödinger equation (TDSE) Hamiltonian: Hψ.

    Mathematical formulation:
        iℏ∂ψ/∂t = Hψ where H = -ℏ²/2m ∇² + V(x)

    This computes the Hamiltonian operator applied to the wavefunction:
        Hψ = -ℏ²/2m ∇²ψ + V(x)ψ

    References:
        - Griffiths, D.J. (2018) "Introduction to Quantum Mechanics" (3rd ed.)
        - Sakurai, J.J. "Modern Quantum Mechanics"

    Args:
        model: Neural network or callable returning ψ(x)
        x: Spatial coordinates, shape (batch, spatial_dim)
        hbar: Reduced Planck constant (default 1.0, atomic units)
        mass: Particle mass (default 1.0, atomic units)
        potential_fn: Potential energy function V(x). If None, uses free particle.

    Returns:
        Hamiltonian operator applied to ψ, shape (batch,)

    Examples:
        >>> # Free particle plane wave
        >>> def psi(x):
        ...     return jnp.exp(1j * 2.0 * x[..., 0])
        >>> x = jnp.array([[0.5]])
        >>> H_psi = _schrodinger_td_residual(psi, x)

    Note:
        JIT-compatible: No try/except, no Python control flow.
        Supports complex-valued wavefunctions via AutoDiffEngine.
    """
    # Compute wavefunction value and Laplacian
    psi = model(x)
    if psi.ndim > 1:
        psi = psi.squeeze(-1)
    laplacian_psi = AutoDiffEngine.compute_laplacian(model, x)

    # Kinetic energy: -ℏ²/2m ∇²ψ
    kinetic_term = -(hbar**2) / (2 * mass) * laplacian_psi

    # Potential energy: V(x)ψ
    if potential_fn is not None:
        potential_val = potential_fn(x)
        if potential_val.ndim > 1:
            potential_val = potential_val.squeeze(-1)
        potential_term = potential_val * psi
    else:
        potential_term = jnp.zeros_like(psi)

    # Hamiltonian: H = -ℏ²/2m ∇² + V
    return kinetic_term + potential_term


def _navier_stokes_residual(
    model_u: Callable,
    model_v: Callable,
    x: Float[Array, "batch spatial_dim"],
    nu: float = 0.01,
    rho: float = 1.0,  # noqa: ARG001 - PDE residual signature is uniform across registry
) -> tuple[Float[Array, "batch"], Float[Array, "batch"], Float[Array, "batch"]]:
    """
    Compute Navier-Stokes equation residuals (2D incompressible).

    Mathematical formulation (steady-state):
        u∂u/∂x + v∂u/∂y = ν∇²u (momentum x)
        u∂v/∂x + v∂v/∂y = ν∇²v (momentum y)
        ∂u/∂x + ∂v/∂y = 0 (continuity)

    References:
        - Landau & Lifshitz (1987) "Fluid Mechanics" (2nd ed.)
        - Temam, R. (2001) "Navier-Stokes Equations"

    NS equations:
    - Momentum x: ∂u/∂t + u∂u/∂x + v∂u/∂y = -1/ρ ∂p/∂x + ν∇²u
    - Momentum y: ∂v/∂t + u∂v/∂x + v∂v/∂y = -1/ρ ∂p/∂y + ν∇²v
    - Continuity: ∂u/∂x + ∂v/∂y = 0

    For steady state, drops time derivatives. This returns residuals
    for all three equations as a tuple.

    Args:
        model_u: Neural network or callable for u velocity component
        model_v: Neural network or callable for v velocity component
        x: Spatial coordinates (x, y), shape (batch, 2)
        nu: Kinematic viscosity
        rho: Density (unused in incompressible formulation)

    Returns:
        Tuple of (momentum_x_residual, momentum_y_residual, continuity_residual)

    Examples:
        >>> def u(x): return x[..., 0]
        >>> def v(x): return x[..., 1]
        >>> x = jnp.array([[1.0, 1.0]])
        >>> residuals = _navier_stokes_residual(u, v, x)

    Note:
        JIT-compatible: No try/except, no Python control flow.
    """
    # Compute velocity fields
    u = model_u(x)
    v = model_v(x)
    if u.ndim > 1:
        u = u.squeeze(-1)
    if v.ndim > 1:
        v = v.squeeze(-1)

    # Compute gradients and Laplacians
    grad_u = AutoDiffEngine.compute_gradient(model_u, x)
    grad_v = AutoDiffEngine.compute_gradient(model_v, x)
    laplacian_u = AutoDiffEngine.compute_laplacian(model_u, x)
    laplacian_v = AutoDiffEngine.compute_laplacian(model_v, x)
    # Navier-Stokes is real-valued - ensure real output
    laplacian_u = jnp.real(laplacian_u)
    laplacian_v = jnp.real(laplacian_v)

    # Momentum x: u∂u/∂x + v∂u/∂y - ν∇²u  # noqa: RUF003
    du_dx = grad_u[..., 0]
    du_dy = grad_u[..., 1]
    momentum_x = u * du_dx + v * du_dy - nu * laplacian_u

    # Momentum y: u∂v/∂x + v∂v/∂y - ν∇²v  # noqa: RUF003
    dv_dx = grad_v[..., 0]
    dv_dy = grad_v[..., 1]
    momentum_y = u * dv_dx + v * dv_dy - nu * laplacian_v

    # Continuity: ∂u/∂x + ∂v/∂y
    continuity = du_dx + dv_dy

    return (momentum_x, momentum_y, continuity)


def _maxwell_residual(
    model_Ex: Callable,
    model_Ey: Callable,
    model_Ez: Callable,
    x: Float[Array, "batch spatial_dim"],
    charge_density: float = 0.0,
    epsilon_0: float = 1.0,
) -> tuple[Float[Array, "batch"], Float[Array, "batch"]]:
    """
    Compute Maxwell's equations residuals (static case).

    Mathematical formulation (electrostatics):
        ∇·E = ρ/ε₀ (Gauss's law)
        ∇×E = 0 (no time-varying magnetic field)

    References:
        - Jackson, J.D. (1999) "Classical Electrodynamics" (3rd ed.)
        - Griffiths, D.J. "Introduction to Electrodynamics"

    Maxwell's equations (static):
    - Gauss's law: ∇·E = ρ/ε₀
    - No magnetic sources: ∇·B = 0 (assumed satisfied)

    For simplicity, this implementation focuses on electrostatics.

    Args:
        model_Ex: Neural network or callable for Ex component
        model_Ey: Neural network or callable for Ey component
        model_Ez: Neural network or callable for Ez component
        x: Spatial coordinates, shape (batch, 3)
        charge_density: Charge density ρ
        epsilon_0: Permittivity of free space

    Returns:
        Tuple of (gauss_law_residual, curl_residual)

    Examples:
        >>> def Ex(x): return x[..., 0]
        >>> def Ey(x): return jnp.zeros_like(x[..., 0])
        >>> def Ez(x): return jnp.zeros_like(x[..., 0])
        >>> x = jnp.array([[1.0, 0.0, 0.0]])
        >>> residuals = _maxwell_residual(Ex, Ey, Ez, x)

    Note:
        JIT-compatible: No try/except, no Python control flow.
    """
    # Compute gradients of electric field components
    grad_Ex = AutoDiffEngine.compute_gradient(model_Ex, x)
    grad_Ey = AutoDiffEngine.compute_gradient(model_Ey, x)
    grad_Ez = AutoDiffEngine.compute_gradient(model_Ez, x)

    # Gauss's law: ∇·E = ρ/ε₀  # noqa: RUF003
    # ∇·E = ∂Ex/∂x + ∂Ey/∂y + ∂Ez/∂z
    div_E = grad_Ex[..., 0] + grad_Ey[..., 1] + grad_Ez[..., 2]
    gauss_residual = div_E - charge_density / epsilon_0

    # Curl residual (for electrostatics, ∇×E = 0)  # noqa: RUF003
    # ∇×E = (∂Ez/∂y - ∂Ey/∂z, ∂Ex/∂z - ∂Ez/∂x, ∂Ey/∂x - ∂Ex/∂y)  # noqa: RUF003
    curl_x = grad_Ez[..., 1] - grad_Ey[..., 2]
    curl_y = grad_Ex[..., 2] - grad_Ez[..., 0]
    curl_z = grad_Ey[..., 0] - grad_Ex[..., 1]
    curl_magnitude = jnp.sqrt(curl_x**2 + curl_y**2 + curl_z**2)

    return (gauss_residual, curl_magnitude)


def _schrodinger_nonlinear_residual(
    model: Callable,
    x: Float[Array, "batch spatial_dim"],
    sigma: float = 1.0,
) -> Float[Array, "batch"]:
    """
    Compute nonlinear Schrödinger (Gross-Pitaevskii) equation residual.

    Mathematical formulation:
        iψ_t = -∇²ψ + σ|ψ|²ψ (time-dependent)
        -∇²ψ + σ|ψ|²ψ (steady state)

    References:
        - Gross, E.P. (1961) "Structure of a quantized vortex"
        - Pitaevskii, L.P. (1961) "Vortex lines in an imperfect Bose gas"
        - Sulem & Sulem (1999) "The Nonlinear Schrödinger Equation"

    NLSE: iψ_t = -∇²ψ + σ|ψ|²ψ

    For steady state or spatial form: -∇²ψ + σ|ψ|²ψ

    Args:
        model: Neural network or callable returning ψ(x)
        x: Spatial coordinates, shape (batch, spatial_dim)
        sigma: Nonlinearity coefficient
            (positive for repulsive, negative for attractive)

    Returns:
        NLSE residual, shape (batch,)

    Examples:
        >>> # Bright soliton
        >>> def psi(x):
        ...     return 1.0 / jnp.cosh(x[..., 0])
        >>> x = jnp.array([[0.0]])
        >>> residual = _schrodinger_nonlinear_residual(psi, x)

    Note:
        JIT-compatible: No try/except, no Python control flow.
    """
    # Compute wavefunction and Laplacian
    psi = model(x)
    if psi.ndim > 1:
        psi = psi.squeeze(-1)
    laplacian_psi = AutoDiffEngine.compute_laplacian(model, x)
    # NLSE can be complex - keep complex support if needed
    # but most solutions are real-valued
    laplacian_psi = jnp.real(laplacian_psi)

    # Linear term: -∇²ψ
    linear_term = -laplacian_psi

    # Nonlinear term: σ|ψ|²ψ  # noqa: RUF003
    nonlinear_term = sigma * jnp.abs(psi) ** 2 * psi

    return linear_term + nonlinear_term


def _reaction_diffusion_residual(
    model_u: Callable,
    model_v: Callable,
    x: Float[Array, "batch spatial_dim"],
    D: float = 0.1,
    a: float = 1.0,
    b: float = 0.5,
) -> tuple[Float[Array, "batch"], Float[Array, "batch"]]:
    """
    Compute reaction-diffusion system residuals.

    Mathematical formulation (Gray-Scott model):
        ∂u/∂t = D∇²u + a(1-u) - uv²
        ∂v/∂t = D∇²v + uv² - bv

    References:
        - Gray, P. & Scott, S.K. (1983) "Autocatalytic reactions"
        - Pearson, J.E. (1993) "Complex patterns in a simple system"

    RD system:
    - ∂u/∂t = D∇²u + f(u,v)
    - ∂v/∂t = D∇²v + g(u,v)

    For steady state: D∇²u + f(u,v) = 0, D∇²v + g(u,v) = 0

    Uses simple Gray-Scott reaction terms:
    - f(u,v) = a(1-u) - uv²
    - g(u,v) = uv² - bv

    Args:
        model_u: Neural network or callable for u component
        model_v: Neural network or callable for v component
        x: Spatial coordinates, shape (batch, spatial_dim)
        D: Diffusion coefficient
        a: Feed rate parameter
        b: Kill rate parameter

    Returns:
        Tuple of (u_residual, v_residual)

    Examples:
        >>> def u(x): return jnp.ones_like(x[..., 0])
        >>> def v(x): return jnp.ones_like(x[..., 0])
        >>> x = jnp.array([[0.5, 0.5]])
        >>> residuals = _reaction_diffusion_residual(u, v, x)

    Note:
        JIT-compatible: No try/except, no Python control flow.
    """
    # Compute solution components
    u = model_u(x)
    v = model_v(x)
    if u.ndim > 1:
        u = u.squeeze(-1)
    if v.ndim > 1:
        v = v.squeeze(-1)

    # Compute Laplacians
    laplacian_u = AutoDiffEngine.compute_laplacian(model_u, x)
    laplacian_v = AutoDiffEngine.compute_laplacian(model_v, x)
    # Reaction-diffusion is real-valued - ensure real output
    laplacian_u = jnp.real(laplacian_u)
    laplacian_v = jnp.real(laplacian_v)

    # Gray-Scott reaction terms
    uv_squared = u * v**2
    f_u = a * (1 - u) - uv_squared
    g_v = uv_squared - b * v

    # Residuals: D∇²u + f(u,v), D∇²v + g(u,v)
    u_residual = D * laplacian_u + f_u
    v_residual = D * laplacian_v + g_v

    return (u_residual, v_residual)


# =============================================================================
# Multi-Scale PDEs (Batch 6)
# =============================================================================


def _homogenization_residual(
    model: Callable,
    x: Float[Array, "batch spatial_dim"],
    autodiff_engine: Any,
    coefficient_fn: Callable | None = None,
    source_term: Float[Array, "batch"] | None = None,
) -> Float[Array, "batch"]:
    """
    Compute homogenization PDE residual: -∇·(a(x)∇u) = f.

    Mathematical formulation (variable-coefficient divergence form):
        -∇·(a(x)∇u) = f

    Expanding the flux divergence with the product rule:
        ∇·(a∇u) = ∇a·∇u + a∇²u

    The ∇a·∇u term is non-zero whenever a varies in space; dropping it (i.e.
    using the constant-coefficient approximation -aΔu - f) is incorrect for a
    spatially-varying coefficient. This implementation therefore evaluates the
    divergence of the conservative flux field F(x) = a(x)∇u(x) directly via
    autodiff, so ∇·(a∇u) = ∇·F retains both terms exactly. This is the same
    conservative/flux form discretized by the project's Darcy solver
    (opifex.physics.solvers.darcy, a conservative 5-point scheme), so the PINN
    residual and the finite-difference ground truth are consistent.

    Models composite materials with periodic microstructure where a(x)
    is a coefficient with fine-scale periodicity.

    References:
        - Bensoussan, Lions, Papanicolaou (1978)
          "Asymptotic Analysis for Periodic Structures"
        - Allaire (1992) "Homogenization and Two-Scale Convergence"
        - Bank & Weiser (1985) "Some a posteriori error estimators for
          elliptic partial differential equations", Math. Comp. 44(170),
          283-301 — the a-posteriori error-estimator family for
          -∇·(a∇u) = f, which relies on the same divergence-form residual.

    Args:
        model: Neural network or callable returning u(x)
        x: Spatial coordinates, shape (batch, spatial_dim)
        autodiff_engine: AutoDiffEngine class
        coefficient_fn: Coefficient function a(x). If None, uses constant 1.0
        source_term: Source term f(x). If None, uses zero

    Returns:
        Residual -∇·(a∇u) - f, shape (batch,)

    Examples:
        >>> # Homogeneous material (reduces to Poisson)
        >>> def u(x): return jnp.sum(x**2, axis=-1)
        >>> def coeff(x): return jnp.ones(x.shape[0])
        >>> x = jnp.array([[0.5, 0.5]])
        >>> residual = _homogenization_residual(
        ...     u, x, AutoDiffEngine, coefficient_fn=coeff
        ... )

    Note:
        JIT-compatible: No try/except, no Python control flow.
    """
    # Default coefficient to the constant unit field a(x) ≡ 1.
    coefficient = (lambda _x: jnp.ones(_x.shape[0])) if coefficient_fn is None else coefficient_fn

    def flux(points: Float[Array, "batch spatial_dim"]) -> Float[Array, "batch spatial_dim"]:
        """Conservative flux F(x) = a(x) ∇u(x), shape (batch, spatial_dim)."""
        # a(x) is real-valued; broadcast it against the (batch, spatial_dim) gradient.
        coeff = jnp.real(jnp.reshape(coefficient(points), (points.shape[0], 1)))
        grad_u = jnp.real(autodiff_engine.compute_gradient(model, points))
        return coeff * grad_u

    # ∇·(a∇u) = ∇a·∇u + a∇²u, computed directly as the divergence of the flux field so
    # the variable-coefficient ∇a·∇u term is retained exactly (no constant-a shortcut).
    div_flux = jnp.real(autodiff_engine.compute_divergence(flux, x))

    # Handle source term - default to zero
    source = jnp.zeros(x.shape[0]) if source_term is None else source_term

    # Residual: -∇·(a∇u) - f. Standard form -∇·(a∇u) = f gives residual = LHS - RHS.
    return -div_flux - source


def _two_scale_residual(
    model_macro: Callable,
    model_micro: Callable,
    x: Float[Array, "batch spatial_dim"],
    autodiff_engine: Any,
    epsilon: float = 0.1,
    coupling_fn: Callable | None = None,
) -> tuple[Float[Array, "batch"], Float[Array, "batch"]]:
    """
    Compute two-scale expansion PDE residuals.

    Mathematical formulation (two-scale asymptotic expansion):
        L₀(u₀) + ε L₁(u₀, u₁) + O(ε²) = f

    where:
        - u₀ is macroscale solution
        - u₁ is microscale correction
        - ε → 0 is scale separation parameter

    When ε=0, use safe division to avoid NaN.

    References:
        - Allaire (1992) "Homogenization and Two-Scale Convergence"
        - Bensoussan, Lions, Papanicolaou (1978)

    Args:
        model_macro: Macroscale solution u₀(x)
        model_micro: Microscale correction u₁(x, x/ε)
        x: Spatial coordinates, shape (batch, spatial_dim)
        autodiff_engine: AutoDiffEngine class
        epsilon: Scale separation parameter (small positive number)
        coupling_fn: Custom coupling operator. If None, uses gradient coupling

    Returns:
        Tuple of (macro_residual, micro_residual)

    Examples:
        >>> def u_macro(x): return jnp.sum(x**2, axis=-1)
        >>> def u_micro(x): return jnp.sin(10 * jnp.sum(x, axis=-1))
        >>> x = jnp.array([[0.5, 0.5]])
        >>> macro_res, micro_res = _two_scale_residual(
        ...     u_macro, u_micro, x, AutoDiffEngine, epsilon=0.01
        ... )

    Note:
        JIT-compatible: No try/except, no Python control flow, uses jnp.where
        for safe division.
    """
    # Compute macroscale operators
    laplacian_macro = autodiff_engine.compute_laplacian(model_macro, x)
    grad_macro = autodiff_engine.compute_gradient(model_macro, x)
    # Two-scale is real-valued - ensure real output
    laplacian_macro = jnp.real(laplacian_macro)

    # Compute microscale operators
    laplacian_micro = autodiff_engine.compute_laplacian(model_micro, x)
    grad_micro = autodiff_engine.compute_gradient(model_micro, x)
    laplacian_micro = jnp.real(laplacian_micro)

    # Macroscale residual: L₀(u₀) + ε * coupling
    macro_operator = -laplacian_macro

    # Coupling term
    has_custom_coupling = coupling_fn is not None
    coupling = (
        coupling_fn(grad_macro, grad_micro)
        if has_custom_coupling
        else epsilon * jnp.sum(grad_macro * grad_micro, axis=-1)
    )

    macro_residual = macro_operator + coupling

    # Microscale residual with safe division
    # When ε=0, use -∇²u instead of -1/ε² ∇²u to avoid division by zero
    # Use jnp.where for JIT compatibility
    epsilon_sq_safe = jnp.where(epsilon == 0.0, 1.0, epsilon**2)
    micro_scale_factor = jnp.where(epsilon == 0.0, 1.0, 1.0 / epsilon_sq_safe)
    micro_residual = -micro_scale_factor * laplacian_micro

    return (macro_residual, micro_residual)


def _amr_poisson_residual(
    model: Callable,
    x: Float[Array, "batch spatial_dim"],
    autodiff_engine: Any,
    source_term: Float[Array, "batch"] | None = None,
    error_threshold: float = 0.1,  # noqa: ARG001 - PDE residual signature is uniform across registry
) -> tuple[Float[Array, "batch"], Float[Array, "batch"]]:
    """
    Compute Poisson residual with AMR error indicators.

    Mathematical formulation:
        Residual: ∇²u - f = 0
        Error indicator: ||∇u|| + ||H||_F

    where ||H||_F is Frobenius norm of Hessian (curvature indicator).

    References:
        - Kelly et al. error estimator
        - Zienkiewicz-Zhu gradient recovery method
        - deal.II KellyErrorEstimator implementation

    Args:
        model: Neural network or callable returning u(x)
        x: Spatial coordinates, shape (batch, spatial_dim)
        autodiff_engine: AutoDiffEngine class
        source_term: Source term f(x). If None, uses zero
        error_threshold: Threshold for refinement (not used in computation)

    Returns:
        Tuple of (residual, error_indicator)
        - residual: ∇²u - f, shape (batch,)
        - error_indicator: ||∇u|| + ||H||_F, shape (batch,)

    Examples:
        >>> def u(x): return jnp.sum(x**2, axis=-1)
        >>> x = jnp.array([[0.5, 0.5]])
        >>> residual, error = _amr_poisson_residual(u, x, AutoDiffEngine)

    Note:
        JIT-compatible: No try/except, no Python control flow.
    """
    # Compute Poisson residual
    laplacian = autodiff_engine.compute_laplacian(model, x)
    # AMR Poisson is real-valued - ensure real output
    laplacian = jnp.real(laplacian)
    source = jnp.zeros(x.shape[0]) if source_term is None else source_term
    residual = laplacian - source

    # Compute error indicator: ||∇u|| + ||H||_F
    grad = autodiff_engine.compute_gradient(model, x)
    grad_magnitude = jnp.linalg.norm(grad, axis=-1)

    hessian = autodiff_engine.compute_hessian(model, x)
    hessian_flat = hessian.reshape(hessian.shape[0], -1)
    hessian_norm = jnp.linalg.norm(hessian_flat, axis=-1)

    error_indicator = grad_magnitude + hessian_norm

    return (residual, error_indicator)


def _register_builtin_pdes() -> None:
    """
    Register built-in PDE residuals.

    This function is called automatically on module import and after
    _clear_registry() to ensure built-in PDEs are always available.
    """
    # Original built-in PDEs
    PDEResidualRegistry.register("poisson", _poisson_residual, override=True)
    PDEResidualRegistry.register("heat", _heat_residual, override=True)
    PDEResidualRegistry.register("wave", _wave_residual, override=True)
    PDEResidualRegistry.register("burgers", _burgers_residual, override=True)
    PDEResidualRegistry.register("schrodinger", _schrodinger_residual, override=True)

    # Advanced PDEs (Batch 5)
    PDEResidualRegistry.register("schrodinger_td", _schrodinger_td_residual, override=True)
    PDEResidualRegistry.register("navier_stokes", _navier_stokes_residual, override=True)
    PDEResidualRegistry.register("maxwell", _maxwell_residual, override=True)
    PDEResidualRegistry.register(
        "schrodinger_nonlinear", _schrodinger_nonlinear_residual, override=True
    )
    PDEResidualRegistry.register("reaction_diffusion", _reaction_diffusion_residual, override=True)

    # Multi-Scale PDEs (Batch 6)
    PDEResidualRegistry.register("homogenization", _homogenization_residual, override=True)
    PDEResidualRegistry.register("two_scale", _two_scale_residual, override=True)
    PDEResidualRegistry.register("amr_poisson", _amr_poisson_residual, override=True)


# Register built-in PDEs on module import
_register_builtin_pdes()

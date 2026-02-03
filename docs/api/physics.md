# Physics API Reference

The `opifex.physics` package provides JAX-native physics solvers and numerical methods for scientific computing applications.

## Overview

The physics module offers:

- **PDE Solvers**: Numerical solvers for common PDEs (Burgers, diffusion-advection, shallow water)
- **Spectral Methods**: Fourier-based PDE solvers and analysis tools
- **Numerical Schemes**: Finite difference, finite element, spectral methods
- **Conservation Laws**: Tools for enforcing physical constraints
- **Quantum Spectral**: Quantum chemistry spectral solvers

## PDE Solvers

### Burgers Equation Solver

Numerical solver for the Burgers equation.

```python
from opifex.physics.solvers import BurgersSolver

class BurgersSolver:
    """
    JAX-native solver for Burgers equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²

    Implements adaptive finite difference scheme with automatic
    CFL condition management for stable integration.

    Args:
        spatial_resolution: Number of spatial grid points
        viscosity: Viscosity coefficient ν
        domain_bounds: Spatial domain (x_min, x_max)
        method: Numerical method ('upwind', 'central', 'weno')
        adaptive_dt: Use adaptive time stepping

    Example:
        >>> solver = BurgersSolver(
        ...     spatial_resolution=256,
        ...     viscosity=0.01,
        ...     domain_bounds=(-1.0, 1.0),
        ...     method='upwind'
        ... )
    """

    def __init__(
        self,
        spatial_resolution: int,
        viscosity: float,
        domain_bounds: Tuple[float, float] = (-1.0, 1.0),
        method: str = 'upwind',
        adaptive_dt: bool = True
    ):
        """Initialize Burgers equation solver."""

    def solve(
        self,
        initial_condition: Array,
        time_span: Tuple[float, float],
        num_steps: int
    ) -> Array:
        """
        Solve Burgers equation from initial condition.

        Args:
            initial_condition: Initial velocity field u(x, 0)
            time_span: Time interval (t_start, t_end)
            num_steps: Number of time steps to output

        Returns:
            Solution trajectory of shape (num_steps+1, spatial_resolution)

        Example:
            >>> import jax.numpy as jnp
            >>> # Gaussian initial condition
            >>> x = jnp.linspace(-1, 1, 256)
            >>> u0 = jnp.exp(-10*x**2)
            >>>
            >>> # Solve for t ∈ [0, 2]
            >>> solution = solver.solve(u0, (0.0, 2.0), num_steps=100)
            >>> print(solution.shape)  # (101, 256)
        """

    def solve_ivp(
        self,
        initial_condition: Array,
        t_eval: Array
    ) -> Array:
        """
        Solve with specific evaluation times.

        Args:
            initial_condition: Initial condition
            t_eval: Time points for solution output

        Returns:
            Solution at specified times
        """
```

### Diffusion-Advection Solver

Solver for transport equations with diffusion and advection.

```python
from opifex.physics.solvers import DiffusionAdvectionSolver

class DiffusionAdvectionSolver:
    """
    Solver for diffusion-advection equation: ∂u/∂t + v·∇u = κ∇²u

    Combines advection and diffusion processes, common in transport
    phenomena, heat transfer, and environmental modeling.

    Args:
        spatial_resolution: Grid resolution (1D or 2D)
        diffusion_coeff: Diffusion coefficient κ
        velocity_field: Velocity field v(x) or v(x, y)
        domain_bounds: Domain boundaries
        dimension: Spatial dimension (1 or 2)
        scheme: Numerical scheme ('upwind', 'central', 'tvd')

    Example:
        >>> # 2D heat transport with advection
        >>> velocity = lambda x, y: (jnp.ones_like(x), jnp.zeros_like(y))
        >>> solver = DiffusionAdvectionSolver(
        ...     spatial_resolution=128,
        ...     diffusion_coeff=0.1,
        ...     velocity_field=velocity,
        ...     domain_bounds=((-1, 1), (-1, 1)),
        ...     dimension=2
        ... )
    """

    def solve(
        self,
        initial_condition: Array,
        time_span: Tuple[float, float],
        num_steps: int,
        boundary_conditions: Optional[Dict] = None
    ) -> Array:
        """
        Solve diffusion-advection equation.

        Args:
            initial_condition: Initial scalar field
            time_span: Time interval
            num_steps: Number of output steps
            boundary_conditions: BC specification:
                - 'dirichlet': Fixed values
                - 'neumann': Fixed gradients
                - 'periodic': Periodic boundaries

        Returns:
            Solution trajectory

        Example:
            >>> # Heat source diffusing with flow
            >>> u0 = jnp.exp(-10*(X**2 + Y**2))
            >>> solution = solver.solve(
            ...     u0,
            ...     (0.0, 1.0),
            ...     num_steps=100,
            ...     boundary_conditions={'type': 'dirichlet', 'value': 0.0}
            ... )
        """
```

### Shallow Water Equations Solver

Solver for shallow water equations modeling fluid dynamics.

```python
from opifex.physics.solvers import ShallowWaterSolver

class ShallowWaterSolver:
    """
    Solver for shallow water equations:
    - ∂h/∂t + ∇·(hv) = 0 (continuity)
    - ∂v/∂t + v·∇v + g∇h = 0 (momentum)

    Models fluid flow in thin layers (rivers, oceans, atmosphere).

    Args:
        spatial_resolution: Grid resolution
        gravity: Gravitational acceleration
        domain_bounds: Spatial domain
        friction: Bottom friction coefficient
        coriolis: Coriolis parameter (for rotating frame)

    Example:
        >>> solver = ShallowWaterSolver(
        ...     spatial_resolution=256,
        ...     gravity=9.81,
        ...     domain_bounds=((-100, 100), (-100, 100)),
        ...     friction=0.01
        ... )
    """

    def solve(
        self,
        initial_height: Array,
        initial_velocity: Tuple[Array, Array],
        time_span: Tuple[float, float],
        num_steps: int
    ) -> Tuple[Array, Array, Array]:
        """
        Solve shallow water equations.

        Args:
            initial_height: Initial height field h(x, y, 0)
            initial_velocity: Initial velocity (u, v)
            time_span: Time interval
            num_steps: Number of output steps

        Returns:
            Tuple of (height_trajectory, u_trajectory, v_trajectory)

        Example:
            >>> # Dam break problem
            >>> h0 = jnp.where(X < 0, 2.0, 1.0)  # Step in height
            >>> u0 = jnp.zeros_like(h0)
            >>> v0 = jnp.zeros_like(h0)
            >>>
            >>> h, u, v = solver.solve(
            ...     h0, (u0, v0),
            ...     (0.0, 10.0),
            ...     num_steps=200
            ... )
        """
```

## Spectral Methods

### Fourier Spectral Solver

Pseudo-spectral solver using FFT for spatial derivatives.

```python
from opifex.physics.spectral import FourierSpectralSolver

class FourierSpectralSolver:
    """
    Fourier pseudo-spectral solver for PDEs with periodic BC.

    Uses FFT for high-accuracy spatial derivative computation.
    Ideal for periodic problems and smooth solutions.

    Args:
        grid_shape: Spatial grid shape (nx,) or (nx, ny)
        domain_size: Physical domain size
        equation_type: PDE type ('burgers', 'kdv', 'nls', 'custom')
        dealiasing: Apply 2/3 dealiasing rule

    Example:
        >>> # Korteweg-de Vries equation solver
        >>> solver = FourierSpectralSolver(
        ...     grid_shape=(512,),
        ...     domain_size=2*jnp.pi,
        ...     equation_type='kdv',
        ...     dealiasing=True
        ... )
    """

    def solve(
        self,
        initial_condition: Array,
        time_span: Tuple[float, float],
        dt: float,
        nonlinear_fn: Optional[Callable] = None
    ) -> Array:
        """
        Solve PDE using spectral method.

        Args:
            initial_condition: Initial condition in physical space
            time_span: Time interval
            dt: Time step
            nonlinear_fn: Custom nonlinear term function

        Returns:
            Solution trajectory in physical space

        Example:
            >>> # Solve Burgers equation spectrally
            >>> u0 = jnp.sin(2*jnp.pi*x / L)
            >>> solution = solver.solve(u0, (0, 1), dt=0.001)
        """

    def compute_derivative(
        self,
        field: Array,
        order: int = 1,
        axis: int = -1
    ) -> Array:
        """
        Compute spectral derivative.

        Args:
            field: Field to differentiate
            order: Derivative order
            axis: Axis along which to differentiate

        Returns:
            Spectral derivative

        Example:
            >>> # Compute ∂²u/∂x²
            >>> u_xx = solver.compute_derivative(u, order=2, axis=0)
        """
```

### Quantum Spectral Methods

Spectral methods for quantum chemistry calculations.

```python
from opifex.physics.spectral import QuantumSpectralSolver

class QuantumSpectralSolver:
    """
    Spectral methods for quantum mechanical systems.

    Solves Schrödinger equation and related quantum problems
    using spectral discretization.

    Args:
        basis_type: Basis set ('plane-wave', 'gaussian', 'slater')
        num_basis: Number of basis functions
        system: Physical system specification

    Example:
        >>> solver = QuantumSpectralSolver(
        ...     basis_type='plane-wave',
        ...     num_basis=256,
        ...     system='hydrogen-atom'
        ... )
    """

    def solve_eigenvalue_problem(
        self,
        hamiltonian: Array,
        num_states: int = 1
    ) -> Tuple[Array, Array]:
        """
        Solve quantum eigenvalue problem.

        Args:
            hamiltonian: Hamiltonian matrix
            num_states: Number of lowest states to compute

        Returns:
            Tuple of (eigenvalues, eigenvectors)

        Example:
            >>> # Solve for ground state
            >>> H = construct_hamiltonian(atoms, basis)
            >>> energies, wavefunctions = solver.solve_eigenvalue_problem(
            ...     H,
            ...     num_states=10
            ... )
            >>> ground_state_energy = energies[0]
        """
```

## Numerical Integration

### Time Stepping Methods

Various time integration schemes.

```python
from opifex.physics.solvers import TimeIntegrator

class TimeIntegrator:
    """
    Time integration schemes for PDEs and ODEs.

    Supports:
    - Explicit methods: Forward Euler, RK4, RK45
    - Implicit methods: Backward Euler, BDF
    - IMEX methods: For stiff problems
    - Symplectic methods: For Hamiltonian systems
    """

    @staticmethod
    def rk4_step(
        rhs_fn: Callable,
        y: Array,
        t: float,
        dt: float
    ) -> Array:
        """
        Single RK4 time step.

        Args:
            rhs_fn: Right-hand side function dy/dt = f(y, t)
            y: Current state
            t: Current time
            dt: Time step

        Returns:
            Next state y(t + dt)
        """

    @staticmethod
    def adaptive_rk45(
        rhs_fn: Callable,
        y0: Array,
        t_span: Tuple[float, float],
        rtol: float = 1e-6,
        atol: float = 1e-8
    ) -> Array:
        """
        Adaptive RK45 (Dormand-Prince) integration.

        Args:
            rhs_fn: Right-hand side function
            y0: Initial condition
            t_span: Time interval
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Solution trajectory
        """

    @staticmethod
    def imex_step(
        linear_fn: Callable,
        nonlinear_fn: Callable,
        y: Array,
        t: float,
        dt: float,
        order: int = 2
    ) -> Array:
        """
        IMEX (Implicit-Explicit) time step.

        Treats linear terms implicitly, nonlinear explicitly.
        Ideal for stiff PDEs.

        Args:
            linear_fn: Linear operator L(y)
            nonlinear_fn: Nonlinear term N(y)
            y: Current state
            t: Current time
            dt: Time step
            order: Method order (1, 2, or 3)

        Returns:
            Next state
        """
```

## Conservation and Stability

### Conservation Law Enforcement

Tools for enforcing physical conservation laws.

```python
from opifex.physics import ConservationLaw

class ConservationLaw:
    """
    Enforce conservation laws in numerical solutions.

    Ensures mass, momentum, energy conservation through
    projection or correction methods.
    """

    @staticmethod
    def enforce_mass_conservation(
        solution: Array,
        target_mass: float
    ) -> Array:
        """
        Project solution to conserve total mass.

        Args:
            solution: Current solution field
            target_mass: Target total mass

        Returns:
            Mass-conserving solution
        """

    @staticmethod
    def enforce_energy_conservation(
        solution: Array,
        kinetic_fn: Callable,
        potential_fn: Callable,
        target_energy: float
    ) -> Array:
        """
        Enforce energy conservation.

        Args:
            solution: Current solution
            kinetic_fn: Kinetic energy functional
            potential_fn: Potential energy functional
            target_energy: Target total energy

        Returns:
            Energy-conserving solution
        """

    @staticmethod
    def check_divergence_free(
        velocity_field: Tuple[Array, ...],
        tolerance: float = 1e-6
    ) -> Tuple[bool, float]:
        """
        Check if velocity field is divergence-free.

        Args:
            velocity_field: Velocity components (u, v, w)
            tolerance: Divergence tolerance

        Returns:
            Tuple of (is_divergence_free, max_divergence)
        """
```

### Stability Analysis

Analyze numerical stability.

```python
from opifex.physics import StabilityAnalyzer

class StabilityAnalyzer:
    """Numerical stability analysis tools."""

    @staticmethod
    def compute_cfl_number(
        velocity: Array,
        dx: float,
        dt: float
    ) -> float:
        """
        Compute Courant-Friedrichs-Lewy number.

        Args:
            velocity: Velocity field
            dx: Spatial grid spacing
            dt: Time step

        Returns:
            CFL number (should be < 1 for stability)
        """

    @staticmethod
    def von_neumann_stability(
        scheme_amplification: Callable,
        wave_numbers: Array
    ) -> Tuple[bool, Array]:
        """
        Von Neumann stability analysis.

        Args:
            scheme_amplification: Amplification factor function
            wave_numbers: Wave numbers to test

        Returns:
            Tuple of (is_stable, amplification_factors)
        """
```

## Integration Examples

### Complete PDE Solving Workflow

```python
import jax
import jax.numpy as jnp
from opifex.physics.solvers import BurgersSolver
from opifex.visualization import create_physics_animation

# Setup
key = jax.random.PRNGKey(0)
solver = BurgersSolver(
    spatial_resolution=512,
    viscosity=0.01,
    domain_bounds=(-1.0, 1.0),
    method='upwind',
    adaptive_dt=True
)

# Initial condition: shock wave
x = jnp.linspace(-1, 1, 512)
u0 = jnp.where(x < 0, 1.0, -0.5)

# Solve
solution = solver.solve(
    u0,
    time_span=(0.0, 2.0),
    num_steps=200
)

# Visualize
anim = create_physics_animation(
    solution,
    title='Burgers Equation: Shock Formation',
    save_path='burgers_shock.gif'
)

# Analyze conservation
from opifex.physics import ConservationLaw
initial_mass = jnp.sum(u0)
final_mass = jnp.sum(solution[-1])
mass_error = abs(final_mass - initial_mass) / initial_mass
print(f"Mass conservation error: {mass_error:.2e}")
```

### Coupling with Neural Operators

```python
from opifex.neural.operators.fno import FNO
from opifex.physics.solvers import DiffusionAdvectionSolver

# Generate training data using physics solver
solver = DiffusionAdvectionSolver(
    spatial_resolution=128,
    diffusion_coeff=0.1,
    velocity_field=lambda x, y: (x, -y),
    dimension=2
)

# Generate dataset
n_samples = 1000
X_train, y_train = [], []

for i in range(n_samples):
    # Random initial condition
    u0 = jax.random.normal(key, (128, 128))

    # Solve to get target
    solution = solver.solve(u0, (0, 1), num_steps=10)

    X_train.append(u0)
    y_train.append(solution[-1])  # Final state

X_train = jnp.stack(X_train)
y_train = jnp.stack(y_train)

# Train neural operator to approximate solver
model = FNO(modes=12, width=64)
# ... training code ...

# Use neural operator for fast inference
prediction = model(X_train[0])  # Much faster than numerical solver
```

## Performance Considerations

### JAX Compilation

```python
# JIT compile solvers for performance
@jax.jit
def solve_batch(initial_conditions):
    """Solve multiple instances in parallel."""
    return jax.vmap(solver.solve)(initial_conditions)

# Vectorize over parameter variations
@jax.jit
def parameter_sweep(viscosities, u0):
    """Sweep over viscosity parameter."""
    return jax.vmap(
        lambda nu: BurgersSolver(256, nu).solve(u0, (0, 1), 100)
    )(viscosities)
```

### GPU Acceleration

```python
# Solvers automatically use GPU if available
solution = solver.solve(u0, (0, 1), 1000)  # Runs on GPU

# For multi-GPU
from jax.experimental import multihost_utils
solution = multihost_utils.process_allgather(local_solution)
```

## Neural Tangent Kernel (NTK) Analysis {: #ntk }

Tools for spectral analysis and training diagnostics via the Neural Tangent Kernel.

### NTK Wrapper

::: opifex.core.physics.ntk.wrapper
    options:
        show_root_heading: true
        show_source: false
        members:
            - NTKWrapper
            - NTKConfig

### Spectral Analysis

::: opifex.core.physics.ntk.spectral_analysis
    options:
        show_root_heading: true
        show_source: false
        members:
            - NTKSpectralAnalyzer
            - compute_effective_rank
            - estimate_convergence_rate
            - estimate_epochs_to_convergence
            - identify_slow_modes
            - detect_spectral_bias
            - compute_mode_convergence_rates

### Training Diagnostics

::: opifex.core.physics.ntk.diagnostics
    options:
        show_root_heading: true
        show_source: false
        members:
            - NTKDiagnostics
            - NTKDiagnosticsCallback

For detailed usage and theoretical background, see the [NTK Analysis Guide](../methods/ntk-analysis.md).

## GradNorm Loss Balancing {: #gradnorm }

Multi-task loss balancing through gradient magnitude normalization.

::: opifex.core.physics.gradnorm
    options:
        show_root_heading: true
        show_source: false
        members:
            - GradNormBalancer
            - GradNormConfig
            - compute_gradient_norms
            - compute_inverse_training_rates

For algorithm details and best practices, see the [GradNorm Guide](../methods/gradnorm.md).

## See Also

- [Core API](core.md): Problem definition and boundary conditions
- [Neural API](neural.md): Physics-informed neural networks
- [Data API](data.md): PDE datasets
- [Visualization API](visualization.md): Solution visualization
- [NTK Analysis Guide](../methods/ntk-analysis.md): Detailed NTK usage
- [GradNorm Guide](../methods/gradnorm.md): Multi-task loss balancing

# Problem Definition Guide

## Overview

The Opifex framework provides a unified, extensible interface for defining scientific problems across multiple domains. This comprehensive system supports partial differential equations (PDEs), ordinary differential equations (ODEs), optimization problems, and quantum mechanical calculations, all built on JAX for high-performance computation and automatic differentiation.

The problem definition system is designed with modularity and extensibility in mind, allowing researchers to easily specify complex scientific problems while maintaining compatibility with the entire Opifex ecosystem of neural operators, physics-informed neural networks, and quantum neural networks.

## Core Problem Types

### 1. Partial Differential Equations (PDEs)

PDEs form the backbone of many scientific simulations. The Opifex framework provides comprehensive support for defining and solving PDEs using both traditional numerical methods and neural approaches.

#### Basic PDE Problem Definition

```python
from opifex.core.problems import PDEProblem
from opifex.core.conditions import DirichletBC, NeumannBC, InitialCondition
import jax.numpy as jnp

class HeatEquationProblem(PDEProblem):
    """2D Heat equation with mixed boundary conditions."""

    def __init__(self, diffusivity=0.01):
        # Define spatial-temporal domain
        domain = {
            "x": (0.0, 1.0),
            "y": (0.0, 1.0),
            "t": (0.0, 1.0)
        }

        # Define boundary conditions
        boundary_conditions = [
            DirichletBC(boundary="left", value=0.0),
            DirichletBC(boundary="right", value=1.0),
            NeumannBC(boundary="top", value=0.0),
            NeumannBC(boundary="bottom", value=0.0)
        ]

        # Define initial condition
        initial_conditions = [
            InitialCondition(
                name="u",
                value=lambda x: jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
                dimension=1
            )
        ]

        super().__init__(
            domain=domain,
            equation=self._heat_equation,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            parameters={"diffusivity": diffusivity},
            time_dependent=True
        )

    def residual(self, x, u, u_derivatives):
        """Compute PDE residual for physics-informed training."""
        alpha = self.parameters["diffusivity"]
        u_t = u_derivatives["t"]
        u_xx = u_derivatives["xx"]
        u_yy = u_derivatives["yy"]
        return u_t - alpha * (u_xx + u_yy)

    def _heat_equation(self, x, y, t, u, u_derivatives, params):
        """Heat equation: ∂u/∂t = α∇²u"""
        return self.residual(jnp.array([x, y, t]), u, u_derivatives)

# Create and use the problem
heat_problem = HeatEquationProblem(diffusivity=0.01)
print(f"Domain: {heat_problem.get_domain()}")
print(f"Parameters: {heat_problem.get_parameters()}")
```

#### Advanced PDE Examples

#### Navier-Stokes Equations

```python
class NavierStokesProblem(PDEProblem):
    """2D incompressible Navier-Stokes equations."""

    def __init__(self, reynolds_number=100):
        domain = {
            "x": (0.0, 2.0),
            "y": (0.0, 1.0),
            "t": (0.0, 10.0)
        }

        # No-slip boundary conditions on walls
        boundary_conditions = [
            DirichletBC(boundary="top", value=jnp.array([0.0, 0.0])),    # u, v = 0
            DirichletBC(boundary="bottom", value=jnp.array([0.0, 0.0])), # u, v = 0
            DirichletBC(boundary="left", value=jnp.array([1.0, 0.0])),   # inlet: u=1, v=0
            NeumannBC(boundary="right", value=jnp.array([0.0, 0.0]))     # outlet: ∂u/∂n=0
        ]

        super().__init__(
            domain=domain,
            equation=self._navier_stokes,
            boundary_conditions=boundary_conditions,
            parameters={"Re": reynolds_number},
            time_dependent=True
        )

    def residual(self, x, u, u_derivatives):
        """Navier-Stokes residual: ∂u/∂t + u·∇u = -∇p + (1/Re)∇²u"""
        Re = self.parameters["Re"]
        u_vel, v_vel, pressure = u[..., 0], u[..., 1], u[..., 2]

        # Velocity derivatives
        u_t = u_derivatives["t"][..., 0]
        v_t = u_derivatives["t"][..., 1]
        u_x, u_y = u_derivatives["x"][..., 0], u_derivatives["y"][..., 0]
        v_x, v_y = u_derivatives["x"][..., 1], u_derivatives["y"][..., 1]
        u_xx, u_yy = u_derivatives["xx"][..., 0], u_derivatives["yy"][..., 0]
        v_xx, v_yy = u_derivatives["xx"][..., 1], u_derivatives["yy"][..., 1]

        # Pressure derivatives
        p_x, p_y = u_derivatives["x"][..., 2], u_derivatives["y"][..., 2]

        # Momentum equations
        momentum_x = u_t + u_vel * u_x + v_vel * u_y + p_x - (1/Re) * (u_xx + u_yy)
        momentum_y = v_t + u_vel * v_x + v_vel * v_y + p_y - (1/Re) * (v_xx + v_yy)

        # Continuity equation
        continuity = u_x + v_y

        return jnp.stack([momentum_x, momentum_y, continuity], axis=-1)
```

#### Wave Equation with Source Terms

```python
class WaveEquationProblem(PDEProblem):
    """2D wave equation with source terms."""

    def __init__(self, wave_speed=1.0):
        domain = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "t": (0.0, 2.0)}

        # Absorbing boundary conditions
        boundary_conditions = [
            RobinBC(boundary="all", alpha=1.0, beta=wave_speed, gamma=0.0)
        ]

        # Initial conditions: Gaussian pulse
        initial_conditions = [
            InitialCondition(
                variable="u",
                function=lambda x, y: jnp.exp(-(x**2 + y**2) / 0.1)
            ),
            InitialCondition(
                variable="u_t",
                function=lambda x, y: jnp.zeros_like(x)
            )
        ]

        super().__init__(
            domain=domain,
            equation=self._wave_equation,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            parameters={"c": wave_speed}
        )

    def residual(self, x, u, u_derivatives):
        """Wave equation: ∂²u/∂t² = c²∇²u + f(x,y,t)"""
        c = self.parameters["c"]
        u_tt = u_derivatives["tt"]
        u_xx = u_derivatives["xx"]
        u_yy = u_derivatives["yy"]

        # Source term (moving Gaussian)
        x_pos, y_pos, t = x[..., 0], x[..., 1], x[..., 2]
        source = jnp.exp(-((x_pos - 0.5*t)**2 + y_pos**2) / 0.05)

        return u_tt - c**2 * (u_xx + u_yy) - source
```

### 2. Ordinary Differential Equations (ODEs)

The framework supports both initial value problems (IVPs) and boundary value problems (BVPs) with sophisticated parameter handling.

#### Basic ODE Systems

```python
from opifex.core.problems import ODEProblem
import jax.numpy as jnp

class LorenzSystem(ODEProblem):
    """Chaotic Lorenz system."""

    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        super().__init__(
            time_span=(0.0, 20.0),
            equation=self._lorenz_rhs,
            initial_conditions={"u": jnp.array([1.0, 1.0, 1.0])},
            parameters={"sigma": sigma, "rho": rho, "beta": beta}
        )

    def rhs(self, t, y):
        """Lorenz system: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz"""
        x, y_val, z = y
        sigma, rho, beta = self.parameters["sigma"], self.parameters["rho"], self.parameters["beta"]

        dxdt = sigma * (y_val - x)
        dydt = x * (rho - z) - y_val
        dzdt = x * y_val - beta * z

        return jnp.array([dxdt, dydt, dzdt])

    def _lorenz_rhs(self, t, y, params):
        return self.rhs(t, y)

# Stiff ODE example
class VanDerPolOscillator(ODEProblem):
    """Van der Pol oscillator with adjustable stiffness."""

    def __init__(self, mu=1.0):
        super().__init__(
            time_span=(0.0, 20.0),
            equation=self._van_der_pol_rhs,
            initial_conditions={"u": jnp.array([2.0, 0.0])},
            parameters={"mu": mu}
        )

    def rhs(self, t, y):
        """Van der Pol: d²x/dt² - μ(1-x²)dx/dt + x = 0"""
        x, v = y
        mu = self.parameters["mu"]

        dxdt = v
        dvdt = mu * (1 - x**2) * v - x

        return jnp.array([dxdt, dvdt])
```

#### Coupled ODE-PDE Systems

```python
class ReactionDiffusionSystem(PDEProblem):
    """Coupled reaction-diffusion system with ODE kinetics."""

    def __init__(self, D_u=1.0, D_v=0.5, reaction_params=None):
        if reaction_params is None:
            reaction_params = {"a": 1.0, "b": 3.0, "k": 1.0}

        domain = {"x": (0.0, 10.0), "y": (0.0, 10.0), "t": (0.0, 50.0)}

        # No-flux boundary conditions
        boundary_conditions = [
            NeumannBC(boundary="all", value=0.0)
        ]

        super().__init__(
            domain=domain,
            equation=self._reaction_diffusion,
            boundary_conditions=boundary_conditions,
            parameters={"D_u": D_u, "D_v": D_v, **reaction_params}
        )

    def residual(self, x, u, u_derivatives):
        """Reaction-diffusion: ∂u/∂t = D∇²u + R(u,v)"""
        D_u, D_v = self.parameters["D_u"], self.parameters["D_v"]
        a, b, k = self.parameters["a"], self.parameters["b"], self.parameters["k"]

        u_conc, v_conc = u[..., 0], u[..., 1]
        u_t, v_t = u_derivatives["t"][..., 0], u_derivatives["t"][..., 1]
        u_laplacian = u_derivatives["xx"][..., 0] + u_derivatives["yy"][..., 0]
        v_laplacian = u_derivatives["xx"][..., 1] + u_derivatives["yy"][..., 1]

        # Reaction terms (Schnakenberg kinetics)
        reaction_u = a - u_conc + u_conc**2 * v_conc
        reaction_v = b - u_conc**2 * v_conc

        residual_u = u_t - D_u * u_laplacian - reaction_u
        residual_v = v_t - D_v * v_laplacian - reaction_v

        return jnp.stack([residual_u, residual_v], axis=-1)
```

### 3. Optimization Problems

The framework provides sophisticated optimization problem definitions with support for constraints, multi-objective optimization, and learn-to-optimize applications.

#### Constrained Optimization

```python
from opifex.core.problems import OptimizationProblem
import jax
import jax.numpy as jnp

class ConstrainedQuadraticProblem(OptimizationProblem):
    """Quadratic programming with equality and inequality constraints."""

    def __init__(self, Q, c, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None):
        dimension = Q.shape[0]

        # Define constraint functions
        constraints = []
        if A_eq is not None:
            constraints.extend([
                lambda x, i=i: A_eq[i] @ x - b_eq[i]
                for i in range(A_eq.shape[0])
            ])
        if A_ineq is not None:
            constraints.extend([
                lambda x, i=i: A_ineq[i] @ x - b_ineq[i]
                for i in range(A_ineq.shape[0])
            ])

        super().__init__(
            dimension=dimension,
            bounds=[(-10.0, 10.0)] * dimension,
            constraints=constraints,
            parameters={
                "Q": Q, "c": c,
                "n_eq": A_eq.shape[0] if A_eq is not None else 0,
                "n_ineq": A_ineq.shape[0] if A_ineq is not None else 0
            }
        )
        self.Q = Q
        self.c = c

    def objective(self, x):
        """Quadratic objective: f(x) = 0.5 * x^T Q x + c^T x"""
        return 0.5 * x.T @ self.Q @ x + self.c.T @ x

# Multi-objective optimization
class MultiObjectiveProblem(OptimizationProblem):
    """Multi-objective optimization problem."""

    def __init__(self, objectives, weights=None):
        self.objectives = objectives
        self.weights = weights or jnp.ones(len(objectives))

        super().__init__(
            dimension=2,  # Example: 2D problem
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            parameters={"n_objectives": len(objectives)}
        )

    def objective(self, x):
        """Weighted sum of objectives."""
        values = jnp.array([obj(x) for obj in self.objectives])
        return jnp.sum(self.weights * values)

    def pareto_objectives(self, x):
        """Return all objective values for Pareto analysis."""
        return jnp.array([obj(x) for obj in self.objectives])

# Example usage
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def sphere(x):
    return jnp.sum(x**2)

multi_obj = MultiObjectiveProblem([rosenbrock, sphere], weights=jnp.array([0.7, 0.3]))
```

### 4. Quantum Mechanical Problems

The framework includes first-class support for quantum mechanical calculations, including electronic structure problems and molecular dynamics.

#### Electronic Structure Problems

```python
from opifex.core.problems import QuantumProblem
from opifex.core.quantum.molecular_system import create_molecular_system

class DFTProblem(QuantumProblem):
    """Density Functional Theory problem for molecular systems."""

    def __init__(self, atoms, positions, charge=0, multiplicity=1):
        # Create molecular system
        molecular_system = create_molecular_system(
            atoms=atoms,
            positions=positions,
            charge=charge,
            multiplicity=multiplicity
        )

        super().__init__(
            molecular_system=molecular_system,
            method="neural_dft",
            convergence_threshold=1e-8,
            parameters={
                "exchange_functional": "PBE",
                "correlation_functional": "PBE",
                "basis_set": "def2-TZVP",
                "grid_density": "fine"
            }
        )

    def compute_energy(self, density=None):
        """Compute total electronic energy."""
        if density is None:
            # Use self-consistent field density
            density = self._scf_density()

        # Kinetic energy
        T = self._kinetic_energy(density)

        # External potential energy (electron-nuclear)
        V_ext = self._external_potential_energy(density)

        # Hartree energy (electron-electron repulsion)
        V_H = self._hartree_energy(density)

        # Exchange-correlation energy
        E_xc = self._exchange_correlation_energy(density)

        # Nuclear repulsion energy
        V_nn = self._nuclear_repulsion_energy()

        return T + V_ext + V_H + E_xc + V_nn

    def compute_forces(self, density=None):
        """Compute forces on nuclei using automatic differentiation."""
        energy_fn = lambda positions: self._energy_at_positions(positions, density)
        forces = -jax.grad(energy_fn)(self.molecular_system.positions)
        return forces

# Quantum dynamics problem
class QuantumDynamicsProblem(QuantumProblem):
    """Time-dependent Schrödinger equation."""

    def __init__(self, hamiltonian, initial_wavefunction, time_span=(0.0, 1.0)):
        # Create a minimal molecular system for the interface
        molecular_system = create_molecular_system(
            atoms=["H"],
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            charge=0
        )

        super().__init__(
            molecular_system=molecular_system,
            method="time_dependent_dft",
            parameters={
                "hamiltonian": hamiltonian,
                "initial_wavefunction": initial_wavefunction,
                "time_span": time_span
            }
        )

    def time_evolution(self, t, psi):
        """Time-dependent Schrödinger equation: iℏ ∂ψ/∂t = Ĥψ"""
        H = self.parameters["hamiltonian"]
        hbar = 1.0  # Atomic units
        return -1j / hbar * H @ psi
```

## Advanced Boundary Conditions

### Classical Boundary Conditions

The Opifex framework provides comprehensive support for all standard boundary condition types with advanced features like time-dependence and spatial variation.

#### Dirichlet Conditions

Dirichlet boundary conditions specify function values at boundaries. They are essential for problems where the solution value is known or constrained at domain boundaries.

```python
from opifex.core.conditions import DirichletBC
import jax.numpy as jnp

# Simple constant Dirichlet condition
constant_bc = DirichletBC(
    boundary="left",
    value=1.0
)

# Time-dependent Dirichlet condition
time_varying_bc = DirichletBC(
    boundary="right",
    value=lambda x, y, t: jnp.sin(2 * jnp.pi * t) * jnp.exp(-x**2),
    time_dependent=True
)

# Spatially-varying Dirichlet condition
spatial_bc = DirichletBC(
    boundary="top",
    value=lambda x, y, t: x**2 + y**2,
    spatial_dependent=True
)

# Vector-valued Dirichlet condition (for systems)
vector_bc = DirichletBC(
    boundary="inlet",
    value=jnp.array([1.0, 0.0, 0.0]),  # Velocity components [u, v, w]
    vector_valued=True
)

print("Dirichlet boundary conditions configured for various scenarios")
```

#### Neumann Conditions

Neumann boundary conditions specify derivative (flux) values at boundaries, commonly used for heat flux, mass flux, or stress conditions.

```python
from opifex.core.conditions import NeumannBC

# Constant flux condition
constant_flux = NeumannBC(
    boundary="top",
    value=0.1  # Heat flux
)

# Zero flux (insulation) condition
no_flux = NeumannBC(
    boundary="bottom",
    value=0.0
)

# Spatially-varying flux
def parabolic_flux(x, y, t):
    """Parabolic flux profile."""
    return -0.1 * x * (1 - x)  # Maximum at center, zero at edges

varying_flux = NeumannBC(
    boundary="right",
    value=parabolic_flux,
    spatial_dependent=True
)

print("Neumann boundary conditions configured for flux problems")
```

#### Robin Conditions

Robin (mixed) boundary conditions combine function values and derivatives, commonly used for convective heat transfer and radiation problems.

```python
from opifex.core.conditions import RobinBC

# Convective heat transfer: h(T - T_ambient) + k(dT/dn) = 0
convective_bc = RobinBC(
    boundary="surface",
    alpha=1.0,      # Coefficient of u (temperature)
    beta=0.1,       # Coefficient of ∂u/∂n (heat conduction)
    gamma=20.0      # External condition (ambient temperature)
)

# Time-varying ambient condition
def ambient_temperature(x, y, t):
    """Daily temperature variation."""
    return 20.0 + 10.0 * jnp.sin(2 * jnp.pi * t / 24.0)  # 24-hour cycle

time_varying_robin = RobinBC(
    boundary="exterior",
    alpha=1.0,
    beta=0.05,
    gamma=ambient_temperature,
    time_dependent=True
)

print("Robin boundary conditions configured for heat transfer problems")
```

#### Periodic Conditions

Periodic boundary conditions enforce solution continuity across domain boundaries, essential for problems with inherent periodicity.

```python
from opifex.core.conditions import PeriodicBC

# Simple periodic condition
periodic_x = PeriodicBC(
    boundary_pair=("left", "right"),
    direction="x"
)

# Periodic condition with phase shift
phase_shifted = PeriodicBC(
    boundary_pair=("bottom", "top"),
    direction="y",
    phase_shift=jnp.pi/4
)

# Vector periodic condition for fluid flow
vector_periodic = PeriodicBC(
    boundary_pair=("inlet", "outlet"),
    direction="x",
    vector_valued=True,
    components=[0, 1, 2]  # All velocity components
)

print("Periodic boundary conditions configured for various symmetries")
```

## Domain Specification and Geometry

### Geometric Domains

The Opifex framework provides sophisticated domain specification capabilities, from simple geometric shapes to complex multi-physics domains.

#### Basic Geometric Shapes

```python
from opifex.geometry import Rectangle, Circle, Polygon, Box, Sphere
import jax.numpy as jnp

# 2D Rectangular domain
rectangle = Rectangle(
    corner1=(0.0, 0.0),
    corner2=(2.0, 1.0),
    boundary_markers={
        "left": "inlet",
        "right": "outlet",
        "top": "wall",
        "bottom": "wall"
    }
)

# Circular domain with refined boundary
circle = Circle(
    center=(0.0, 0.0),
    radius=1.0,
    boundary_resolution=100,  # High resolution for curved boundary
    interior_points=5000
)

# Polygonal domain (airfoil shape)
airfoil_vertices = jnp.array([
    [1.0, 0.0],      # Trailing edge
    [0.8, 0.1],      # Upper surface
    [0.4, 0.15],
    [0.0, 0.05],     # Leading edge
    [0.4, -0.1],     # Lower surface
    [0.8, -0.05]
])

airfoil = Polygon(
    vertices=airfoil_vertices,
    boundary_markers={
        "airfoil_surface": [1, 2, 3, 4, 5],  # Surface elements
        "wake": [0]                           # Trailing edge
    }
)

print("Basic geometric domains configured")
```

#### Complex Geometric Operations

```python
from opifex.geometry import Union, Intersection, Difference
from opifex.geometry.csg import CSGDomain

# Complex domain using CSG operations
outer_circle = Circle(center=(0.0, 0.0), radius=2.0)
inner_circle = Circle(center=(0.0, 0.0), radius=0.5)
rectangular_slot = Rectangle(corner1=(-0.2, -3.0), corner2=(0.2, 3.0))

# Annular domain with rectangular slot
annular_region = Difference(outer_circle, inner_circle)
slotted_annulus = Difference(annular_region, rectangular_slot)

# Multi-hole geometry for heat transfer
base_plate = Rectangle(corner1=(-2.0, -1.0), corner2=(2.0, 1.0))
holes = [
    Circle(center=(-1.0, 0.0), radius=0.2),
    Circle(center=(0.0, 0.0), radius=0.2),
    Circle(center=(1.0, 0.0), radius=0.2)
]

perforated_plate = base_plate
for hole in holes:
    perforated_plate = Difference(perforated_plate, hole)

print("Complex CSG domains created")
```

#### Adaptive and Multi-Resolution Domains

```python
class AdaptiveDomain:
    """Domain with adaptive mesh refinement capabilities."""

    def __init__(self, base_geometry, initial_resolution=32):
        self.base_geometry = base_geometry
        self.resolution = initial_resolution
        self.refinement_levels = []

    def create_initial_mesh(self):
        """Create initial uniform mesh."""
        bounds = self.base_geometry.bounding_box()
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        x = jnp.linspace(x_min, x_max, self.resolution)
        y = jnp.linspace(y_min, y_max, self.resolution)

        X, Y = jnp.meshgrid(x, y, indexing='ij')
        points = jnp.stack([X.flatten(), Y.flatten()], axis=1)

        # Keep only points inside geometry
        inside_mask = self.base_geometry.contains(points)
        return points[inside_mask]

    def refine_mesh(self, solution, error_threshold=1e-3):
        """Adaptive mesh refinement based on solution gradients."""
        gradients = jnp.gradient(solution)
        error_indicator = jnp.linalg.norm(gradients, axis=0)

        # Mark elements for refinement
        refine_mask = error_indicator > error_threshold

        if jnp.any(refine_mask):
            refined_points = self._local_refinement(refine_mask)
            self.refinement_levels.append(refined_points)
            return True
        return False

print("Adaptive domains implemented")
```

### Graph Domains

For problems on irregular structures, networks, and discrete systems, the framework supports graph-based domains.

#### Network Structures

```python
from opifex.geometry.topology import GraphTopology, NetworkDomain
import jax.numpy as jnp

# Create molecular graph domain
def create_molecular_graph_domain(positions, atomic_numbers, cutoff_radius=3.0):
    """Create graph domain for molecular systems."""
    n_atoms = len(positions)

    # Compute pairwise distances
    distances = jnp.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=2
    )

    # Create edges for atoms within cutoff
    edge_mask = (distances < cutoff_radius) & (distances > 0)
    edge_indices = jnp.where(edge_mask)

    # Node features (atomic properties)
    node_features = jnp.column_stack([
        atomic_numbers.astype(float),           # Atomic number
        jnp.linalg.norm(positions, axis=1),     # Distance from origin
        jnp.sum(edge_mask, axis=1).astype(float)  # Coordination number
    ])

    # Edge features (bond properties)
    edge_distances = distances[edge_mask]
    edge_vectors = positions[edge_indices[1]] - positions[edge_indices[0]]
    edge_features = jnp.column_stack([
        edge_distances[:, None],
        edge_vectors,
        jnp.exp(-edge_distances[:, None])  # Exponential decay
    ])

    return GraphTopology(
        nodes=node_features,
        edges=jnp.stack(edge_indices, axis=1),
        edge_features=edge_features,
        domain_type="molecular"
    )

print("Graph domains created for molecular systems")
```

#### Irregular Connectivity Patterns

```python
# Irregular mesh connectivity
class IrregularMeshDomain:
    """Domain with irregular mesh connectivity."""

    def __init__(self, vertices, elements, boundary_markers=None):
        self.vertices = vertices
        self.elements = elements  # Connectivity matrix
        self.boundary_markers = boundary_markers or {}

        # Compute mesh properties
        self.adjacency_matrix = self._compute_adjacency()
        self.element_areas = self._compute_element_areas()

    def _compute_adjacency(self):
        """Compute vertex adjacency matrix."""
        n_vertices = len(self.vertices)
        adjacency = jnp.zeros((n_vertices, n_vertices))

        for element in self.elements:
            # Connect all vertices in each element
            for i in range(len(element)):
                for j in range(i+1, len(element)):
                    v1, v2 = element[i], element[j]
                    adjacency = adjacency.at[v1, v2].set(1)
                    adjacency = adjacency.at[v2, v1].set(1)

        return adjacency

    def get_boundary_vertices(self, marker=None):
        """Get vertices on specified boundary."""
        if marker is None:
            # Return all boundary vertices
            boundary_vertices = set()
            for marker_vertices in self.boundary_markers.values():
                boundary_vertices.update(marker_vertices)
            return list(boundary_vertices)
        else:
            return self.boundary_markers.get(marker, [])

print("Irregular connectivity patterns implemented")
```

#### Dynamic Graphs

```python
# Time-evolving graph domain
class DynamicGraphDomain:
    """Graph domain that evolves over time."""

    def __init__(self, initial_graph, evolution_rules):
        self.current_graph = initial_graph
        self.evolution_rules = evolution_rules
        self.time_history = [initial_graph]

    def evolve(self, dt, current_time):
        """Evolve graph structure based on rules."""
        new_graph = self.current_graph.copy()

        # Apply evolution rules
        for rule in self.evolution_rules:
            new_graph = rule.apply(new_graph, dt, current_time)

        self.current_graph = new_graph
        self.time_history.append(new_graph)

        return new_graph

    def get_graph_at_time(self, time_index):
        """Get graph state at specific time."""
        return self.time_history[time_index]

print("Dynamic graph domains implemented")
```

## Best Practices and Guidelines

### Problem Definition Checklist

1. **Domain Specification**

    - Ensure domain bounds are physically meaningful
    - Check for proper boundary condition coverage
    - Validate initial conditions for time-dependent problems

2. **Parameter Validation**

    - Implement parameter bounds checking
    - Use dimensionally consistent units
    - Document parameter physical meanings

3. **Numerical Stability**

    - Consider CFL conditions for time-dependent problems
    - Implement adaptive time stepping when needed
    - Use appropriate boundary condition types

4. **Testing and Validation**

    - Implement analytical solution comparisons when available
    - Use method of manufactured solutions for verification
    - Perform convergence studies

### Performance Optimization

```python
# Use JAX transformations for performance
@jax.jit
def optimized_residual_computation(problem, x, u, u_derivatives):
    """JIT-compiled residual computation."""
    return problem.residual(x, u, u_derivatives)

# Vectorized parameter studies
@jax.vmap
def solve_parameter_sweep(problem_params):
    """Vectorized solution over parameter space."""
    problem = create_problem_with_params(problem_params)
    return solve_problem(problem)

# Memory-efficient large-scale problems
def chunked_problem_solve(problem, chunk_size=1000):
    """Solve large problems in chunks to manage memory."""
    domain_points = problem.generate_domain_points()
    n_chunks = len(domain_points) // chunk_size

    solutions = []
    for i in range(n_chunks):
        chunk = domain_points[i*chunk_size:(i+1)*chunk_size]
        chunk_solution = solve_chunk(problem, chunk)
        solutions.append(chunk_solution)

    return jnp.concatenate(solutions)
```

This complete guide provides the foundation for defining and working with scientific problems in the Opifex framework. The unified interface allows seamless integration with neural networks, traditional solvers, and advanced optimization techniques while maintaining the flexibility needed for modern scientific machine learning research.

# Geometry & Computational Domains Guide

## Overview

The Opifex geometry framework provides comprehensive geometric modeling capabilities for scientific machine learning applications. Built on JAX for high-performance computation, it supports 2D/3D domain handling, constructive solid geometry (CSG) operations, Lie groups, Riemannian manifolds, graph neural networks, and molecular geometry modeling.

This system is designed to handle complex geometric problems in scientific computing, from simple rectangular domains to advanced manifold-based neural operators and molecular systems with quantum mechanical constraints.

## Core Geometric Primitives

### 2D Basic Shapes

The framework provides fundamental 2D shapes with comprehensive geometric operations:

```python
import jax
import jax.numpy as jnp
from opifex.geometry import Rectangle, Circle, Polygon

# Rectangle with center and dimensions
rect = Rectangle(
    center=jnp.array([0.0, 0.0]),
    width=2.0,
    height=1.5
)

# Circle with center and radius
circle = Circle(
    center=jnp.array([1.0, 0.5]),
    radius=0.8
)

# Polygon from vertices (counterclockwise ordering)
vertices = jnp.array([
    [-1.0, -1.0],
    [1.0, -1.0],
    [0.5, 1.0],
    [-0.5, 1.0]
])
polygon = Polygon(vertices=vertices)

# Basic geometric properties
rect_area = rect.width * rect.height
circle_area = jnp.pi * circle.radius**2

print(f"Rectangle area: {rect_area:.4f}")
print(f"Circle area: {circle_area:.4f}")
```

### Point Containment and Distance Functions

All shapes support efficient point containment testing and signed distance functions:

```python
# Test points for containment
test_points = jnp.array([
    [0.0, 0.0],    # Center of rectangle
    [1.0, 0.5],    # Center of circle
    [2.0, 2.0],    # Outside both
    [0.5, 0.25]    # Potential intersection
])

# Point containment (vectorized)
rect_contains = rect.contains(test_points)
circle_contains = circle.contains(test_points)

# Signed distance functions
rect_distances = jnp.array([rect.distance(pt) for pt in test_points])
circle_distances = jnp.array([circle.distance(pt) for pt in test_points])

print("Point containment and distances:")
for i, point in enumerate(test_points):
    print(f"Point {point}:")
    print(f"  Rectangle: contains={rect_contains[i]}, distance={rect_distances[i]:.3f}")
    print(f"  Circle: contains={circle_contains[i]}, distance={circle_distances[i]:.3f}")
```

### Boundary Sampling and Normal Computation

```python
# Sample points on shape boundaries
key = jax.random.PRNGKey(42)
rect_boundary = rect.sample_boundary(n_points=50, key=key)
circle_boundary = circle.sample_boundary(n_points=50, key=key)

# Compute outward normals at boundary points
rect_normals = jnp.array([rect.compute_normal(pt) for pt in rect_boundary])
circle_normals = jnp.array([circle.compute_normal(pt) for pt in circle_boundary])

print(f"Sampled {len(rect_boundary)} rectangle boundary points")
print(f"Sampled {len(circle_boundary)} circle boundary points")
print(f"Normal vectors computed for boundary analysis")
```

## Constructive Solid Geometry (CSG)

CSG operations enable complex shape creation through boolean operations:

### Basic CSG Operations

```python
from opifex.geometry import union, intersection, difference

# Create base shapes
base_rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=2.0)
cutout_circle = Circle(center=jnp.array([0.5, 0.5]), radius=0.6)

# Boolean operations
union_shape = union(base_rect, cutout_circle)           # A ∪ B
intersection_shape = intersection(base_rect, cutout_circle)  # A ∩ B
difference_shape = difference(base_rect, cutout_circle)      # A - B

# Test complex shape properties
test_point = jnp.array([0.3, 0.3])
print(f"Point {test_point} containment:")
print(f"  Union: {union_shape.contains(test_point)}")
print(f"  Intersection: {intersection_shape.contains(test_point)}")
print(f"  Difference: {difference_shape.contains(test_point)}")
```

### Advanced CSG Compositions

```python
# Create complex geometries through composition
outer_boundary = Circle(center=jnp.array([0.0, 0.0]), radius=2.0)
inner_hole = Circle(center=jnp.array([0.0, 0.0]), radius=0.8)
rectangular_slot = Rectangle(center=jnp.array([0.0, 0.0]), width=0.4, height=3.0)

# Annular region with rectangular slot
annular_region = difference(outer_boundary, inner_hole)
slotted_annulus = difference(annular_region, rectangular_slot)

# Multi-hole geometry
holes = [
    Circle(center=jnp.array([0.8, 0.8]), radius=0.2),
    Circle(center=jnp.array([-0.8, 0.8]), radius=0.2),
    Circle(center=jnp.array([0.8, -0.8]), radius=0.2),
    Circle(center=jnp.array([-0.8, -0.8]), radius=0.2)
]

multi_hole_plate = base_rect
for hole in holes:
    multi_hole_plate = difference(multi_hole_plate, hole)

print("Complex CSG geometries created successfully")
```

### Smooth CSG with SDF Operations

The framework uses signed distance functions (SDFs) for smooth, differentiable CSG operations:

```python
from opifex.geometry.csg import _SDFOperations

# Access internal SDF operations for custom compositions
sdf_ops = _SDFOperations()

def smooth_union_distance(point, shape1, shape2, smoothing=0.1):
    """Smooth union with controllable blending."""
    d1 = shape1.distance(point)
    d2 = shape2.distance(point)
    return sdf_ops.union_sdf(d1, d2)

def smooth_intersection_distance(point, shape1, shape2, smoothing=0.1):
    """Smooth intersection with controllable blending."""
    d1 = shape1.distance(point)
    d2 = shape2.distance(point)
    return sdf_ops.intersection_sdf(d1, d2)

# Example: Smooth blending between shapes
blend_point = jnp.array([0.5, 0.0])
smooth_dist = smooth_union_distance(blend_point, base_rect, cutout_circle)
print(f"Smooth union distance at {blend_point}: {smooth_dist:.4f}")
```

## Molecular Geometry and 3D Systems

### Molecular System Definition

```python
from opifex.geometry.csg import MolecularSystem
from opifex.geometry import create_molecular_geometry_from_dft_problem

# Define a water molecule (H2O) in atomic units
water_positions = jnp.array([
    [0.0000,  0.0000,  0.1173],   # Oxygen
    [0.0000,  0.7572, -0.4692],   # Hydrogen 1
    [0.0000, -0.7572, -0.4692]    # Hydrogen 2
])

atomic_numbers = jnp.array([8, 1, 1])  # O, H, H

water_molecule = MolecularSystem(
    positions=water_positions,
    atomic_numbers=atomic_numbers,
    charge=0,
    multiplicity=1
)

print(f"Water molecule properties:")
print(f"  Number of atoms: {water_molecule.n_atoms}")
print(f"  Total charge: {water_molecule.charge}")
print(f"  Spin multiplicity: {water_molecule.multiplicity}")
print(f"  Center of mass: {water_molecule.center_of_mass}")
```

### Periodic Systems and Crystal Structures

```python
from opifex.geometry.csg import PeriodicCell

# Define a cubic unit cell
lattice_vectors = jnp.array([
    [5.0, 0.0, 0.0],  # a vector
    [0.0, 5.0, 0.0],  # b vector
    [0.0, 0.0, 5.0]   # c vector
])

# Create periodic cell for crystal systems
unit_cell = PeriodicCell(
    lattice_vectors=lattice_vectors,
    atomic_positions=jnp.array([
        [0.0, 0.0, 0.0],    # Atom at origin
        [2.5, 2.5, 2.5]     # Atom at body center
    ]),
    atomic_numbers=jnp.array([14, 14]),  # Silicon atoms
    periodic_dimensions=[True, True, True]
)

print(f"Unit cell volume: {unit_cell.volume:.4f}")
print(f"Lattice parameters: {unit_cell.lattice_parameters}")
```

### Molecular Exclusion Domains

```python
# Create computational domain excluding molecular regions
molecular_geometry = create_molecular_geometry_from_dft_problem(water_molecule)

# Define computational box around molecule
box_size = 10.0  # Atomic units
computational_domain = create_computational_domain_with_molecular_exclusion(
    molecular_geometry=molecular_geometry,
    box_dimensions=jnp.array([box_size, box_size, box_size]),
    exclusion_radius=2.0,  # Exclude within 2 a.u. of atoms
    buffer_zone=1.0        # Additional buffer for numerical stability
)

print("Molecular exclusion domain created for quantum calculations")
```

## Advanced Manifolds and Differential Geometry

### Riemannian Manifolds

```python
from opifex.geometry.manifolds import SphericalManifold, TangentSpace

# Create spherical manifold for geometric deep learning
sphere_manifold = SphericalManifold(dim=2)  # 2-sphere (surface of 3D ball)

# Sample points on the manifold
key = jax.random.PRNGKey(123)
manifold_points = sphere_manifold.sample_uniform(n_points=100, key=key)

# Compute tangent spaces at sampled points
tangent_spaces = [
    TangentSpace(manifold=sphere_manifold, base_point=point)
    for point in manifold_points[:5]  # First 5 points
]

# Manifold operations
def parallel_transport_vector(manifold, vector, start_point, end_point):
    """Parallel transport a vector along the manifold."""
    # Simplified parallel transport for sphere
    # In practice, this would use proper Riemannian geometry
    return vector - jnp.dot(vector, end_point) * end_point

# Example: Transport vectors between points
start_point = manifold_points[0]
end_point = manifold_points[1]
tangent_vector = jnp.array([0.1, 0.2, 0.0])  # Tangent at start_point

transported_vector = parallel_transport_vector(
    sphere_manifold, tangent_vector, start_point, end_point
)

print(f"Manifold points sampled: {len(manifold_points)}")
print(f"Tangent spaces computed: {len(tangent_spaces)}")
print(f"Vector transport completed")
```

### Lie Groups and Algebraic Structures

```python
from opifex.geometry.algebra import SO3Group, SE3Group

# Special Orthogonal Group SO(3) - 3D rotations
so3_group = SO3Group()

# Generate random rotation matrices
rotation_key = jax.random.PRNGKey(456)
random_rotations = so3_group.sample_uniform(n_samples=10, key=rotation_key)

# Compose rotations (group operation)
R1 = random_rotations[0]
R2 = random_rotations[1]
composed_rotation = so3_group.compose(R1, R2)

# Compute group inverse
R1_inverse = so3_group.inverse(R1)

# Verify group properties
identity_check = so3_group.compose(R1, R1_inverse)
print(f"Group identity verification (should be close to I):")
print(f"Max deviation from identity: {jnp.max(jnp.abs(identity_check - jnp.eye(3))):.6f}")

# Special Euclidean Group SE(3) - 3D rigid transformations
se3_group = SE3Group()

# Create transformation matrices (rotation + translation)
translation = jnp.array([1.0, 2.0, 3.0])
transformation = se3_group.from_rotation_translation(R1, translation)

# Apply transformation to points
points_3d = jnp.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

transformed_points = se3_group.apply_transformation(transformation, points_3d)
print(f"Applied SE(3) transformation to {len(points_3d)} points")
```

## Graph Neural Networks and Topology

### Graph Structures for Scientific Computing

```python
from opifex.geometry.topology import GraphTopology, GraphNeuralOperator

# Create graph from molecular structure
def create_molecular_graph(positions, atomic_numbers, cutoff_radius=3.0):
    """Create molecular graph with distance-based edges."""
    n_atoms = len(positions)

    # Compute pairwise distances
    distances = jnp.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=2
    )

    # Create edges for atoms within cutoff
    edge_mask = (distances < cutoff_radius) & (distances > 0)
    edge_indices = jnp.where(edge_mask)

    # Edge features (distances and relative positions)
    edge_distances = distances[edge_mask]
    edge_vectors = (
        positions[edge_indices[1]] - positions[edge_indices[0]]
    )

    return GraphTopology(
        nodes=atomic_numbers.astype(float),  # Node features: atomic numbers
        edges=jnp.stack(edge_indices, axis=1),  # Edge connectivity
        edge_features=jnp.column_stack([
            edge_distances[:, None],
            edge_vectors
        ])
    )

# Create molecular graph for water
molecular_graph = create_molecular_graph(
    water_positions, atomic_numbers, cutoff_radius=2.0
)

print(f"Molecular graph created:")
print(f"  Nodes: {molecular_graph.nodes.shape}")
print(f"  Edges: {molecular_graph.edges.shape}")
print(f"  Edge features: {molecular_graph.edge_features.shape}")
```

### Graph Neural Operators

```python
from opifex.geometry.topology import GraphMessagePassing

# Create graph neural operator for molecular property prediction
graph_operator = GraphNeuralOperator(
    node_features=molecular_graph.nodes.shape[-1],
    edge_features=molecular_graph.edge_features.shape[-1],
    hidden_dim=64,
    n_layers=3,
    output_dim=1  # Scalar property prediction
)

# Message passing layer for custom graph operations
message_passing = GraphMessagePassing(
    node_dim=64,
    edge_dim=molecular_graph.edge_features.shape[-1],
    message_dim=32
)

print("Graph neural operators initialized for molecular ML")
```

### Topological Spaces and Simplicial Complexes

```python
from opifex.geometry.topology import SimplicialComplex, TopologicalSpace

# Create simplicial complex for topological data analysis
vertices = jnp.array([
    [0.0, 0.0], [1.0, 0.0], [0.5, 1.0],  # Triangle vertices
    [1.5, 0.5], [2.0, 1.0]                # Additional vertices
])

# Define simplices (0-simplices: vertices, 1-simplices: edges, 2-simplices: faces)
simplices = {
    0: jnp.arange(len(vertices)),  # All vertices
    1: jnp.array([[0, 1], [1, 2], [2, 0], [1, 3], [3, 4]]),  # Edges
    2: jnp.array([[0, 1, 2]])  # Triangle face
}

simplicial_complex = SimplicialComplex(
    vertices=vertices,
    simplices=simplices
)

# Compute topological properties
betti_numbers = simplicial_complex.compute_betti_numbers()
euler_characteristic = simplicial_complex.euler_characteristic()

print(f"Topological analysis:")
print(f"  Betti numbers: {betti_numbers}")
print(f"  Euler characteristic: {euler_characteristic}")
```

## Domain Discretization and Mesh Generation

### Structured Grid Generation

```python
def create_structured_grid(domain_bounds, resolution):
    """Create structured Cartesian grid."""
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]

    x = jnp.linspace(x_min, x_max, resolution[0])
    y = jnp.linspace(y_min, y_max, resolution[1])

    X, Y = jnp.meshgrid(x, y, indexing='ij')
    grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)

    return grid_points, (X, Y)

# Create grid for rectangular domain
domain_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
resolution = [64, 64]

grid_points, (X, Y) = create_structured_grid(domain_bounds, resolution)
print(f"Structured grid created: {grid_points.shape[0]} points")
```

### Adaptive Mesh Refinement

```python
def adaptive_refinement(geometry, initial_resolution=32, max_levels=3):
    """Adaptive mesh refinement based on geometry complexity."""

    def refinement_criterion(points):
        """Refine near boundaries and complex regions."""
        distances = jnp.array([geometry.distance(pt) for pt in points])
        return jnp.abs(distances) < 0.1  # Refine near boundaries

    # Start with coarse grid
    current_points, _ = create_structured_grid(
        [(-2.0, 2.0), (-2.0, 2.0)], [initial_resolution, initial_resolution]
    )

    refined_points = []

    for level in range(max_levels):
        # Identify points needing refinement
        refine_mask = refinement_criterion(current_points)

        # Keep non-refined points
        refined_points.extend(current_points[~refine_mask])

        # Refine marked regions
        if jnp.any(refine_mask):
            refine_centers = current_points[refine_mask]
            # Add finer points around each center
            for center in refine_centers:
                local_spacing = 2.0 / (initial_resolution * (2 ** (level + 1)))
                local_points = center + local_spacing * jnp.array([
                    [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]
                ])
                refined_points.extend(local_points)

    return jnp.array(refined_points)

# Apply adaptive refinement to complex geometry
refined_mesh = adaptive_refinement(slotted_annulus, initial_resolution=16, max_levels=2)
print(f"Adaptive mesh created: {len(refined_mesh)} points")
```

### Unstructured Mesh Generation

```python
def delaunay_triangulation_2d(points):
    """Simple Delaunay triangulation for 2D points."""
    # This is a simplified version - in practice, use scipy.spatial.Delaunay
    # or specialized mesh generation libraries

    from scipy.spatial import Delaunay
    import numpy as np

    # Convert JAX arrays to numpy for scipy
    points_np = np.array(points)
    tri = Delaunay(points_np)

    # Convert back to JAX arrays
    triangles = jnp.array(tri.simplices)

    return triangles

def generate_boundary_conforming_mesh(geometry, target_edge_length=0.1):
    """Generate mesh that conforms to geometry boundaries."""

    # Sample boundary points
    key = jax.random.PRNGKey(789)
    boundary_points = geometry.sample_boundary(
        n_points=int(2 * jnp.pi / target_edge_length), key=key
    )

    # Add interior points
    bbox_min = jnp.min(boundary_points, axis=0) - 0.5
    bbox_max = jnp.max(boundary_points, axis=0) + 0.5

    # Generate candidate interior points
    n_interior = 1000
    interior_candidates = jax.random.uniform(
        key, (n_interior, 2), minval=bbox_min, maxval=bbox_max
    )

    # Keep only points inside geometry
    inside_mask = geometry.contains(interior_candidates)
    interior_points = interior_candidates[inside_mask]

    # Combine boundary and interior points
    all_points = jnp.vstack([boundary_points, interior_points])

    # Generate triangulation
    triangles = delaunay_triangulation_2d(all_points)

    return all_points, triangles

# Generate mesh for complex geometry
mesh_points, mesh_triangles = generate_boundary_conforming_mesh(
    slotted_annulus, target_edge_length=0.05
)

print(f"Unstructured mesh generated:")
print(f"  Vertices: {len(mesh_points)}")
print(f"  Triangles: {len(mesh_triangles)}")
```

## Coordinate Systems and Transformations

### Coordinate System Transformations

```python
def cartesian_to_polar(x, y):
    """Convert Cartesian to polar coordinates."""
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    return r, theta

def polar_to_cartesian(r, theta):
    """Convert polar to Cartesian coordinates."""
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return x, y

def cartesian_to_spherical(x, y, z):
    """Convert Cartesian to spherical coordinates."""
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z / (r + 1e-10))  # Polar angle
    phi = jnp.arctan2(y, x)              # Azimuthal angle
    return r, theta, phi

# Example coordinate transformations
cartesian_points = jnp.array([
    [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]
])

polar_coords = jnp.array([
    cartesian_to_polar(pt[0], pt[1]) for pt in cartesian_points
])

print("Coordinate transformations:")
for i, (cart, polar) in enumerate(zip(cartesian_points, polar_coords)):
    print(f"  Point {i}: ({cart[0]:.1f}, {cart[1]:.1f}) → (r={polar[0]:.3f}, θ={polar[1]:.3f})")
```

### Geometric Transformations

```python
def create_transformation_matrix_2d(translation, rotation_angle, scale):
    """Create 2D transformation matrix."""
    cos_theta = jnp.cos(rotation_angle)
    sin_theta = jnp.sin(rotation_angle)

    # Homogeneous transformation matrix
    T = jnp.array([
        [scale[0] * cos_theta, -scale[0] * sin_theta, translation[0]],
        [scale[1] * sin_theta,  scale[1] * cos_theta, translation[1]],
        [0.0,                   0.0,                  1.0]
    ])

    return T

def apply_transformation_2d(points, transformation_matrix):
    """Apply 2D transformation to points."""
    # Convert to homogeneous coordinates
    homogeneous_points = jnp.column_stack([points, jnp.ones(len(points))])

    # Apply transformation
    transformed_homogeneous = homogeneous_points @ transformation_matrix.T

    # Convert back to Cartesian coordinates
    return transformed_homogeneous[:, :2]

# Example: Transform a square
square_vertices = jnp.array([
    [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]
])

# Create transformation: translate, rotate 45°, scale by 2
transform_matrix = create_transformation_matrix_2d(
    translation=jnp.array([1.0, 1.0]),
    rotation_angle=jnp.pi / 4,
    scale=jnp.array([2.0, 2.0])
)

transformed_square = apply_transformation_2d(square_vertices, transform_matrix)

print("Geometric transformation applied:")
print(f"Original square vertices: {square_vertices.shape}")
print(f"Transformed square vertices: {transformed_square.shape}")
```

## Integration with Physics Problems

### Domain Definition for PDE Problems

```python
from opifex.core.problems import PDEProblem
from opifex.core.conditions import DirichletBC, NeumannBC

class ComplexDomainPDEProblem(PDEProblem):
    """PDE problem on complex geometric domain."""

    def __init__(self, geometry, physics_parameters):
        # Use geometry for domain definition
        self.geometry = geometry

        # Define boundary conditions based on geometry
        boundary_conditions = [
            DirichletBC(boundary="outer", value=1.0),
            NeumannBC(boundary="inner", value=0.0)
        ]

        # Domain includes geometry information
        domain = {
            "geometry": geometry,
            "t": (0.0, 1.0)
        }

        super().__init__(
            domain=domain,
            equation=self._heat_equation_with_geometry,
            boundary_conditions=boundary_conditions,
            parameters=physics_parameters
        )

    def _heat_equation_with_geometry(self, x, y, t, u, u_derivatives, params):
        """Heat equation with geometry-dependent source term."""
        alpha = params["diffusivity"]
        u_t = u_derivatives["t"]
        u_xx = u_derivatives["xx"]
        u_yy = u_derivatives["yy"]

        # Geometry-dependent source term
        point = jnp.array([x, y])
        distance_to_boundary = self.geometry.distance(point)
        source_term = jnp.exp(-distance_to_boundary**2)

        return u_t - alpha * (u_xx + u_yy) - source_term

    def generate_collocation_points(self, n_points, key):
        """Generate physics-informed collocation points."""
        # Sample points inside the geometry
        bbox_min = jnp.array([-2.0, -2.0])
        bbox_max = jnp.array([2.0, 2.0])

        candidates = jax.random.uniform(
            key, (n_points * 3, 2), minval=bbox_min, maxval=bbox_max
        )

        # Keep only points inside geometry
        inside_mask = self.geometry.contains(candidates)
        interior_points = candidates[inside_mask][:n_points]

        return interior_points

# Create PDE problem with complex geometry
complex_pde = ComplexDomainPDEProblem(
    geometry=slotted_annulus,
    physics_parameters={"diffusivity": 0.01}
)

# Generate collocation points for PINN training
key = jax.random.PRNGKey(999)
collocation_points = complex_pde.generate_collocation_points(1000, key)

print(f"Complex domain PDE problem created")
print(f"Generated {len(collocation_points)} collocation points")
```

### Molecular Geometry for Quantum Problems

```python
from opifex.core.problems import QuantumProblem

class MolecularQuantumProblem(QuantumProblem):
    """Quantum problem with molecular geometry constraints."""

    def __init__(self, molecular_system, computational_domain):
        self.molecular_system = molecular_system
        self.computational_domain = computational_domain

        super().__init__(
            molecular_system=molecular_system,
            method="neural_dft",
            parameters={
                "computational_domain": computational_domain,
                "basis_cutoff": 10.0,  # Atomic units
                "grid_spacing": 0.1
            }
        )

    def generate_grid_points(self, spacing=0.1):
        """Generate computational grid excluding molecular regions."""
        # Create regular grid in computational domain
        bounds = self.computational_domain.bounds

        x = jnp.arange(bounds[0][0], bounds[0][1], spacing)
        y = jnp.arange(bounds[1][0], bounds[1][1], spacing)
        z = jnp.arange(bounds[2][0], bounds[2][1], spacing)

        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        grid_points = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

        # Exclude points too close to nuclei
        valid_points = []
        for point in grid_points:
            min_distance = jnp.min(jnp.linalg.norm(
                point - self.molecular_system.positions, axis=1
            ))
            if min_distance > 0.5:  # Minimum distance in atomic units
                valid_points.append(point)

        return jnp.array(valid_points)

# Create quantum problem with molecular geometry
quantum_problem = MolecularQuantumProblem(
    molecular_system=water_molecule,
    computational_domain=computational_domain
)

grid_points = quantum_problem.generate_grid_points(spacing=0.2)
print(f"Quantum grid generated: {len(grid_points)} points")
```

## Performance Optimization and Best Practices

### JAX Optimization Techniques

```python
# JIT compilation for geometric operations
@jax.jit
def batch_distance_computation(geometry, points):
    """JIT-compiled batch distance computation."""
    return jnp.array([geometry.distance(pt) for pt in points])

@jax.jit
def batch_containment_test(geometry, points):
    """JIT-compiled batch containment testing."""
    return geometry.contains(points)

# Vectorized operations for performance
@jax.vmap
def vectorized_normal_computation(geometry, points):
    """Vectorized normal computation."""
    return geometry.compute_normal(points)

# Example usage with performance timing
import time

large_point_set = jax.random.uniform(
    jax.random.PRNGKey(1000), (10000, 2), minval=-2.0, maxval=2.0
)

# Time JIT-compiled operations
start_time = time.time()
distances = batch_distance_computation(circle, large_point_set)
jit_time = time.time() - start_time

print(f"JIT-compiled distance computation: {jit_time:.4f}s for {len(large_point_set)} points")
```

### Memory-Efficient Geometry Operations

```python
def chunked_geometry_operations(geometry, points, chunk_size=1000):
    """Process large point sets in chunks to manage memory."""
    n_points = len(points)
    n_chunks = (n_points + chunk_size - 1) // chunk_size

    results = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_points)
        chunk = points[start_idx:end_idx]

        # Process chunk
        chunk_distances = batch_distance_computation(geometry, chunk)
        results.append(chunk_distances)

    return jnp.concatenate(results)

# Process very large point set efficiently
very_large_points = jax.random.uniform(
    jax.random.PRNGKey(1001), (50000, 2), minval=-3.0, maxval=3.0
)

chunked_distances = chunked_geometry_operations(
    slotted_annulus, very_large_points, chunk_size=5000
)

print(f"Processed {len(very_large_points)} points in chunks")
print(f"Memory-efficient computation completed")
```

### Geometry Caching and Precomputation

```python
class CachedGeometry:
    """Geometry wrapper with caching for expensive operations."""

    def __init__(self, base_geometry):
        self.base_geometry = base_geometry
        self._distance_cache = {}
        self._normal_cache = {}

    def distance(self, point):
        """Cached distance computation."""
        point_key = tuple(point.tolist())
        if point_key not in self._distance_cache:
            self._distance_cache[point_key] = self.base_geometry.distance(point)
        return self._distance_cache[point_key]

    def compute_normal(self, point):
        """Cached normal computation."""
        point_key = tuple(point.tolist())
        if point_key not in self._normal_cache:
            self._normal_cache[point_key] = self.base_geometry.compute_normal(point)
        return self._normal_cache[point_key]

    def clear_cache(self):
        """Clear all cached results."""
        self._distance_cache.clear()
        self._normal_cache.clear()

# Use cached geometry for repeated operations
cached_geometry = CachedGeometry(slotted_annulus)

# Repeated queries will be faster
test_point = jnp.array([0.5, 0.5])
for _ in range(100):
    distance = cached_geometry.distance(test_point)  # Cached after first call

print("Geometry caching implemented for performance optimization")
```

This comprehensive geometry guide provides the foundation for working with complex geometric problems in scientific machine learning. The unified framework supports everything from simple 2D domains to advanced manifold-based neural operators and quantum molecular systems, all optimized for high-performance computation with JAX.

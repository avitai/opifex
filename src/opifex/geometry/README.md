# Opifex Geometry: Geometric Framework with Advanced Manifolds

This package provides geometric modeling capabilities for scientific machine
learning, including 2D/3D domain handling, constructive solid geometry (CSG)
operations, Lie groups, Riemannian manifolds, and graph topology with manifold
neural operators.

## Package Structure

- **CSG operations** (`csg/`): Rectangle, Circle, Polygon with containment
  testing; union / intersection / difference; boundary sampling; 3D molecular
  geometry with periodic boundary conditions.
- **Lie groups** (`algebra/`): `SO3Group` and `SE3Group` with exponential /
  logarithm maps, group actions, and rigid-body transformations.
- **Riemannian manifolds** (`manifolds/`): `SphericalManifold`,
  `HyperbolicManifold` (Poincaré disk with gyrovector operations), and a general
  `RiemannianManifold` with custom metrics, plus manifold neural operators
  (`ManifoldNeuralOperator`, `HyperbolicNeuralOperator`,
  `RiemannianNeuralOperator`).
- **Topology** (`topology/`): `GraphTopology` graph container with adjacency /
  degree / Laplacian operators, simplicial complexes, and topological spaces.

## Usage Examples

### 1. Constructive Solid Geometry (CSG) Operations

```python
import jax
import jax.numpy as jnp
from opifex.geometry.csg import Rectangle, Circle, Polygon
from opifex.geometry import union, intersection, difference

# Create basic 2D shapes
rect = Rectangle(
    center=jnp.array([0.0, 0.0]),
    width=2.0,
    height=1.5
)

circle = Circle(
    center=jnp.array([1.0, 0.5]),
    radius=0.8
)

# Create polygon from vertices
vertices = jnp.array([
    [-1.0, -1.0],
    [1.0, -1.0],
    [0.5, 1.0],
    [-0.5, 1.0]
])
polygon = Polygon(vertices=vertices)

# CSG operations (binary operations on two shapes)
union_shape = union(rect, circle)
intersection_shape = intersection(rect, circle)
difference_shape = difference(rect, circle)

print(f"Union shape: {type(union_shape).__name__}")
print(f"Intersection shape: {type(intersection_shape).__name__}")
print(f"Difference shape: {type(difference_shape).__name__}")

# Point containment testing
test_points = jnp.array([
    [0.0, 0.0],    # Center of rectangle
    [1.0, 0.5],    # Center of circle
    [2.0, 2.0],    # Outside both
    [0.5, 0.25]    # In intersection
])

rect_contains = rect.contains(test_points)
circle_contains = circle.contains(test_points)
union_contains = union_shape.contains(test_points)

# Boundary sampling
boundary_points = rect.sample_boundary(n_points=50, key=jax.random.PRNGKey(42))
print(f"Boundary points shape: {boundary_points.shape}")
```

### 2. Molecular Geometry and 3D Structures

```python
from opifex.core.quantum.molecular_system import MolecularSystem

# Create a water molecule (H2O)
water_positions = jnp.array([
    [0.0000,  0.0000,  0.1173],   # O
    [0.0000,  0.7572, -0.4692],   # H
    [0.0000, -0.7572, -0.4692]    # H
])

atomic_numbers = jnp.array([8, 1, 1])  # O, H, H

water_molecule = MolecularSystem(
    positions=water_positions,
    atomic_numbers=atomic_numbers,
    charge=0,
    multiplicity=1
)

print(f"  Number of atoms: {water_molecule.n_atoms}")
print(f"  Total charge: {water_molecule.charge}")
print(f"  Center of mass: {water_molecule.center_of_mass}")

# Bond lengths from the distance matrix
oh_distance = jnp.linalg.norm(water_positions[0] - water_positions[1])

print(f"  O-H bond length: {oh_distance:.4f} bohr")
print(f"  Molecular formula: {water_molecule.molecular_formula}")

# Periodic boundary conditions for crystal structures
lattice_vectors = jnp.array([
    [5.0, 0.0, 0.0],
    [0.0, 5.0, 0.0],
    [0.0, 0.0, 5.0]
])

crystal = MolecularSystem(
    positions=water_positions,
    atomic_numbers=atomic_numbers,
    cell=lattice_vectors,
    pbc=(True, True, True)
)
```

### 3. Lie Groups and Symmetry Operations

```python
from opifex.geometry.algebra.groups import SO3Group, SE3Group

# SO(3) - 3D Rotation Group
so3 = SO3Group()

# Create rotations from axis-angle representation
axis_angle_1 = jnp.array([0.1, 0.2, 0.3])  # Small rotation
axis_angle_2 = jnp.array([jnp.pi/4, 0.0, 0.0])  # 45° around x-axis

R1 = so3.exp(axis_angle_1)
R2 = so3.exp(axis_angle_2)

# Group operations
R_composed = so3.compose(R1, R2)  # R2 * R1
R_inverse = so3.inverse(R1)
R_identity = so3.compose(R1, R_inverse)

print(f"Identity deviation: {jnp.max(jnp.abs(R_identity - jnp.eye(3))):.10f}")

# Logarithm map (rotation matrix to axis-angle)
recovered_axis_angle = so3.log(R1)
print(f"Recovery error: {jnp.linalg.norm(axis_angle_1 - recovered_axis_angle):.10f}")

# SE(3) - 3D Rigid Body Transformations
se3 = SE3Group()

# Create rigid transformation (rotation + translation)
translation = jnp.array([1.0, 2.0, 3.0])
transformation_matrix = se3.from_rotation_translation(R1, translation)

# Apply to molecular coordinates
water_positions = jnp.array([
    [0.0000,  0.0000,  0.1173],
    [0.0000,  0.7572, -0.4692],
    [0.0000, -0.7572, -0.4692]
])
rotated_water = so3.action(R2, water_positions)
transformed_water = se3.action(transformation_matrix, water_positions)
```

### 4. Riemannian Manifolds and Differential Geometry

```python
from opifex.geometry.manifolds import (
    SphericalManifold,
    HyperbolicManifold,
    RiemannianManifold,
)

# Spherical manifolds
sphere = SphericalManifold(dimension=2)  # 2-sphere (surface of 3D ball)

# Random points and tangent vectors
key = jax.random.PRNGKey(456)
point = sphere.random_point(key)
tangent_vector = sphere.random_tangent_vector(key, point)

print(f"Point norm: {jnp.linalg.norm(point):.10f} (should be 1.0)")
print(f"Orthogonality check: {jnp.dot(point, tangent_vector):.10f} (should be 0.0)")

# Geodesic computations
geodesic_point = sphere.exp(point, tangent_vector)  # Exponential map
recovered_tangent = sphere.log(point, geodesic_point)  # Logarithm map
print(f"Tangent recovery error: {jnp.linalg.norm(tangent_vector - recovered_tangent):.10f}")

# Geodesic distance
point2 = sphere.random_point(jax.random.split(key)[0])
geodesic_distance = sphere.distance(point, point2)

# Parallel transport
transported_vector = sphere.parallel_transport(point, geodesic_point, tangent_vector)

# Hyperbolic manifolds for hierarchical data
hyperbolic_space = HyperbolicManifold(curvature=-1.0, dimension=3)

h_point1 = hyperbolic_space.random_point(key)
h_point2 = hyperbolic_space.random_point(jax.random.split(key)[1])

# Gyrovector operations (unique to hyperbolic geometry)
gyrosum = hyperbolic_space.gyroaddition(h_point1, h_point2)
h_distance = hyperbolic_space.distance(h_point1, h_point2)
print(f"Hyperbolic distance: {h_distance:.6f}")

# Custom Riemannian manifold
def custom_metric_tensor(x):
    """Custom metric: g_ij = δ_ij * (1 + ||x||²)"""
    return jnp.eye(len(x)) * (1.0 + jnp.sum(x**2))

custom_manifold = RiemannianManifold(
    dimension=3,
    metric_function=custom_metric_tensor,
    embedding_dimension=3
)

# Differential geometry computations
custom_point = jnp.array([0.1, 0.2, 0.3])
metric = custom_manifold.metric_tensor(custom_point)
christoffel = custom_manifold.christoffel_symbols(custom_point)
riemann_tensor = custom_manifold.riemann_curvature(custom_point)
ricci_tensor = custom_manifold.ricci_curvature(custom_point)
scalar_curvature = custom_manifold.scalar_curvature(custom_point)
print(f"Scalar curvature: {scalar_curvature:.6f}")
```

### 5. Manifold Neural Operators

```python
from opifex.geometry.manifolds import ManifoldNeuralOperator
import flax.nnx as nnx

# Create manifold neural operators for different geometries
key = jax.random.PRNGKey(789)
rngs = nnx.Rngs(key)

# Spherical neural operator
sphere_operator = ManifoldNeuralOperator(
    manifold=sphere,
    hidden_dim=64,
    output_dim=16,
    num_layers=3,
    activation=nnx.tanh,
    rngs=rngs
)

# Hyperbolic neural operator for hierarchical data
hyperbolic_operator = ManifoldNeuralOperator(
    manifold=hyperbolic_space,
    hidden_dim=128,
    output_dim=32,
    num_layers=4,
    activation=nnx.swish,
    use_residual=True,
    rngs=rngs
)

# Generate test data
batch_size = 32
sphere_data = jax.vmap(sphere.random_point)(jax.random.split(key, batch_size))
hyperbolic_data = jax.vmap(hyperbolic_space.random_point)(jax.random.split(key, batch_size))

# Process data through manifold neural operators
sphere_output = sphere_operator(sphere_data)
hyperbolic_output = hyperbolic_operator(hyperbolic_data)

print(f"Spherical operator output shape: {sphere_output.shape}")
print(f"Hyperbolic operator output shape: {hyperbolic_output.shape}")
```

### 6. Graph topology and graph neural networks

`opifex.geometry.topology.GraphTopology` is the lightweight graph container — nodes,
edges, and the derived adjacency / degree / Laplacian operators:

```python
import jax.numpy as jnp
from opifex.geometry.topology import GraphTopology

# A molecular graph (water): nodes are atoms, edges are bonds
nodes = jnp.array([[8.0, 6.0], [1.0, 1.0], [1.0, 1.0]])  # per-atom features
edges = jnp.array([[0, 1], [0, 2]])                       # O-H1, O-H2
graph = GraphTopology(nodes=nodes, edges=edges)

print(graph.num_nodes, graph.num_edges)        # 3 2
adjacency = graph.adjacency_matrix             # (3, 3)
degree = graph.degree_matrix()                 # (3, 3)
laplacian = graph.laplacian_matrix(normalized=True)
```

For *learning* on graphs, use the dedicated neural subsystems:

- **Molecular / atomistic property prediction** (energy, forces, stress from atomic
  positions): `opifex.neural.atomistic` — E(3)-equivariant message-passing potentials
  (SchNet, PaiNN, NequIP) assembled as an `AtomisticModel` with energy / forces /
  stress heads.
- **Graph / mesh operators** (e.g. learned PDE operators on irregular meshes):
  `opifex.neural.operators.graph` (`GraphNeuralOperator`, `MeshGraphNet`).

## Technical Implementation

### JAX-Native Architecture

- **Pure JAX Operations**: All computations use `jax.numpy` and `jax.Array`.
- **Automatic Differentiation**: Full JAX autodiff compatibility for gradients.
- **JIT Compilation**: Optimized for JAX JIT compilation and vectorization.
- **GPU Acceleration**: Ready for CUDA/TPU acceleration through JAX.

### Type Safety & Validation

- **jax.Array Integration**: Native JAX array types throughout.
- **jaxtyping Annotations**: Precise shape and dtype specifications.
- **Protocol-Based Design**: Runtime-checkable interfaces for extensibility.
- **Input Validation**: Error checking and constraint validation.

## Key Features

- **Mathematical Rigor**: Differential geometry with JAX automatic differentiation.
- **Physical Realism**: Molecular geometry and quantum mechanical compatibility.
- **Neural Integration**: Connection with FLAX NNX neural networks.
- **Performance**: JIT-compiled operations with GPU acceleration.
- **Extensibility**: Protocol-based design for custom geometries.
- **Scientific Applications**: Quantum chemistry, protein folding, materials science.

## Dependencies

- **JAX**: Core array operations and automatic differentiation.
- **jaxtyping**: Type annotations for JAX arrays (Float, Int shapes).
- **Python 3.11+**: Modern Python features and type system.
</content>

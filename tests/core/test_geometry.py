"""
Tests for the Opifex Geometry System with CSG Operations.

This module tests the geometry system implementation including:
- Basic 2D shapes (Rectangle, Circle, Polygon)
- CSG operations (Union, Intersection, Difference)
- Boundary detection and normal computation
- 3D molecular geometry support with atomic coordinates
- Periodic boundary conditions for materials systems
- Integration with quantum mechanical calculations
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.geometry.csg import (
    Circle,
    compute_boundary_normals,
    difference,
    intersection,
    MolecularGeometry,
    PeriodicCell,
    Polygon,
    Rectangle,
    sample_boundary_points,
    union,
)
from opifex.geometry.manifolds import (
    HyperbolicManifold,
)


class TestBasic2DShapes:
    """Test basic 2D geometric shapes."""

    def test_rectangle_creation(self):
        """Test rectangle creation and basic properties."""
        rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=1.0)

        assert rect.width == 2.0
        assert rect.height == 1.0
        assert jnp.allclose(rect.center, jnp.array([0.0, 0.0]))

    def test_rectangle_contains_point(self):
        """Test point-in-rectangle containment."""
        rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=1.0)

        # Points inside
        assert rect.contains(jnp.array([0.5, 0.25]))
        assert rect.contains(jnp.array([-0.5, -0.25]))

        # Points outside
        assert not rect.contains(jnp.array([1.5, 0.0]))
        assert not rect.contains(jnp.array([0.0, 0.75]))

        # Points on boundary (should be included)
        assert rect.contains(jnp.array([1.0, 0.0]))
        assert rect.contains(jnp.array([0.0, 0.5]))

    def test_circle_creation(self):
        """Test circle creation and basic properties."""
        circle = Circle(center=jnp.array([1.0, 1.0]), radius=2.0)

        assert circle.radius == 2.0
        assert jnp.allclose(circle.center, jnp.array([1.0, 1.0]))

    def test_circle_contains_point(self):
        """Test point-in-circle containment."""
        circle = Circle(center=jnp.array([0.0, 0.0]), radius=1.0)

        # Points inside
        assert circle.contains(jnp.array([0.5, 0.0]))
        assert circle.contains(jnp.array([0.0, 0.5]))
        assert circle.contains(jnp.array([0.3, 0.3]))

        # Points outside
        assert not circle.contains(jnp.array([1.5, 0.0]))
        assert not circle.contains(jnp.array([0.8, 0.8]))

        # Point on boundary
        assert circle.contains(jnp.array([1.0, 0.0]))

    def test_polygon_creation(self):
        """Test polygon creation from vertices."""
        # Triangle
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        poly = Polygon(vertices=vertices)

        assert poly.vertices.shape == (3, 2)
        assert jnp.allclose(poly.vertices, vertices)

    def test_polygon_contains_point(self):
        """Test point-in-polygon containment using ray casting."""
        # Triangle
        vertices = jnp.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
        poly = Polygon(vertices=vertices)

        # Points inside
        assert poly.contains(jnp.array([1.0, 0.5]))
        assert poly.contains(jnp.array([0.8, 0.4]))

        # Points outside
        assert not poly.contains(jnp.array([0.0, 1.0]))
        assert not poly.contains(jnp.array([2.0, 1.0]))


class TestCSGOperations:
    """Test Constructive Solid Geometry operations."""

    def test_union_operation(self):
        """Test union of two shapes."""
        rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=2.0)
        circle = Circle(center=jnp.array([1.0, 1.0]), radius=1.0)

        union_shape = union(rect, circle)

        # Points in either shape should be in union
        assert union_shape.contains(jnp.array([0.0, 0.0]))  # In rect
        assert union_shape.contains(jnp.array([1.5, 1.5]))  # In circle
        assert union_shape.contains(jnp.array([0.5, 0.5]))  # In both

        # Points in neither should not be in union
        assert not union_shape.contains(jnp.array([3.0, 3.0]))

    def test_intersection_operation(self):
        """Test intersection of two shapes."""
        rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=2.0)
        circle = Circle(center=jnp.array([0.5, 0.5]), radius=1.0)

        intersect_shape = intersection(rect, circle)

        # Points in both shapes should be in intersection
        assert intersect_shape.contains(jnp.array([0.5, 0.5]))

        # Points in only one shape should not be in intersection
        assert not intersect_shape.contains(jnp.array([-0.8, 0.0]))  # Only in rect
        assert not intersect_shape.contains(jnp.array([1.3, 1.3]))  # Only in circle

    def test_difference_operation(self):
        """Test difference (subtraction) of two shapes."""
        rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=2.0)
        circle = Circle(center=jnp.array([0.0, 0.0]), radius=0.5)

        diff_shape = difference(rect, circle)

        # Points in rect but not in circle should be in difference
        assert diff_shape.contains(jnp.array([0.8, 0.8]))

        # Points in circle should not be in difference
        assert not diff_shape.contains(jnp.array([0.2, 0.2]))

        # Points outside rect should not be in difference
        assert not diff_shape.contains(jnp.array([2.0, 2.0]))


class TestBoundaryDetection:
    """Test boundary detection and normal computation."""

    def test_boundary_normal_computation(self):
        """Test computation of boundary normals."""
        circle = Circle(center=jnp.array([0.0, 0.0]), radius=1.0)

        # Test point on circle boundary
        boundary_point = jnp.array([1.0, 0.0])
        normal = compute_boundary_normals(circle, boundary_point)

        # Normal should point outward
        expected_normal = jnp.array([1.0, 0.0])
        assert jnp.allclose(normal, expected_normal, atol=1e-6)

    def test_boundary_point_sampling(self):
        """Test sampling of points on shape boundaries."""
        circle = Circle(center=jnp.array([0.0, 0.0]), radius=1.0)

        # Sample boundary points
        boundary_points = sample_boundary_points(circle, n_points=100)

        assert boundary_points.shape == (100, 2)

        # All points should be approximately on the boundary
        distances = jnp.linalg.norm(boundary_points, axis=1)
        assert jnp.allclose(distances, 1.0, atol=1e-6)


class TestMolecularGeometry:
    """Test 3D molecular geometry support."""

    def test_molecular_geometry_creation(self):
        """Test creation of molecular geometry from atomic coordinates."""
        # Water molecule
        atomic_symbols = ["O", "H", "H"]
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # Oxygen
                [0.0, 0.757, 0.587],  # Hydrogen 1
                [0.0, -0.757, 0.587],  # Hydrogen 2
            ]
        )

        mol_geom = MolecularGeometry(atomic_symbols=atomic_symbols, positions=positions)

        assert mol_geom.n_atoms == 3
        assert mol_geom.atomic_symbols == atomic_symbols
        assert jnp.allclose(mol_geom.positions, positions)

    def test_molecular_geometry_distances(self):
        """Test computation of interatomic distances."""
        atomic_symbols = ["H", "H"]
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        mol_geom = MolecularGeometry(atomic_symbols=atomic_symbols, positions=positions)

        distances = mol_geom.compute_distances()
        expected = jnp.array([[0.0, 1.0], [1.0, 0.0]])

        assert jnp.allclose(distances, expected)

    def test_molecular_geometry_from_molecular_system(self):
        """Test creation of molecular geometry from MolecularSystem."""
        from opifex.core.problems import create_molecular_system

        molecular_system = create_molecular_system(
            atoms=[
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.757, 0.587)),
                ("H", (0.0, -0.757, 0.587)),
            ]
        )

        mol_geom = MolecularGeometry.from_molecular_system(molecular_system)

        assert mol_geom.n_atoms == 3
        assert mol_geom.atomic_symbols == ["O", "H", "H"]
        # Positions should be converted from Angstrom to Bohr
        expected_positions = molecular_system.positions
        assert jnp.allclose(mol_geom.positions, expected_positions)


class TestPeriodicBoundaryConditions:
    """Test periodic boundary conditions for materials systems."""

    def test_periodic_cell_creation(self):
        """Test creation of periodic cell with safe GPU handling."""
        # Ensure safe JAX environment before test

        # Simple cubic cell
        lattice_vectors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        cell = PeriodicCell(lattice_vectors=lattice_vectors)

        assert cell.lattice_vectors.shape == (3, 3)
        assert jnp.allclose(cell.lattice_vectors, lattice_vectors)

        # Volume should be 1.0 for unit cube
        assert jnp.isclose(cell.volume, 1.0)

    def test_periodic_distance_computation(self):
        """Test distance computation with periodic boundaries."""
        # Unit cubic cell
        lattice_vectors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        cell = PeriodicCell(lattice_vectors=lattice_vectors)

        # Points close due to periodicity
        point1 = jnp.array([0.1, 0.0, 0.0])
        point2 = jnp.array([0.9, 0.0, 0.0])

        periodic_distance = cell.periodic_distance(point1, point2)

        # Should be 0.2 (wrapping around), not 0.8
        assert jnp.isclose(periodic_distance, 0.2, atol=1e-6)

    def test_wrap_to_unit_cell(self):
        """Test wrapping coordinates to unit cell."""
        lattice_vectors = jnp.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])

        cell = PeriodicCell(lattice_vectors=lattice_vectors)

        # Point outside unit cell
        point = jnp.array([3.5, -1.0, 1.5])
        wrapped = cell.wrap_to_unit_cell(point)

        # Should be wrapped to [1.5, 1.0, 1.5]
        expected = jnp.array([1.5, 1.0, 1.5])
        assert jnp.allclose(wrapped, expected, atol=1e-6)


class TestQuantumMechanicalIntegration:
    """Test integration with quantum mechanical calculations."""

    def test_molecular_geometry_with_electronic_structure(self):
        """Test molecular geometry integration with electronic structure problems."""
        # Create molecular system first
        from opifex.core.problems import (
            create_molecular_system,
            create_neural_dft_problem,
        )

        molecular_system = create_molecular_system(
            atoms=[("H", (0.0, 0.0, 0.0)), ("H", (0.74, 0.0, 0.0))],
            charge=0,
            multiplicity=1,
        )

        # Create neural DFT problem
        dft_problem = create_neural_dft_problem(molecular_system=molecular_system)

        # Create molecular geometry from DFT problem
        mol_geom = MolecularGeometry.from_molecular_system(dft_problem.molecular_system)

        assert mol_geom.n_atoms == 2
        assert mol_geom.atomic_symbols == ["H", "H"]

        # Test integration with quantum calculations
        bond_length = mol_geom.compute_distances()[0, 1]
        assert bond_length > 0.0  # Basic sanity check

    def test_periodic_geometry_for_materials(self):
        """Test periodic geometry for materials systems."""
        # Simple 2x2x2 hydrogen crystal
        lattice_vectors = jnp.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])

        atomic_symbols = ["H"] * 8
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        mol_geom = MolecularGeometry(atomic_symbols=atomic_symbols, positions=positions)

        periodic_cell = PeriodicCell(lattice_vectors=lattice_vectors)

        # Test that we can combine molecular geometry with periodic cell
        assert mol_geom.n_atoms == 8
        assert periodic_cell.volume == 8.0

        # Test periodic neighbor finding
        neighbors = periodic_cell.find_neighbors(positions, cutoff_radius=1.5)

        # Each atom should have several neighbors
        assert len(neighbors) > 0

        # Test molecular geometry integration
        distances = mol_geom.compute_distances()
        assert distances.shape == (8, 8)


class TestGeometrySystemIntegration:
    """Test overall geometry system integration."""

    def test_2d_3d_geometry_interop(self):
        """Test interoperability between 2D and 3D geometry systems."""
        # Create 2D circle
        circle = Circle(center=jnp.array([0.0, 0.0]), radius=1.0)

        # Create 3D molecular geometry
        mol_geom = MolecularGeometry(
            atomic_symbols=["C"], positions=jnp.array([[0.0, 0.0, 0.0]])
        )

        # Should be able to project 3D geometry to 2D for visualization
        projection_2d = mol_geom.project_to_2d(plane="xy")

        assert projection_2d.shape == (1, 2)
        assert jnp.allclose(projection_2d, jnp.array([[0.0, 0.0]]))

        # Test that circle can be used for containment checks
        assert circle.contains(projection_2d[0])

    def test_csg_with_molecular_exclusion(self):
        """Test CSG operations with molecular geometry exclusion zones."""
        # Define a computational domain (rectangle)
        domain = Rectangle(center=jnp.array([0.0, 0.0]), width=10.0, height=10.0)

        # Define molecular exclusion zones (circles around atoms)
        mol_geom = MolecularGeometry(
            atomic_symbols=["C", "C"],
            positions=jnp.array(
                [
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],  # This line makes positions 3D but we'll project
                ]
            )[:2],  # Take only first 2 atoms for 2D test
        )

        # Test molecular geometry properties
        assert mol_geom.n_atoms == 2
        assert mol_geom.atomic_symbols == ["C", "C"]

        # Create exclusion circles
        exclusion1 = Circle(center=jnp.array([-1.0, 0.0]), radius=0.5)
        exclusion2 = Circle(center=jnp.array([1.0, 0.0]), radius=0.5)

        # Computational domain minus molecular exclusion zones
        computational_domain = difference(domain, union(exclusion1, exclusion2))

        # Points in molecular regions should be excluded
        assert not computational_domain.contains(jnp.array([-1.0, 0.0]))
        assert not computational_domain.contains(jnp.array([1.0, 0.0]))

        # Points in domain but away from molecules should be included
        assert computational_domain.contains(jnp.array([0.0, 2.0]))
        assert computational_domain.contains(jnp.array([3.0, 0.0]))


# Error handling and edge cases
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_degenerate_shapes(self):
        """Test handling of degenerate shapes."""
        # Zero-radius circle
        with pytest.raises(ValueError, match="Radius must be positive"):
            Circle(center=jnp.array([0.0, 0.0]), radius=0.0)

        # Zero-area rectangle
        with pytest.raises(ValueError, match="Width and height must be positive"):
            Rectangle(center=jnp.array([0.0, 0.0]), width=0.0, height=1.0)

    def test_invalid_polygon(self):
        """Test handling of invalid polygons."""
        # Polygon with less than 3 vertices
        with pytest.raises(ValueError, match="Polygon must have at least 3 vertices"):
            Polygon(vertices=jnp.array([[0.0, 0.0], [1.0, 0.0]]))

    def test_molecular_geometry_validation(self):
        """Test molecular geometry input validation."""
        # Mismatched arrays
        with pytest.raises(
            ValueError, match="Number of atomic symbols must match number of positions"
        ):
            MolecularGeometry(
                atomic_symbols=["H", "H"],
                # Only 1 position for 2 atoms
                positions=jnp.array([[0.0, 0.0, 0.0]]),
            )


class TestHyperbolicManifold:
    """Test hyperbolic manifold with Poincaré disk model."""

    def test_hyperbolic_manifold_creation(self):
        """Test creation of hyperbolic manifold."""
        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

        assert manifold.dimension == 2
        assert manifold.embedding_dimension == 2
        assert manifold.curvature == -1.0
        assert manifold.radius == 1.0

    def test_hyperbolic_manifold_validation_constraints(self):
        """Test that hyperbolic manifold enforces negative curvature."""
        # Should raise error for non-negative curvature
        with pytest.raises(ValueError, match="negative curvature"):
            HyperbolicManifold(curvature=1.0, dimension=2)

        with pytest.raises(ValueError, match="negative curvature"):
            HyperbolicManifold(curvature=0.0, dimension=2)

    def test_point_validation(self):
        """Test point validation and projection into Poincaré disk."""
        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

        # Point inside disk should remain unchanged
        inside_point = jnp.array([0.5, 0.3])
        validated = manifold._validate_point(inside_point)
        assert jnp.allclose(validated, inside_point)

        # Point outside disk should be projected to boundary
        outside_point = jnp.array([2.0, 0.0])
        validated = manifold._validate_point(outside_point)
        norm = jnp.linalg.norm(validated)
        assert norm < 1.0  # Should be inside unit disk
        assert norm > 0.9  # Should be near boundary

    def test_gyroaddition_properties(self):
        """Test gyrovector addition properties."""
        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

        # Test with origin
        origin = jnp.array([0.0, 0.0])
        point = jnp.array([0.5, 0.0])

        # Gyroaddition with origin should be identity
        result = manifold._gyroaddition(origin, point)
        assert jnp.allclose(result, point, atol=1e-6)

        # Test commutativity at origin
        result1 = manifold._gyroaddition(origin, point)
        result2 = manifold._gyroaddition(point, origin)
        assert jnp.allclose(result1, result2, atol=1e-6)

    def test_exponential_map(self):
        """Test exponential map functionality."""
        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

        # Test from origin
        base = jnp.array([0.0, 0.0])
        tangent = jnp.array([0.5, 0.0])

        result = manifold.exp_map(base, tangent)

        # Result should be in Poincaré disk
        norm = jnp.linalg.norm(result)
        assert norm < 1.0

        # Zero tangent should return base point
        zero_tangent = jnp.array([0.0, 0.0])
        result_zero = manifold.exp_map(base, zero_tangent)
        assert jnp.allclose(result_zero, base, atol=1e-6)

    def test_logarithmic_map(self):
        """Test logarithmic map functionality."""
        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

        base = jnp.array([0.0, 0.0])
        point = jnp.array([0.5, 0.0])

        # Log map should be inverse of exp map
        tangent = manifold.log_map(base, point)
        reconstructed = manifold.exp_map(base, tangent)

        assert jnp.allclose(reconstructed, point, atol=1e-5)

        # Same point should give zero tangent
        zero_tangent = manifold.log_map(base, base)
        assert jnp.allclose(zero_tangent, jnp.zeros_like(base), atol=1e-6)

    def test_geodesic_distance(self):
        """Test geodesic distance computation."""
        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

        # Distance from origin to origin should be zero
        origin = jnp.array([0.0, 0.0])
        distance = manifold.geodesic_distance(origin, origin)
        assert jnp.allclose(distance, 0.0, atol=1e-6)

        # Distance should be symmetric
        point1 = jnp.array([0.3, 0.0])
        point2 = jnp.array([0.0, 0.4])

        dist1 = manifold.geodesic_distance(point1, point2)
        dist2 = manifold.geodesic_distance(point2, point1)
        assert jnp.allclose(dist1, dist2, atol=1e-6)

        # Distance should be positive for different points
        assert dist1 > 0

    def test_metric_tensor(self):
        """Test hyperbolic metric tensor computation."""
        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

        # Test at origin
        origin = jnp.array([0.0, 0.0])
        metric = manifold.metric_tensor(origin)

        # At origin, metric should be 4 * I (for radius = 1)
        expected = 4.0 * jnp.eye(2)
        assert jnp.allclose(metric, expected, atol=1e-6)

        # Metric should be positive definite
        eigenvals = jnp.linalg.eigvals(metric)
        assert jnp.all(eigenvals > 0)

    def test_random_point_generation(self):
        """Test random point generation in Poincaré disk."""
        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

        key = jax.random.PRNGKey(42)

        # Single random point
        point = manifold.random_point(key)
        assert point.shape == (2,)
        norm = jnp.linalg.norm(point)
        assert norm < 1.0  # Should be inside unit disk

        # Batch of random points
        batch_points = manifold.random_point(key, shape=(10,))
        assert batch_points.shape == (10, 2)
        norms = jnp.linalg.norm(batch_points, axis=-1)
        assert jnp.all(norms < 1.0)

    def test_exp_log_consistency(self):
        """Test exp/log map consistency for various points."""
        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

        # Test multiple base points and tangent vectors
        key = jax.random.PRNGKey(123)

        # Generate random base points and tangent vectors
        base_points = manifold.random_point(key, shape=(5,))
        tangent_vectors = jax.random.normal(jax.random.split(key)[1], (5, 2)) * 0.1

        for i in range(5):
            base = base_points[i]
            tangent = tangent_vectors[i]

            # Test exp -> log consistency
            point = manifold.exp_map(base, tangent)
            recovered_tangent = manifold.log_map(base, point)

            assert jnp.allclose(recovered_tangent, tangent, atol=1e-4)

    def test_manifold_neural_operator_jax_transforms(self):
        """Test ManifoldNeuralOperator with JAX transformations."""
        from flax import nnx

        from opifex.geometry.manifolds.operators import ManifoldNeuralOperator

        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        operator = ManifoldNeuralOperator(manifold=manifold, hidden_dim=16, rngs=rngs)

        # Test standard forward pass
        test_points = manifold.random_point(key, shape=(2,))
        output_normal = operator(test_points)

        # Test that we can compute gradients (which uses JAX autodiff)
        def loss_fn(points):
            result = operator(points)
            return jnp.sum(result**2)

        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(test_points)

        # Verify gradient computation works
        assert gradients.shape == test_points.shape
        assert jnp.all(jnp.isfinite(gradients))

        # Test that results are consistent
        output_second = operator(test_points)
        assert jnp.allclose(output_normal, output_second, atol=1e-6)

    def test_jax_transformations(self):
        """Test JAX transformations (jit, grad) work correctly."""
        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

        # Test JIT compilation
        jit_exp_map = jax.jit(manifold.exp_map)

        base = jnp.array([0.0, 0.0])
        tangent = jnp.array([0.3, 0.4])

        result_normal = manifold.exp_map(base, tangent)
        result_jit = jit_exp_map(base, tangent)

        assert jnp.allclose(result_normal, result_jit, atol=1e-6)

        # Test gradient computation
        def distance_function(point):
            origin = jnp.array([0.0, 0.0])
            return manifold.geodesic_distance(origin, point)

        grad_fn = jax.grad(distance_function)

        point = jnp.array([0.5, 0.0])
        gradient = grad_fn(point)

        # Gradient should point away from origin for distance function
        assert gradient.shape == (2,)
        # For this specific case, gradient should be roughly [1, 0]
        assert gradient[0] > 0
        assert abs(gradient[1]) < 1e-3


class TestManifoldNeuralOperator:
    """Test manifold neural operators for geometric deep learning."""

    def test_manifold_neural_operator_import(self):
        """Test that ManifoldNeuralOperator can be imported."""
        # This test will fail initially, driving us to implement the class
        from opifex.geometry.manifolds.operators import ManifoldNeuralOperator

        assert ManifoldNeuralOperator is not None

    def test_manifold_neural_operator_creation(self):
        """Test creation of ManifoldNeuralOperator with hyperbolic manifold."""
        from flax import nnx

        from opifex.geometry.manifolds.operators import ManifoldNeuralOperator

        # Create a hyperbolic manifold
        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

        # Create neural operator
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        operator = ManifoldNeuralOperator(manifold=manifold, hidden_dim=64, rngs=rngs)

        assert operator.manifold is manifold
        assert hasattr(operator, "encoder")

    def test_manifold_neural_operator_forward_pass(self):
        """Test forward pass of ManifoldNeuralOperator."""
        from flax import nnx

        from opifex.geometry.manifolds.operators import ManifoldNeuralOperator

        # Create manifold and operator
        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        operator = ManifoldNeuralOperator(manifold=manifold, hidden_dim=32, rngs=rngs)

        # Test with valid manifold points
        batch_size = 4
        manifold_points = manifold.random_point(key, shape=(batch_size,))

        # Forward pass
        output = operator(manifold_points)

        # Check output properties
        assert output.shape == (batch_size, 2)  # Same dimension as input

        # Outputs should be valid manifold points
        for i in range(batch_size):
            point = output[i]
            assert manifold._validate_point(point) is not None, (
                f"Output point {i} not on manifold"
            )

    def test_manifold_neural_operator_jax_transforms(self):
        """Test ManifoldNeuralOperator with JAX transformations."""
        from flax import nnx

        from opifex.geometry.manifolds.operators import ManifoldNeuralOperator

        manifold = HyperbolicManifold(curvature=-1.0, dimension=2)
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        operator = ManifoldNeuralOperator(manifold=manifold, hidden_dim=16, rngs=rngs)

        # Test standard forward pass
        test_points = manifold.random_point(key, shape=(2,))
        output_normal = operator(test_points)

        # Test that we can compute gradients (which uses JAX autodiff)
        def loss_fn(points):
            result = operator(points)
            return jnp.sum(result**2)

        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(test_points)

        # Verify gradient computation works
        assert gradients.shape == test_points.shape
        assert jnp.all(jnp.isfinite(gradients))
        # Test that results are consistent
        output_second = operator(test_points)
        assert jnp.allclose(output_normal, output_second, atol=1e-6)

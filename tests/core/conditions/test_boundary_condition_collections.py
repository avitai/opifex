"""
Tests for Boundary Condition Collections

Tests for BoundaryConditionCollection functionality.
"""

from opifex.core.conditions import (
    BoundaryCondition,
    BoundaryConditionCollection,
    DirichletBC,
    NeumannBC,
    RobinBC,
    WavefunctionBC,
)


class TestBoundaryConditionCollection:
    """Test boundary condition collections."""

    def test_initialization(self):
        """Test boundary condition collection initialization."""
        bc1 = DirichletBC(boundary="left", value=1.0)
        bc2 = NeumannBC(boundary="right", value=0.0)
        bc3 = RobinBC(boundary="top", alpha=1.0, beta=1.0, gamma=0.0)

        collection = BoundaryConditionCollection([bc1, bc2, bc3])

        assert len(collection) == 3
        assert collection.get_boundary_condition("left") == bc1
        assert collection.get_boundary_condition("right") == bc2
        assert collection.get_boundary_condition("top") == bc3

    def test_length(self):
        """Test collection length."""
        bc1 = DirichletBC(boundary="left", value=1.0)
        bc2 = NeumannBC(boundary="right", value=0.0)

        collection = BoundaryConditionCollection([bc1, bc2])
        assert len(collection) == 2

    def test_validate_all_valid(self):
        """Test validation when all boundary conditions are valid."""
        bc1 = DirichletBC(boundary="left", value=1.0)
        bc2 = NeumannBC(boundary="right", value=0.0)

        collection = BoundaryConditionCollection([bc1, bc2])
        assert collection.validate() is True

    def test_get_boundary_condition_exists(self):
        """Test getting existing boundary condition."""
        bc1 = DirichletBC(boundary="left", value=1.0)
        bc2 = NeumannBC(boundary="right", value=0.0)

        collection = BoundaryConditionCollection([bc1, bc2])

        result = collection.get_boundary_condition("left")
        assert result == bc1

    def test_get_boundary_condition_not_exists(self):
        """Test getting non-existent boundary condition."""
        bc1 = DirichletBC(boundary="left", value=1.0)

        collection = BoundaryConditionCollection([bc1])

        result = collection.get_boundary_condition("right")
        assert result is None

    def test_get_by_type(self):
        """Test getting boundary conditions by type."""
        bc1 = DirichletBC(boundary="left", value=1.0)
        bc2 = DirichletBC(boundary="right", value=2.0)
        bc3 = NeumannBC(boundary="top", value=0.0)

        collection = BoundaryConditionCollection([bc1, bc2, bc3])

        dirichlet_bcs = collection.get_by_type("dirichlet")
        assert len(dirichlet_bcs) == 2
        assert bc1 in dirichlet_bcs
        assert bc2 in dirichlet_bcs

    def test_add_condition(self):
        """Test adding boundary condition to collection."""
        bc1 = DirichletBC(boundary="left", value=1.0)
        collection = BoundaryConditionCollection([bc1])

        bc2 = NeumannBC(boundary="right", value=0.0)
        collection.add_condition(bc2)

        assert len(collection) == 2
        assert collection.get_boundary_condition("right") == bc2

    def test_remove_condition_exists(self):
        """Test removing existing boundary condition."""
        bc1 = DirichletBC(boundary="left", value=1.0)
        bc2 = NeumannBC(boundary="right", value=0.0)

        collection = BoundaryConditionCollection([bc1, bc2])

        result = collection.remove_condition("left")
        assert result is True
        assert len(collection) == 1
        assert collection.get_boundary_condition("left") is None

    def test_remove_condition_not_exists(self):
        """Test removing non-existent boundary condition."""
        bc1 = DirichletBC(boundary="left", value=1.0)

        collection = BoundaryConditionCollection([bc1])

        result = collection.remove_condition("right")
        assert result is False
        assert len(collection) == 1

    def test_comprehensive_collection(self):
        """Test BoundaryConditionCollection with comprehensive scenarios."""
        # Create diverse boundary conditions
        bc1 = DirichletBC(boundary="left", value=1.0)
        bc2 = NeumannBC(boundary="right", value=0.0)
        bc3 = RobinBC(boundary="top", alpha=1.0, beta=1.0, gamma=0.0)
        bc4 = WavefunctionBC(condition_type="vanishing", boundary="bottom")

        collection = BoundaryConditionCollection([bc1, bc2, bc3, bc4])

        # Test length
        assert len(collection) == 4

        # Test validation (all should be valid)
        assert collection.validate() is True

        # Test get_boundary_condition
        assert collection.get_boundary_condition("left") == bc1
        assert collection.get_boundary_condition("nonexistent") is None

        # Test get_by_type with various type specifications
        dirichlet_bcs = collection.get_by_type("dirichlet")
        assert len(dirichlet_bcs) == 1
        assert dirichlet_bcs[0] == bc1

        neumann_bcs = collection.get_by_type("neumann")
        assert len(neumann_bcs) == 1
        assert neumann_bcs[0] == bc2

        robin_bcs = collection.get_by_type("robin")
        assert len(robin_bcs) == 1
        assert robin_bcs[0] == bc3

        # Test class name matching
        dirichlet_class = collection.get_by_type("DirichletBC")
        assert len(dirichlet_class) == 1

        # Test add_condition
        bc5 = DirichletBC(boundary="front", value=2.0)
        collection.add_condition(bc5)
        assert len(collection) == 5
        assert collection.get_boundary_condition("front") == bc5

        # Test remove_condition
        assert collection.remove_condition("front") is True
        assert len(collection) == 4
        assert collection.get_boundary_condition("front") is None

        # Test remove non-existent condition
        assert collection.remove_condition("nonexistent") is False

    def test_validation_edge_cases(self):
        """Test BoundaryConditionCollection validation edge cases."""

        # Test with invalid boundary condition
        class InvalidBC(BoundaryCondition):
            def validate(self):
                return False

            def evaluate(self, x, t=0.0):
                return x

        valid_bc = DirichletBC(boundary="left", value=1.0)
        invalid_bc = InvalidBC(boundary="right")

        # Collection with invalid BC should fail validation
        collection_invalid = BoundaryConditionCollection([valid_bc, invalid_bc])
        assert collection_invalid.validate() is False

        # Test empty collection
        collection_empty = BoundaryConditionCollection([])
        assert collection_empty.validate() is True  # Empty collection is valid
        assert len(collection_empty) == 0

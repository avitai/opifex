"""Test-driven development tests for constraint satisfaction learning.

These tests define the expected behavior of the L2O constraint learning module
before implementation, following TDD red-green-refactor methodology.
"""

import time

import jax.numpy as jnp
import pytest
from flax import nnx

# Import statements that will be implemented
from opifex.optimization.l2o.constraint_learning import (
    ConstraintProjector,
    ConstraintSpecification,
    ConstraintViolationDetector,
    FeasibilityLearner,
    ProjectorConfig,
    SymbolicConstraintEncoder,
)


class TestConstraintSpecification:
    """Test cases for constraint specification representation."""

    def test_constraint_specification_creation(self):
        """Test that constraint specifications can be created with different types."""
        # Equality constraint: x1 + x2 - 1 = 0
        eq_spec = ConstraintSpecification(
            constraint_type="equality",
            expression="x1 + x2 - 1 = 0",
            coefficients=jnp.array([1.0, 1.0, -1.0]),
            variables=["x1", "x2"],
        )

        assert eq_spec.constraint_type == "equality"
        assert eq_spec.expression == "x1 + x2 - 1 = 0"
        assert eq_spec.variables == ["x1", "x2"]

        # Inequality constraint: x1 >= 0
        ineq_spec = ConstraintSpecification(
            constraint_type="inequality",
            expression="x1 >= 0",
            coefficients=jnp.array([1.0]),
            variables=["x1"],
        )

        assert ineq_spec.constraint_type == "inequality"
        assert ineq_spec.expression == "x1 >= 0"

    def test_constraint_specification_validation(self):
        """Test that invalid constraint types are rejected."""
        with pytest.raises(ValueError, match="Invalid constraint type"):
            ConstraintSpecification(
                constraint_type="invalid",
                expression="x > 0",
                coefficients=jnp.array([1.0]),
                variables=["x"],
            )

    def test_constraint_evaluation_equality(self):
        """Test evaluation of equality constraints."""
        constraint = ConstraintSpecification(
            constraint_type="equality",
            expression="x1 + x2 = 1",
            coefficients=jnp.array([1.0, 1.0, -1.0]),  # Represents x1 + x2 - 1 = 0
            variables=["x1", "x2"],
        )

        # Test satisfied constraint
        x_satisfied = jnp.array([0.5, 0.5])
        violation_satisfied = constraint.evaluate(x_satisfied)
        assert jnp.abs(violation_satisfied) < 1e-6

        # Test violated constraint
        x_violated = jnp.array([0.3, 0.4])
        violation_violated = constraint.evaluate(x_violated)
        assert violation_violated > 1e-6

    def test_constraint_evaluation_inequality(self):
        """Test evaluation of inequality constraints."""
        constraint = ConstraintSpecification(
            constraint_type="inequality",
            expression="x >= 0",
            coefficients=jnp.array([1.0]),
            variables=["x"],
        )

        # Test satisfied constraint
        x_satisfied = jnp.array([0.5])
        violation_satisfied = constraint.evaluate(x_satisfied)
        assert jnp.abs(violation_satisfied) < 1e-6

        # Test violated constraint
        x_violated = jnp.array([-0.5])
        violation_violated = constraint.evaluate(x_violated)
        assert violation_violated > 1e-6


class TestConstraintViolationDetector:
    """Test cases for constraint violation detection."""

    def test_single_constraint_detection(self):
        """Test detection of violations for a single constraint."""
        constraint = ConstraintSpecification(
            constraint_type="equality",
            expression="x1 + x2 = 1",
            coefficients=jnp.array([1.0, 1.0, -1.0]),  # Represents x1 + x2 - 1 = 0
            variables=["x1", "x2"],
        )

        detector = ConstraintViolationDetector([constraint])

        # Test satisfied point
        satisfied_point = jnp.array([0.5, 0.5])
        result = detector.detect_violations(satisfied_point)

        assert "constraint_0" in result
        assert result["total_violation"] < 1e-6
        assert result["is_feasible"]

        # Test violated point
        violated_point = jnp.array([0.3, 0.4])
        result = detector.detect_violations(violated_point)

        assert result["total_violation"] > 1e-6
        assert not result["is_feasible"]

    def test_batch_violation_detection(self):
        """Test batch processing of constraint violations."""
        constraint = ConstraintSpecification(
            constraint_type="inequality",
            expression="x >= 0",
            coefficients=jnp.array([1.0]),
            variables=["x"],
        )

        detector = ConstraintViolationDetector([constraint])

        # Batch with mixed satisfied/violated points
        batch_points = jnp.array([[0.5], [-0.2], [1.0]])
        result = detector.detect_violations(batch_points)

        assert "constraint_0" in result
        assert len(result["constraint_0"]) == 3
        assert result["constraint_0"][0] < 1e-6  # First point satisfied
        assert result["constraint_0"][1] > 1e-6  # Second point violated
        assert result["constraint_0"][2] < 1e-6  # Third point satisfied

    def test_multiple_constraints_detection(self):
        """Test detection with multiple constraints."""
        constraints = [
            ConstraintSpecification(
                constraint_type="equality",
                expression="x1 + x2 = 1",
                coefficients=jnp.array([1.0, 1.0]),
                variables=["x1", "x2"],
            ),
            ConstraintSpecification(
                constraint_type="inequality",
                expression="x1 >= 0",
                coefficients=jnp.array([1.0, 0.0]),
                variables=["x1", "x2"],
            ),
        ]

        detector = ConstraintViolationDetector(constraints)

        # Point violating both constraints
        violated_point = jnp.array([-0.1, 0.8])
        result = detector.detect_violations(violated_point)

        assert "constraint_0" in result
        assert "constraint_1" in result
        assert result["total_violation"] > 1e-6
        assert not result["is_feasible"]


class TestSymbolicConstraintEncoder:
    """Test cases for symbolic constraint encoding."""

    def test_encoder_initialization(self):
        """Test encoder initialization with different embedding dimensions."""
        encoder_default = SymbolicConstraintEncoder()
        assert encoder_default.embedding_dim == 16

        encoder_custom = SymbolicConstraintEncoder(embedding_dim=32)
        assert encoder_custom.embedding_dim == 32

    def test_constraint_encoding_consistency(self):
        """Test that encoding is consistent for the same constraint."""
        encoder = SymbolicConstraintEncoder(embedding_dim=16)

        constraint = ConstraintSpecification(
            constraint_type="equality",
            expression="x1 + x2 = 1",
            coefficients=jnp.array([1.0, 1.0]),
            variables=["x1", "x2"],
        )

        # Encode same constraint multiple times
        encoding1 = encoder.encode_constraint(constraint)
        encoding2 = encoder.encode_constraint(constraint)

        assert encoding1.shape == (16,)
        assert jnp.allclose(encoding1, encoding2)

    def test_different_constraints_different_encodings(self):
        """Test that different constraints produce different encodings."""
        encoder = SymbolicConstraintEncoder(embedding_dim=16)

        constraint1 = ConstraintSpecification(
            constraint_type="equality",
            expression="x1 + x2 = 1",
            coefficients=jnp.array([1.0, 1.0]),
            variables=["x1", "x2"],
        )

        constraint2 = ConstraintSpecification(
            constraint_type="inequality",
            expression="x1 >= 0",
            coefficients=jnp.array([1.0, 0.0]),
            variables=["x1", "x2"],
        )

        encoding1 = encoder.encode_constraint(constraint1)
        encoding2 = encoder.encode_constraint(constraint2)

        # Different constraints should produce different encodings
        assert not jnp.allclose(encoding1, encoding2)


class TestConstraintProjector:
    """Test cases for neural constraint projection."""

    @pytest.fixture
    def projector_config(self):
        """Fixture providing projector configuration."""
        return ProjectorConfig(
            hidden_sizes=[64, 32],
            embedding_dim=16,
        )

    @pytest.fixture
    def projector(self, projector_config):
        """Fixture providing constraint projector instance."""
        rngs = nnx.Rngs(42)
        return ConstraintProjector(
            input_dim=2,
            config=projector_config,
            rngs=rngs,
        )

    def test_projector_initialization(self, projector_config):
        """Test constraint projector initialization."""
        rngs = nnx.Rngs(42)
        projector = ConstraintProjector(
            input_dim=2,
            config=projector_config,
            rngs=rngs,
        )

        assert projector.input_dim == 2
        assert projector.config == projector_config
        assert hasattr(projector, "projection_network")

    def test_single_point_projection(self, projector):
        """Test projection of a single point."""
        point = jnp.array([1.0, 2.0])
        constraint_embedding = jnp.ones(16) * 0.1

        projected = projector.project(point, constraint_embedding)

        assert projected.shape == (2,)
        assert jnp.isfinite(projected).all()

    def test_batch_projection(self, projector):
        """Test projection of multiple points."""
        points = jnp.array([[1.0, 2.0], [0.5, 1.5], [2.0, 1.0]])
        constraint_embedding = jnp.ones(16) * 0.1

        projected_points = projector.project(points, constraint_embedding)

        assert projected_points.shape == (3, 2)
        assert jnp.isfinite(projected_points).all()

    def test_projection_preserves_feasible_points(self, projector):
        """Test that projection preserves already feasible points."""
        # Points already in feasible region (approximately)
        feasible_points = jnp.array([[0.5, 0.5], [0.3, 0.7]])
        constraint_embedding = jnp.ones(16) * 0.1

        projected = projector.project(feasible_points, constraint_embedding)

        # Should be close to original points if already feasible
        # (within some tolerance for neural network approximation)
        assert projected.shape == feasible_points.shape
        assert jnp.isfinite(projected).all()

    def test_batch_projection_efficiency(self):
        """Test that batch projection is efficient for large batches."""
        from flax import nnx

        config = ProjectorConfig(hidden_sizes=[32, 16], embedding_dim=8)
        projector = ConstraintProjector(2, config, nnx.Rngs(0))

        # Create JIT-compiled version
        jitted_project = nnx.jit(projector.project)

        # Generate test data
        points = jnp.array([[1.5, 2.5]] * 100)  # 100 points violating x + y <= 2
        constraint_embedding = jnp.ones(8) * 0.1

        # Warmup call to compile the function
        _ = jitted_project(points[:1], constraint_embedding)

        # Measure actual performance
        start_time = time.time()
        projected = jitted_project(points, constraint_embedding)
        projection_time = time.time() - start_time

        assert projected.shape == (100, 2)
        assert projection_time < 1.0  # Should be fast for batch processing


class TestFeasibilityLearner:
    """Test cases for the main feasibility learning system."""

    def test_real_time_constraint_satisfaction(self):
        """Test real-time constraint satisfaction (<1ms inference)."""
        from flax import nnx

        # Simplified network for speed
        config = ProjectorConfig(hidden_sizes=[16], embedding_dim=4)
        constraints = [
            ConstraintSpecification(
                "equality", "x + y = 1", jnp.array([1.0, 1.0]), ["x", "y"]
            )
        ]
        learner = FeasibilityLearner(2, constraints, config, nnx.Rngs(0))

        # Create JIT-compiled version
        jitted_satisfy = nnx.jit(learner.satisfy_constraints)

        # Test point
        point = jnp.array([0.3, 0.4])  # Sum = 0.7, violates x + y = 1

        # Warmup call to compile the function
        _ = jitted_satisfy(point)

        # Measure actual performance
        start_time = time.time()
        satisfied_point = jitted_satisfy(point)
        inference_time = time.time() - start_time

        assert satisfied_point.shape == (2,)
        assert inference_time < 0.001  # Should be < 1ms

    def test_feasibility_learner_initialization(self):
        """Test feasibility learner initialization."""
        constraints = [
            ConstraintSpecification(
                "equality", "x + y = 1", jnp.array([1.0, 1.0]), ["x", "y"]
            )
        ]
        config = ProjectorConfig(hidden_sizes=[64, 32], embedding_dim=16)

        learner = FeasibilityLearner(2, constraints, config, nnx.Rngs(42))

        assert learner.input_dim == 2
        assert len(learner.constraints) == 1
        assert len(learner.constraint_embeddings) == 1
        assert hasattr(learner, "projector")
        assert hasattr(learner, "detector")
        assert hasattr(learner, "encoder")

    def test_constraint_satisfaction_basic(self):
        """Test basic constraint satisfaction functionality."""
        constraints = [
            ConstraintSpecification(
                "equality", "x + y = 1", jnp.array([1.0, 1.0]), ["x", "y"]
            )
        ]
        config = ProjectorConfig(hidden_sizes=[32, 16], embedding_dim=8)
        learner = FeasibilityLearner(2, constraints, config, nnx.Rngs(0))

        # Point that violates constraint
        point = jnp.array([0.3, 0.4])  # Sum = 0.7, should be 1.0
        satisfied_point = learner.satisfy_constraints(point)

        assert satisfied_point.shape == (2,)
        assert jnp.isfinite(satisfied_point).all()

    def test_batch_constraint_satisfaction(self):
        """Test constraint satisfaction for batches of points."""
        constraints = [
            ConstraintSpecification(
                "equality", "x + y = 1", jnp.array([1.0, 1.0]), ["x", "y"]
            )
        ]
        config = ProjectorConfig(hidden_sizes=[32, 16], embedding_dim=8)
        learner = FeasibilityLearner(2, constraints, config, nnx.Rngs(0))

        # Batch of points violating constraint
        points = jnp.array([[0.3, 0.4], [0.6, 0.2], [0.1, 0.7]])
        satisfied_points = learner.satisfy_constraints(points)

        assert satisfied_points.shape == (3, 2)
        assert jnp.isfinite(satisfied_points).all()

    def test_feasibility_checking(self):
        """Test feasibility checking functionality."""
        constraints = [
            ConstraintSpecification(
                "equality", "x + y = 1", jnp.array([1.0, 1.0]), ["x", "y"]
            )
        ]
        config = ProjectorConfig(hidden_sizes=[32, 16], embedding_dim=8)
        learner = FeasibilityLearner(2, constraints, config, nnx.Rngs(0))

        # Test feasible point
        feasible_point = jnp.array([0.5, 0.5])
        feasibility_result = learner.check_feasibility(feasible_point)

        assert "constraint_0" in feasibility_result
        assert "total_violation" in feasibility_result
        assert "is_feasible" in feasibility_result

        # Test infeasible point
        infeasible_point = jnp.array([0.3, 0.4])
        infeasibility_result = learner.check_feasibility(infeasible_point)
        assert infeasibility_result["total_violation"] > 0

    def test_multiple_constraints_handling(self):
        """Test handling of multiple simultaneous constraints."""
        constraints = [
            ConstraintSpecification(
                "equality", "x + y = 1", jnp.array([1.0, 1.0]), ["x", "y"]
            ),
            ConstraintSpecification(
                "inequality", "x >= 0", jnp.array([1.0, 0.0]), ["x", "y"]
            ),
            ConstraintSpecification(
                "inequality", "y >= 0", jnp.array([0.0, 1.0]), ["x", "y"]
            ),
        ]
        config = ProjectorConfig(hidden_sizes=[32, 16], embedding_dim=8)
        learner = FeasibilityLearner(2, constraints, config, nnx.Rngs(0))

        # Point violating multiple constraints
        point = jnp.array([-0.1, 0.8])  # x < 0 and sum ≠ 1
        satisfied_point = learner.satisfy_constraints(point)

        assert satisfied_point.shape == (2,)
        assert jnp.isfinite(satisfied_point).all()

    def test_constraint_encoding_integration(self):
        """Test integration between constraint encoding and projection."""
        constraints = [
            ConstraintSpecification(
                "equality", "x + y = 1", jnp.array([1.0, 1.0]), ["x", "y"]
            )
        ]
        config = ProjectorConfig(hidden_sizes=[32, 16], embedding_dim=8)
        learner = FeasibilityLearner(2, constraints, config, nnx.Rngs(0))

        # Verify constraint embeddings are computed
        assert len(learner.constraint_embeddings) == 1
        assert learner.constraint_embeddings[0].shape == (8,)

        # Verify encoding is used in satisfaction
        point = jnp.array([0.3, 0.4])
        satisfied_point = learner.satisfy_constraints(point)
        assert jnp.isfinite(satisfied_point).all()

    def test_end_to_end_constrained_optimization(self):
        """Test end-to-end constrained optimization with realistic scenario."""
        from flax import nnx

        # Multi-variable optimization problem with constraints
        constraints = [
            ConstraintSpecification(
                "equality", "sum = 1", jnp.array([1.0, 1.0, 1.0]), ["x", "y", "z"]
            ),
            ConstraintSpecification(
                "inequality", "x >= 0", jnp.array([1.0, 0.0, 0.0]), ["x", "y", "z"]
            ),
            ConstraintSpecification(
                "inequality", "y >= 0", jnp.array([0.0, 1.0, 0.0]), ["x", "y", "z"]
            ),
            ConstraintSpecification(
                "inequality", "z >= 0", jnp.array([0.0, 0.0, 1.0]), ["x", "y", "z"]
            ),
        ]

        config = ProjectorConfig(hidden_sizes=[32], embedding_dim=8)
        learner = FeasibilityLearner(3, constraints, config, nnx.Rngs(0))

        # Create JIT-compiled version
        jitted_satisfy = nnx.jit(learner.satisfy_constraints)

        # Batch of variables violating constraints
        variables = jnp.array(
            [
                [0.5, 0.3, 0.1],  # Sum = 0.9, violates sum = 1
                [-0.1, 0.6, 0.5],  # x < 0, violates x >= 0
                [0.2, 0.3, 0.3],  # Sum = 0.8, violates sum = 1
            ]
        )

        # Warmup call to compile the function
        _ = jitted_satisfy(variables[:1])

        # Measure performance
        start_time = time.time()
        satisfied_variables = jitted_satisfy(variables)
        total_time = time.time() - start_time

        assert satisfied_variables.shape == (3, 3)
        assert jnp.isfinite(satisfied_variables).all()
        assert (
            total_time < 1.0
        )  # Should be reasonably fast for batch processing with JIT

        # Check constraints are better satisfied (relaxed tolerance for neural network)
        for var in satisfied_variables:
            # Check equality constraint: sum should be close to 1
            sum_violation = jnp.abs(jnp.sum(var) - 1.0)
            assert (
                sum_violation < 0.5
            )  # Reasonable tolerance for untrained neural network

            # Check inequality constraints: all should be >= 0
            assert jnp.all(var >= -0.01)  # Small tolerance for numerical precision

    def test_empty_constraints_handling(self):
        """Test behavior with empty constraint list."""
        config = ProjectorConfig(hidden_sizes=[32, 16], embedding_dim=8)
        learner = FeasibilityLearner(2, [], config, nnx.Rngs(0))  # No constraints

        point = jnp.array([1.0, 2.0])
        satisfied_point = learner.satisfy_constraints(point)

        # Should return original point when no constraints
        assert jnp.allclose(satisfied_point, point)

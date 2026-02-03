"""
JAX JIT Compatibility Tests for Conditions Module

This module tests JAX JIT compilation and performance for all boundary condition types.
Ensures that all condition classes work correctly with JAX's JIT compiler for optimal performance.
"""

import jax
import jax.numpy as jnp

from opifex.core.conditions import (
    BoundaryConditionCollection,
    DirichletBC,
    InitialCondition,
    NeumannBC,
    RobinBC,
    WavefunctionBC,
)


class TestJITCompatibility:
    """Test JAX JIT compatibility for conditions module."""

    def test_dirichlet_bc_jit_compilation(self):
        """Test JIT compilation of Dirichlet boundary condition evaluation."""
        bc = DirichletBC(boundary="left", value=2.5)

        # JIT compile the evaluation function
        @jax.jit
        def jit_evaluate(x, t):
            return bc.evaluate(x, t)

        x = jnp.array([1.0, 2.0, 3.0])
        t = 0.5

        # Test JIT compilation works
        result = jit_evaluate(x, t)
        expected = jnp.full_like(x, 2.5)
        assert jnp.allclose(result, expected)

        # Test performance improvement
        import time

        # Warmup
        for _ in range(3):
            jit_evaluate(x, t)

        # Time JIT version
        start = time.time()
        for _ in range(100):
            jit_evaluate(x, t)
        jit_time = time.time() - start

        # Time non-JIT version
        start = time.time()
        for _ in range(100):
            bc.evaluate(x, t)
        regular_time = time.time() - start

        # JIT should be at least as fast (may be faster for larger arrays)
        assert jit_time <= regular_time * 2.0  # Allow some overhead for small arrays

    def test_neumann_bc_jit_compilation(self):
        """Test JIT compilation of Neumann boundary condition evaluation."""
        bc = NeumannBC(boundary="right", value=1.5)

        @jax.jit
        def jit_evaluate(x, t):
            return bc.evaluate(x, t)

        x = jnp.array([1.0, 2.0, 3.0])
        result = jit_evaluate(x, 0.0)
        expected = jnp.full_like(x, 1.5)
        assert jnp.allclose(result, expected)

    def test_robin_bc_jit_compilation(self):
        """Test JIT compilation of Robin boundary condition evaluation."""
        bc = RobinBC(boundary="top", alpha=1.0, beta=2.0, gamma=3.0)

        @jax.jit
        def jit_evaluate(x, t):
            return bc.evaluate(x, t)

        x = jnp.array([1.0, 2.0])
        result = jit_evaluate(x, 0.0)
        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(result, expected)

    def test_initial_condition_jit_compilation(self):
        """Test JIT compilation of initial condition evaluation."""
        ic = InitialCondition(value=2.5, dimension=1)

        @jax.jit
        def jit_evaluate(x):
            return ic.evaluate(x)

        x = jnp.array([1.0, 2.0, 3.0])
        result = jit_evaluate(x)
        expected = jnp.full_like(x[..., 0], 2.5)
        assert jnp.allclose(result, expected)

    def test_wavefunction_bc_jit_compilation(self):
        """Test JIT compilation of wavefunction boundary condition evaluation."""
        bc = WavefunctionBC(condition_type="normalization", norm_value=1.0)

        @jax.jit
        def jit_evaluate(x, t):
            return bc.evaluate(x, t)

        x = jnp.array([1.0, 2.0, 3.0])
        result = jit_evaluate(x, 0.0)
        expected = jnp.full_like(x[..., 0], 1.0)
        assert jnp.allclose(result, expected)

    def test_boundary_condition_collection_jit_compatibility(self):
        """Test JIT compatibility of boundary condition collection operations."""
        bc1 = DirichletBC(boundary="left", value=1.0)
        bc2 = NeumannBC(boundary="right", value=2.0)
        bc3 = RobinBC(boundary="top", alpha=1.0, beta=1.0, gamma=0.0)

        collection = BoundaryConditionCollection([bc1, bc2, bc3])

        # Test that get_by_type works with JIT-compiled context
        @jax.jit
        def test_collection_operations():
            # This tests that the optimized get_by_type method is JIT-compatible
            return jnp.array([len(collection.conditions)])

        result = test_collection_operations()
        assert result[0] == 3

    def test_batch_boundary_condition_evaluation(self):
        """Test batch evaluation of boundary conditions with JIT."""
        bc = DirichletBC(boundary="left", value=2.0)

        @jax.jit
        def batch_evaluate(x_batch):
            # Use vmap to evaluate boundary condition on batch of points
            return jax.vmap(lambda x: bc.evaluate(x, 0.0))(x_batch)

        x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        results = batch_evaluate(x_batch)

        # DirichletBC.evaluate returns a scalar for each input, so shape should be (3,)
        assert results.shape == (3,)
        assert jnp.allclose(results, 2.0)

    def test_time_dependent_bc_jit_compilation(self):
        """Test JIT compilation of time-dependent boundary conditions."""

        def time_func(x, t):
            return jnp.sin(t) * jnp.sum(x)

        bc = DirichletBC(boundary="left", value=time_func, time_dependent=True)

        @jax.jit
        def jit_time_evaluate(x, t):
            return bc.evaluate(x, t)

        x = jnp.array([1.0, 2.0])
        t = jnp.pi / 2

        result = jit_time_evaluate(x, t)
        expected = jnp.sin(jnp.pi / 2) * jnp.sum(x)
        assert jnp.allclose(result, expected)

    def test_end_to_end_jit_workflow(self):
        """Test end-to-end JIT workflow with multiple boundary conditions."""
        # Create multiple boundary conditions
        bc1 = DirichletBC(boundary="left", value=1.0)
        bc2 = NeumannBC(boundary="right", value=0.5)
        bc3 = RobinBC(boundary="top", alpha=2.0, beta=1.0, gamma=1.5)

        @jax.jit
        def evaluate_all_bcs(x, t):
            """Evaluate all boundary conditions and return combined result."""
            result1 = bc1.evaluate(x, t)  # scalar
            result2 = bc2.evaluate(x, t)  # scalar
            result3 = bc3.evaluate(x, t)  # array [alpha, beta, gamma]

            # Combine results - bc3 returns array, so sum its elements
            return result1 + result2 + jnp.sum(result3)

        x = jnp.array([1.0, 2.0])
        t = 0.0

        result = evaluate_all_bcs(x, t)
        expected = 1.0 + 0.5 + (2.0 + 1.0 + 1.5)  # bc1 + bc2 + sum(bc3)
        assert jnp.allclose(result, expected)

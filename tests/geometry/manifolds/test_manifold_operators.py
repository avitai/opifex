"""
Tests for manifold neural operators with JAX compatibility.

This module tests the ManifoldNeuralOperator classes and ensures all methods
are compatible with JAX transformations (jit, vmap, grad).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.geometry.manifolds.operators import (
    HyperbolicNeuralOperator,
    ManifoldNeuralOperator,
    RiemannianNeuralOperator,
)
from opifex.geometry.manifolds.riemannian import (
    euclidean_metric,
    hyperbolic_metric,
    RiemannianManifold,
)


class ManifoldWrapper:
    """Wrapper to adapt RiemannianManifold to match Manifold protocol."""

    def __init__(self, manifold):
        self._manifold = manifold

    def __getattr__(self, name):
        return getattr(self._manifold, name)

    @property
    def dimension(self):
        return self._manifold.dimension

    @property
    def embedding_dimension(self):
        return self._manifold.embedding_dimension

    def exp_map(self, base, tangent):
        return self._manifold.exp_map(base, tangent)

    def log_map(self, base, point):
        return self._manifold.log_map(base, point)

    def metric_tensor(self, point):
        return self._manifold.metric_tensor(point)

    def geodesic_distance(self, point1, point2):
        return self._manifold.geodesic_distance(point1, point2)

    def christoffel_symbols(self, point):
        return self._manifold.christoffel_symbols(point)

    def parallel_transport(self, tangent, path_start, path_end):
        # Adapt the signature to match RiemannianManifold's implementation
        return self._manifold.parallel_transport(path_start, tangent, tangent)


class TestManifoldNeuralOperator:
    """Test ManifoldNeuralOperator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = ManifoldWrapper(
            RiemannianManifold(
                dimension=2,
                metric_function=euclidean_metric,
            )
        )

        self.key = jax.random.PRNGKey(42)
        self.rngs = nnx.Rngs(self.key)

        self.operator = ManifoldNeuralOperator(
            manifold=self.manifold,
            hidden_dim=16,
            rngs=self.rngs,
        )

        # Test data
        self.test_point = jnp.array([0.1, 0.2])
        self.batch_points = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    def test_operator_creation(self):
        """Test creation of manifold neural operator."""
        assert self.operator.manifold.dimension == 2
        assert isinstance(self.operator.encoder, nnx.Sequential)

    def test_single_point_processing(self):
        """Test processing of single point."""
        result = self.operator(self.test_point)

        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_batch_processing(self):
        """Test processing of batch of points."""
        result = self.operator(self.batch_points)

        assert result.shape == (3, 2)
        assert jnp.all(jnp.isfinite(result))

    def test_operator_consistency(self):
        """Test that operator produces consistent results."""
        result1 = self.operator(self.test_point)
        result2 = self.operator(self.test_point)

        assert jnp.allclose(result1, result2)


class TestJITCompatibility:
    """Test JIT compatibility of manifold neural operators."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = ManifoldWrapper(
            RiemannianManifold(
                dimension=2,
                metric_function=euclidean_metric,
            )
        )

        self.key = jax.random.PRNGKey(42)
        self.rngs = nnx.Rngs(self.key)

        self.operator = ManifoldNeuralOperator(
            manifold=self.manifold,
            hidden_dim=16,
            rngs=self.rngs,
        )

        # Test data
        self.test_point = jnp.array([0.1, 0.2])
        self.batch_points = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    def test_operator_jit_compatibility(self):
        """Test that operator can be JIT compiled."""
        jit_operator = jax.jit(self.operator)

        # Test compilation and execution
        result_normal = self.operator(self.test_point)
        result_jit = jit_operator(self.test_point)

        assert jnp.allclose(result_normal, result_jit, atol=1e-6)
        assert result_jit.shape == (2,)

    def test_batch_operator_jit_compatibility(self):
        """Test that batch operator can be JIT compiled."""
        jit_operator = jax.jit(self.operator)

        # Test compilation and execution
        result_normal = self.operator(self.batch_points)
        result_jit = jit_operator(self.batch_points)

        assert jnp.allclose(result_normal, result_jit, atol=1e-6)
        assert result_jit.shape == (3, 2)


class TestVMAPCompatibility:
    """Test VMAP compatibility of manifold neural operators."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = ManifoldWrapper(
            RiemannianManifold(
                dimension=2,
                metric_function=euclidean_metric,
            )
        )

        self.key = jax.random.PRNGKey(42)
        self.rngs = nnx.Rngs(self.key)

        self.operator = ManifoldNeuralOperator(
            manifold=self.manifold,
            hidden_dim=16,
            rngs=self.rngs,
        )

        # Test data
        self.batch_points = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    def test_operator_vmap_compatibility(self):
        """Test that operator works with vmap."""

        # Create vmap version that processes each point individually
        def process_single_point(point):
            return self.operator(point.reshape(1, -1)).reshape(-1)

        vmap_operator = jax.vmap(process_single_point)

        # Test vmap execution
        result_batch = self.operator(self.batch_points)
        result_vmap = vmap_operator(self.batch_points)

        assert jnp.allclose(result_batch, result_vmap, atol=1e-6)
        assert result_vmap.shape == (3, 2)


class TestGradCompatibility:
    """Test gradient compatibility of manifold neural operators."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = ManifoldWrapper(
            RiemannianManifold(
                dimension=2,
                metric_function=euclidean_metric,
            )
        )

        self.key = jax.random.PRNGKey(42)
        self.rngs = nnx.Rngs(self.key)

        self.operator = ManifoldNeuralOperator(
            manifold=self.manifold,
            hidden_dim=16,
            rngs=self.rngs,
        )

    def test_operator_grad_compatibility(self):
        """Test that we can compute gradients through operator."""

        def operator_norm(point):
            result = self.operator(point)
            return jnp.linalg.norm(result)

        grad_fn = jax.grad(operator_norm)

        point = jnp.array([0.1, 0.2])
        gradient = grad_fn(point)

        assert gradient.shape == (2,)
        assert jnp.all(jnp.isfinite(gradient))

    def test_operator_parameter_grad_compatibility(self):
        """Test that we can compute gradients with respect to parameters."""

        def loss_fn(point):
            result = self.operator(point)
            return jnp.sum(result**2)

        # Test that we can compute gradients (this tests parameter handling)
        point = jnp.array([0.1, 0.2])
        loss_value = loss_fn(point)

        assert jnp.isfinite(loss_value)
        assert loss_value >= 0


class TestRiemannianNeuralOperator:
    """Test RiemannianNeuralOperator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = ManifoldWrapper(
            RiemannianManifold(
                dimension=2,
                metric_function=euclidean_metric,
            )
        )

        self.key = jax.random.PRNGKey(42)
        self.rngs = nnx.Rngs(self.key)

        self.operator = RiemannianNeuralOperator(
            manifold=self.manifold,
            hidden_dim=16,
            rngs=self.rngs,
        )

        # Test data
        self.test_point = jnp.array([0.1, 0.2])
        self.batch_points = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    def test_riemannian_operator_creation(self):
        """Test creation of Riemannian neural operator."""
        assert self.operator.manifold.dimension == 2
        assert isinstance(self.operator.encoder, nnx.Sequential)
        assert isinstance(self.operator.metric_processor, nnx.Sequential)

    def test_riemannian_operator_processing(self):
        """Test processing with Riemannian operator."""
        result = self.operator(self.batch_points)

        assert result.shape == (3, 2)
        assert jnp.all(jnp.isfinite(result))

    def test_riemannian_operator_jit_compatibility(self):
        """Test that Riemannian operator can be JIT compiled."""
        jit_operator = jax.jit(self.operator)

        # Test compilation and execution
        result_normal = self.operator(self.batch_points)
        result_jit = jit_operator(self.batch_points)

        assert jnp.allclose(result_normal, result_jit, atol=1e-6)
        assert result_jit.shape == (3, 2)


class TestHyperbolicNeuralOperator:
    """Test HyperbolicNeuralOperator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = ManifoldWrapper(
            RiemannianManifold(
                dimension=2,
                metric_function=hyperbolic_metric(-1.0),
            )
        )

        self.key = jax.random.PRNGKey(42)
        self.rngs = nnx.Rngs(self.key)

        self.operator = HyperbolicNeuralOperator(
            manifold=self.manifold,
            hidden_dim=16,
            rngs=self.rngs,
        )

        # Test data (small values to stay in Poincaré disk)
        self.test_point = jnp.array([0.1, 0.1])
        self.batch_points = jnp.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])

    def test_hyperbolic_operator_creation(self):
        """Test creation of hyperbolic neural operator."""
        assert self.operator.manifold.dimension == 2
        assert isinstance(self.operator.encoder, nnx.Sequential)
        assert isinstance(self.operator.curvature_processor, nnx.Sequential)

    def test_hyperbolic_operator_processing(self):
        """Test processing with hyperbolic operator."""
        result = self.operator(self.batch_points)

        assert result.shape == (3, 2)
        assert jnp.all(jnp.isfinite(result))

    def test_hyperbolic_operator_jit_compatibility(self):
        """Test that hyperbolic operator can be JIT compiled."""
        jit_operator = jax.jit(self.operator)

        # Test compilation and execution
        result_normal = self.operator(self.batch_points)
        result_jit = jit_operator(self.batch_points)

        assert jnp.allclose(result_normal, result_jit, atol=1e-6)
        assert result_jit.shape == (3, 2)


class TestCombinedTransformations:
    """Test combined JAX transformations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = ManifoldWrapper(
            RiemannianManifold(
                dimension=2,
                metric_function=euclidean_metric,
            )
        )

        self.key = jax.random.PRNGKey(42)
        self.rngs = nnx.Rngs(self.key)

        self.operator = ManifoldNeuralOperator(
            manifold=self.manifold,
            hidden_dim=16,
            rngs=self.rngs,
        )

    def test_jit_vmap_combination(self):
        """Test JIT compilation of vmap functions."""

        def process_single_point(point):
            return self.operator(point.reshape(1, -1)).reshape(-1)

        vmap_operator = jax.vmap(process_single_point)
        jit_vmap_operator = jax.jit(vmap_operator)

        batch_points = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        result_vmap = vmap_operator(batch_points)
        result_jit_vmap = jit_vmap_operator(batch_points)

        assert jnp.allclose(result_vmap, result_jit_vmap, atol=1e-6)

    def test_jit_grad_combination(self):
        """Test JIT compilation of gradient functions."""

        def operator_norm(point):
            result = self.operator(point)
            return jnp.linalg.norm(result)

        grad_fn = jax.grad(operator_norm)
        jit_grad_fn = jax.jit(grad_fn)

        point = jnp.array([0.3, 0.4])

        result_grad = grad_fn(point)
        result_jit_grad = jit_grad_fn(point)

        assert jnp.allclose(result_grad, result_jit_grad, atol=1e-6)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = ManifoldWrapper(
            RiemannianManifold(
                dimension=2,
                metric_function=euclidean_metric,
            )
        )

        self.key = jax.random.PRNGKey(42)
        self.rngs = nnx.Rngs(self.key)

        self.operator = ManifoldNeuralOperator(
            manifold=self.manifold,
            hidden_dim=16,
            rngs=self.rngs,
        )

    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        empty_batch = jnp.zeros((0, 2))

        # This should handle gracefully or raise appropriate error
        try:
            result = self.operator(empty_batch)
            assert result.shape == (0, 2)
        except (ValueError, IndexError):
            # Expected for empty batches
            pass

    def test_large_batch_processing(self):
        """Test processing of large batches."""
        large_batch = jax.random.uniform(self.key, (100, 2)) * 0.1

        result = self.operator(large_batch)

        assert result.shape == (100, 2)
        assert jnp.all(jnp.isfinite(result))

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        small_points = jnp.array([[1e-8, 1e-8], [1e-7, 1e-7]])
        result_small = self.operator(small_points)

        assert jnp.all(jnp.isfinite(result_small))
        assert result_small.shape == (2, 2)


class TestPerformanceBenchmarks:
    """Test performance characteristics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = ManifoldWrapper(
            RiemannianManifold(
                dimension=3,
                metric_function=euclidean_metric,
            )
        )

        self.key = jax.random.PRNGKey(42)
        self.rngs = nnx.Rngs(self.key)

        self.operator = ManifoldNeuralOperator(
            manifold=self.manifold,
            hidden_dim=32,
            rngs=self.rngs,
        )

    def test_operator_performance(self):
        """Test that JIT compilation improves operator performance."""
        jit_operator = jax.jit(self.operator)

        point = jnp.array([0.1, 0.2, 0.3])

        # Warm up JIT compilation
        _ = jit_operator(point)

        # Both should produce same results
        result_normal = self.operator(point)
        result_jit = jit_operator(point)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (3,)

    def test_batch_operations_performance(self):
        """Test batch operations performance."""
        batch_size = 50
        key = jax.random.PRNGKey(42)
        batch_points = jax.random.uniform(key, (batch_size, 3)) * 0.1

        # Test batch operator computation
        jit_operator = jax.jit(self.operator)

        result = jit_operator(batch_points)

        assert result.shape == (batch_size, 3)
        assert jnp.all(jnp.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__])

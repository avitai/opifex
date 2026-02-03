"""Test Sensor Optimization functionality.

Test suite for sensor optimization components used in neural operators
for adaptive sensor placement and data acquisition.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.sensor_optimization import SensorOptimization


class TestSensorOptimization:
    """Test sensor optimization functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.backend = jax.default_backend()
        print(f"Running SensorOptimization tests on {self.backend}")

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_sensor_optimization_initialization(self, rngs):
        """Test sensor optimization initialization."""
        num_sensors = 20
        spatial_dim = 2

        sensor_opt = SensorOptimization(
            num_sensors=num_sensors,
            spatial_dim=spatial_dim,
            rngs=rngs,
        )

        # Check basic attributes
        assert hasattr(sensor_opt, "num_sensors")
        assert hasattr(sensor_opt, "spatial_dim")
        assert callable(sensor_opt)

        # Check for expected components
        assert hasattr(sensor_opt, "sensor_positions")
        assert hasattr(sensor_opt, "sensor_weights")

    def test_sensor_optimization_forward(self, rngs, rng_key):
        """Test sensor optimization forward pass."""
        batch_size = 4
        num_points = 100
        num_sensors = 20
        spatial_dim = 2
        channels = 3

        sensor_opt = SensorOptimization(
            num_sensors=num_sensors,
            spatial_dim=spatial_dim,
            rngs=rngs,
        )

        input_function = jnp.ones((batch_size, num_points, channels))
        spatial_coords = jnp.linspace(-1, 1, num_points).reshape(num_points, 1)
        spatial_coords = jnp.tile(spatial_coords, (1, spatial_dim))

        output = sensor_opt(input_function, spatial_coords)

        expected_shape = (batch_size, num_sensors, channels)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_sensor_optimization_fallback_method(self, rngs, rng_key):
        """Test sensor optimization with fallback method."""
        num_sensors = 10
        spatial_dim = 1

        sensor_opt = SensorOptimization(
            num_sensors=num_sensors,
            spatial_dim=spatial_dim,
            optimization_method="uniform",
            rngs=rngs,
        )

        batch_size = 2
        num_points = 50
        channels = 1

        input_function = jnp.ones((batch_size, num_points, channels))
        spatial_coords = jnp.linspace(-1, 1, num_points).reshape(num_points, 1)

        output = sensor_opt(input_function, spatial_coords)

        expected_shape = (batch_size, num_sensors, channels)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_sensor_optimization_learnable_positions(self, rngs, rng_key):
        """Test learnable sensor position optimization."""
        sensor_opt = SensorOptimization(
            num_sensors=15,
            spatial_dim=2,
            optimization_method="learnable",
            rngs=rngs,
        )

        batch_size = 3
        num_points = 80
        channels = 2

        input_function = jax.random.normal(rng_key, (batch_size, num_points, channels))
        spatial_coords = jax.random.uniform(
            rng_key, (num_points, 2), minval=-1, maxval=1
        )

        output = sensor_opt(input_function, spatial_coords)

        assert output.shape == (batch_size, 15, channels)
        assert jnp.all(jnp.isfinite(output))

    def test_sensor_optimization_different_dimensions(self, rngs, rng_key):
        """Test sensor optimization with different spatial dimensions."""
        spatial_dims = [1, 2, 3]

        for spatial_dim in spatial_dims:
            sensor_opt = SensorOptimization(
                num_sensors=8,
                spatial_dim=spatial_dim,
                rngs=rngs,
            )

            batch_size = 2
            num_points = 32
            channels = 1

            input_function = jnp.ones((batch_size, num_points, channels))
            spatial_coords = jax.random.uniform(
                rng_key, (num_points, spatial_dim), minval=-1, maxval=1
            )

            output = sensor_opt(input_function, spatial_coords)

            assert output.shape == (batch_size, 8, channels)
            assert jnp.all(jnp.isfinite(output))

    def test_sensor_optimization_differentiability(self, rngs, rng_key):
        """Test sensor optimization differentiability."""
        sensor_opt = SensorOptimization(
            num_sensors=12,
            spatial_dim=2,
            optimization_method="learnable",
            rngs=rngs,
        )

        def loss_fn(model, input_function, spatial_coords):
            output = model(input_function, spatial_coords, training=True)
            return jnp.mean(output**2)

        batch_size = 2
        num_points = 40
        channels = 1

        input_function = jax.random.normal(rng_key, (batch_size, num_points, channels))
        spatial_coords = jax.random.uniform(
            rng_key, (num_points, 2), minval=-1, maxval=1
        )

        grads = nnx.grad(loss_fn)(sensor_opt, input_function, spatial_coords)

        # Verify gradients exist
        assert grads is not None

        # Check that gradients are not all zero
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norms = [
            jnp.linalg.norm(leaf) for leaf in grad_leaves if hasattr(leaf, "shape")
        ]
        assert len(grad_norms) > 0
        assert any(norm > 1e-8 for norm in grad_norms)

    def test_sensor_optimization_adaptive_weighting(self, rngs, rng_key):
        """Test sensor optimization with adaptive weighting."""
        sensor_opt = SensorOptimization(
            num_sensors=10,
            spatial_dim=1,
            rngs=rngs,
        )

        batch_size = 2
        num_points = 50
        channels = 1

        # Create function with varying importance
        input_function = jax.random.normal(rng_key, (batch_size, num_points, channels))
        # Make some points more important
        input_function = input_function.at[:, :10, :].mul(5.0)

        spatial_coords = jnp.linspace(-1, 1, num_points).reshape(num_points, 1)

        output = sensor_opt(input_function, spatial_coords)

        assert output.shape == (batch_size, 10, channels)
        assert jnp.all(jnp.isfinite(output))

        # Output should capture the important regions
        assert jnp.std(output) > 0.1  # Should have meaningful variation

    def test_sensor_optimization_batch_consistency(self, rngs, rng_key):
        """Test sensor optimization consistency across batches."""
        sensor_opt = SensorOptimization(
            num_sensors=8,
            spatial_dim=2,
            rngs=rngs,
        )

        num_points = 32
        channels = 1
        spatial_coords = jax.random.uniform(
            rng_key, (num_points, 2), minval=-1, maxval=1
        )

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            input_function = jax.random.normal(
                rng_key, (batch_size, num_points, channels)
            )
            output = sensor_opt(input_function, spatial_coords)

            assert output.shape == (batch_size, 8, channels)
            assert jnp.all(jnp.isfinite(output))

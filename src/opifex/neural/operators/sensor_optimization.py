"""Sensor optimization module for neural operators.

This module provides sensor optimization capabilities for adaptive sampling
and measurement strategies in neural operator networks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class SensorOptimization(nnx.Module):
    """Adaptive sensor optimization for DeepONet variants.

    Optimizes sensor placement for improved data collection and operator learning.
    """

    def __init__(
        self,
        num_sensors: int,
        spatial_dim: int,
        *,
        optimization_method: str = "learnable",
        rngs: nnx.Rngs,
    ):
        """Initialize sensor optimization.

        Args:
            num_sensors: Number of sensors to optimize
            spatial_dim: Spatial dimension of the domain
            optimization_method: Method for sensor optimization
            rngs: Random number generators
        """
        super().__init__()
        self.num_sensors = num_sensors
        self.spatial_dim = spatial_dim
        self.optimization_method = optimization_method

        if optimization_method == "learnable":
            # Learnable sensor positions with improved initialization
            # Use Xavier initialization scaled appropriately for the domain
            self.sensor_positions = nnx.Param(
                nnx.initializers.xavier_normal()(
                    rngs.params(),
                    (num_sensors, spatial_dim),
                )
                * 0.5  # Scale to reasonable spatial range
            )
            # Add learnable sensor weights with better initialization for gradients
            self.sensor_weights = nnx.Param(
                nnx.initializers.normal(stddev=0.1)(
                    rngs.params(),
                    (num_sensors,),
                )
                + 1.0  # Initialize around 1.0 with small variation
            )
        else:
            # Fixed sensor positions - create as parameter for consistency
            fixed_positions = jnp.linspace(0, 1, num_sensors).reshape(-1, 1)
            if spatial_dim > 1:
                # Extend to multiple dimensions by repeating
                fixed_positions = jnp.tile(fixed_positions, (1, spatial_dim))
            self.sensor_positions = nnx.Param(fixed_positions)
            # Add fixed sensor weights for test compatibility
            self.sensor_weights = nnx.Param(jnp.ones((num_sensors,)))

    def __call__(
        self,
        x: jax.Array,
        sensor_positions: jax.Array | None = None,
        *,
        training: bool = False,
    ) -> jax.Array:
        """Apply sensor optimization.

        Args:
            x: Input field (batch, spatial_dims..., features)
            sensor_positions: Optional sensor positions for optimization
            training: Whether in training mode (enables learnable weights and noise)

        Returns:
            Optimized sensor measurements (batch, num_sensors, features)
        """
        # Apply sensor weights if available (training mode affects weight application)
        weighted_x = self._apply_sensor_weights(x, training=training)

        # Extract sensor measurements
        measurements = self._extract_sensor_measurements(weighted_x, sensor_positions)

        # Ensure output has correct shape
        return self._adjust_measurement_shape(measurements, x.shape[0])

    def _apply_sensor_weights(
        self, x: jax.Array, *, training: bool = False
    ) -> jax.Array:
        """Apply sensor weights to input data.

        Args:
            x: Input field data
            training: Whether in training mode (affects weight application)

        Returns:
            Weighted input data
        """
        if not hasattr(self, "sensor_weights"):
            return x

        weights = self.sensor_weights.value

        # In inference mode with learnable optimization, stop gradients for efficiency
        # In training mode, gradients flow through weights for learning
        if not training and self.optimization_method == "learnable":
            # Inference mode: stop gradients for computational efficiency
            weights = jax.lax.stop_gradient(weights)

        if x.ndim == 3:  # (batch, spatial, features)
            return self._apply_weights_3d(x, weights)
        return x

    def _apply_weights_3d(self, x: jax.Array, weights: jax.Array) -> jax.Array:
        """Apply weights to 3D input data.

        Args:
            x: Input data (batch, spatial, features)
            weights: Sensor weights

        Returns:
            Weighted input data
        """
        spatial_size = x.shape[1]

        if weights.shape[0] == spatial_size:
            # Broadcast weights: (spatial,) -> (1, spatial, 1)
            weights_expanded = weights[None, :, None]
            return x * weights_expanded
        if weights.shape[0] == self.num_sensors and spatial_size >= self.num_sensors:
            # Apply weights to first num_sensors positions
            weights_expanded = jnp.ones_like(x[:, : self.num_sensors, :])
            for i in range(self.num_sensors):
                weights_expanded = weights_expanded.at[:, i, :].set(weights[i])
            return x.at[:, : self.num_sensors, :].set(
                x[:, : self.num_sensors, :] * weights_expanded
            )
        return x

    def _extract_sensor_measurements(
        self, weighted_x: jax.Array, sensor_positions: jax.Array | None
    ) -> jax.Array:
        """Extract measurements from weighted input data.

        Args:
            weighted_x: Weighted input data
            sensor_positions: Optional sensor positions

        Returns:
            Sensor measurements
        """
        if sensor_positions is not None:
            return self._extract_from_positions(weighted_x, sensor_positions)
        return self._extract_default_measurements(weighted_x)

    def _extract_from_positions(
        self, weighted_x: jax.Array, sensor_positions: jax.Array
    ) -> jax.Array:
        """Extract measurements from specific sensor positions.

        Args:
            weighted_x: Weighted input data
            sensor_positions: Sensor positions

        Returns:
            Measurements at specified positions
        """
        if isinstance(sensor_positions, (int, jnp.integer)) or (
            isinstance(sensor_positions, jax.Array) and sensor_positions.ndim == 0
        ):
            return self._extract_single_position(weighted_x, int(sensor_positions))
        if isinstance(sensor_positions, jax.Array) and sensor_positions.ndim == 1:
            return self._extract_multiple_positions(weighted_x, sensor_positions)
        return weighted_x

    def _extract_single_position(
        self, weighted_x: jax.Array, sensor_idx: int
    ) -> jax.Array:
        """Extract measurement from single sensor position.

        Args:
            weighted_x: Weighted input data
            sensor_idx: Sensor index

        Returns:
            Single sensor measurement
        """
        if weighted_x.ndim == 3 and sensor_idx < weighted_x.shape[1]:
            return weighted_x[:, sensor_idx : sensor_idx + 1, :]
        return weighted_x[:, :1, :] if weighted_x.ndim >= 3 else weighted_x

    def _extract_multiple_positions(
        self, weighted_x: jax.Array, sensor_positions: jax.Array
    ) -> jax.Array:
        """Extract measurements from multiple sensor positions.

        Args:
            weighted_x: Weighted input data
            sensor_positions: Multiple sensor positions

        Returns:
            Multiple sensor measurements
        """
        sensor_indices = jnp.asarray(sensor_positions, dtype=jnp.int_)
        if weighted_x.ndim == 3:
            valid_indices = jnp.clip(sensor_indices, 0, weighted_x.shape[1] - 1)
            return weighted_x[:, valid_indices, :]
        return weighted_x

    def _extract_default_measurements(self, weighted_x: jax.Array) -> jax.Array:
        """Extract default measurements when no positions specified.

        Args:
            weighted_x: Weighted input data

        Returns:
            Default sensor measurements
        """
        if weighted_x.ndim == 3:
            if weighted_x.shape[1] >= self.num_sensors:
                return weighted_x[:, : self.num_sensors, :]
            # Pad measurements to match num_sensors
            return self._pad_measurements(weighted_x)
        # Create measurements with proper shape
        batch_size = weighted_x.shape[0]
        return jnp.zeros((batch_size, self.num_sensors, weighted_x.shape[-1]))

    def _pad_measurements(self, weighted_x: jax.Array) -> jax.Array:
        """Pad measurements to match required number of sensors.

        Args:
            weighted_x: Weighted input data

        Returns:
            Padded measurements
        """
        batch_size = weighted_x.shape[0]
        padding_needed = self.num_sensors - weighted_x.shape[1]
        padding = jnp.zeros((batch_size, padding_needed, weighted_x.shape[2]))
        return jnp.concatenate([weighted_x, padding], axis=1)

    def _adjust_measurement_shape(
        self, measurements: jax.Array, batch_size: int
    ) -> jax.Array:
        """Adjust measurement shape to match expected output.

        Args:
            measurements: Raw sensor measurements
            batch_size: Expected batch size

        Returns:
            Correctly shaped measurements
        """
        if measurements.ndim == 3 and measurements.shape[1] != self.num_sensors:
            if measurements.shape[1] > self.num_sensors:
                return measurements[:, : self.num_sensors, :]
            # Pad with zeros
            padding_needed = self.num_sensors - measurements.shape[1]
            padding = jnp.zeros(
                (batch_size, padding_needed, measurements.shape[2]),
            )
            return jnp.concatenate([measurements, padding], axis=1)
        return measurements

    def _extract_measurements(
        self, function_vals: jax.Array, coords: jax.Array
    ) -> jax.Array:
        """Extract measurements from function values at coordinates.

        Args:
            function_vals: Function values [spatial_points, features]
            coords: Spatial coordinates [spatial_points, spatial_dim]

        Returns:
            Measurements [num_sensors, features]
        """
        # Simple nearest neighbor interpolation for now
        # In practice, you might want more sophisticated interpolation

        # Ensure coords is 2D
        if coords.ndim == 1:
            coords = coords.reshape(-1, 1)

        # Ensure sensor_positions is 2D
        sensor_pos = self.sensor_positions.value
        if sensor_pos.ndim == 1:
            sensor_pos = sensor_pos.reshape(-1, 1)

        distances = jnp.linalg.norm(
            coords[:, None, :] - sensor_pos[None, :, :], axis=-1
        )
        nearest_indices = jnp.argmin(distances, axis=0)
        return function_vals[nearest_indices]

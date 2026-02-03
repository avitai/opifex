"""Riemannian manifold implementation with custom metrics.

This module implements general Riemannian manifolds M with custom metric tensors
for scientific machine learning applications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Float

    from opifex.geometry.manifolds.base import (
        ManifoldPoint,
        MetricTensor,
        TangentVector,
    )


class RiemannianManifold:
    """General Riemannian manifold M with custom metric tensor.

    Implements n-dimensional Riemannian manifolds with user-defined metric
    tensors and coordinate systems. All operations are JAX-compatible.
    """

    def __init__(
        self,
        dimension: int,
        metric_function: Callable[[ManifoldPoint], MetricTensor],
        embedding_dimension: int | None = None,
        coordinate_chart: Callable[[ManifoldPoint], ManifoldPoint] | None = None,
        inverse_chart: Callable[[ManifoldPoint], ManifoldPoint] | None = None,
    ):
        """Initialize Riemannian manifold.

        Args:
            dimension: Intrinsic dimension of the manifold
            metric_function: Function that returns metric tensor at each point
            embedding_dimension: Embedding space dimension (defaults to dimension)
            coordinate_chart: Optional coordinate transformation
            inverse_chart: Inverse coordinate transformation
        """
        if dimension <= 0:
            msg = "Dimension must be positive"
            raise ValueError(msg)

        self.dimension = dimension
        self.embedding_dimension = embedding_dimension or dimension
        self.metric_function = metric_function
        self.coordinate_chart = coordinate_chart or (lambda x: x)
        self.inverse_chart = inverse_chart or (lambda x: x)

    def metric_tensor(self, point: ManifoldPoint) -> MetricTensor:
        """Compute metric tensor at given point.

        Args:
            point: Point on manifold

        Returns:
            Metric tensor matrix at the point
        """
        return self.metric_function(point)

    def christoffel_symbols(
        self, point: ManifoldPoint
    ) -> Float[jax.Array, "dim dim dim"]:
        """Compute Christoffel symbols at given point using vectorized operations.

        Args:
            point: Point on manifold

        Returns:
            Christoffel symbols Γ^k_{ij}
        """

        def metric_at_point(p):
            return self.metric_tensor(p)

        # Compute partial derivatives of metric tensor
        metric_grad = jax.jacfwd(metric_at_point)(point)  # ∂g_ij/∂x^k

        # Get metric tensor and its inverse
        g = self.metric_tensor(point)
        g_inv = jnp.linalg.inv(g)

        # Vectorized computation of Christoffel symbols using einsum
        # Γ^k_{ij} = (1/2) g^{kl}(∂g_{il}/∂x^j + ∂g_{jl}/∂x^i - ∂g_{ij}/∂x^l)

        # Rearrange metric gradients for efficient computation
        term1 = jnp.einsum("ilj->ijl", metric_grad)  # ∂g_{il}/∂x^j
        term2 = jnp.einsum("jli->ijl", metric_grad)  # ∂g_{jl}/∂x^i
        term3 = jnp.einsum("ijl->ijl", metric_grad)  # ∂g_{ij}/∂x^l

        # Combine terms
        christoffel_terms = term1 + term2 - term3

        # Contract with inverse metric
        return 0.5 * jnp.einsum("kl,ijl->kij", g_inv, christoffel_terms)

    def riemann_curvature(
        self, point: ManifoldPoint
    ) -> Float[jax.Array, "dim dim dim dim"]:
        """Compute Riemann curvature tensor at given point using vectorized operations.

        Args:
            point: Point on manifold

        Returns:
            Riemann curvature tensor R^i_{jkl}
        """

        # Compute Christoffel symbols and their derivatives
        def christoffel_at_point(p):
            return self.christoffel_symbols(p)

        gamma = self.christoffel_symbols(point)
        gamma_grad = jax.jacfwd(christoffel_at_point)(point)  # ∂Γ^i_{jk}/∂x^l

        # Vectorized computation of Riemann tensor
        # R^i_{jkl} = ∂Γ^i_{jl}/∂x^k - ∂Γ^i_{jk}/∂x^l +
        # Γ^i_{mk}Γ^m_{jl} - Γ^i_{ml}Γ^m_{jk}

        # First term: ∂Γ^i_{jl}/∂x^k - ∂Γ^i_{jk}/∂x^l
        term1 = jnp.einsum("ijlk->ijkl", gamma_grad) - jnp.einsum(
            "ijkl->ijkl", gamma_grad
        )

        # Second term: Γ^i_{mk}Γ^m_{jl} - Γ^i_{ml}Γ^m_{jk}
        term2 = jnp.einsum("imk,mjl->ijkl", gamma, gamma) - jnp.einsum(
            "iml,mjk->ijkl", gamma, gamma
        )

        return term1 + term2

    def ricci_tensor(self, point: ManifoldPoint) -> MetricTensor:
        """Compute Ricci tensor at given point using vectorized operations.

        Args:
            point: Point on manifold

        Returns:
            Ricci tensor R_ij
        """
        riemann = self.riemann_curvature(point)

        # Vectorized Ricci tensor: R_ij = R^k_{ikj} (trace over first and third indices)
        return jnp.einsum("kikj->ij", riemann)

    def scalar_curvature(self, point: ManifoldPoint) -> jax.Array:
        """Compute scalar curvature at given point.

        Args:
            point: Point on manifold

        Returns:
            Scalar curvature R
        """
        ricci = self.ricci_tensor(point)
        g_inv = jnp.linalg.inv(self.metric_tensor(point))

        # Vectorized scalar curvature: R = g^{ij}R_ij
        return jnp.einsum("ij,ij->", g_inv, ricci)

    def exp_map(self, base: ManifoldPoint, tangent: TangentVector) -> ManifoldPoint:
        """Exponential map using JAX-native geodesic integration.

        Args:
            base: Base point on manifold
            tangent: Tangent vector at base point

        Returns:
            Point reached by following geodesic
        """
        return self._jax_geodesic_integration(base, tangent)

    def _jax_geodesic_integration(
        self, base: ManifoldPoint, tangent: TangentVector
    ) -> ManifoldPoint:
        """JAX-native geodesic integration using Runge-Kutta method."""

        def geodesic_ode(state):
            """Geodesic ODE: [position, velocity] -> [velocity, acceleration]."""
            pos, vel = state[: self.dimension], state[self.dimension :]

            # Compute Christoffel symbols at current position
            gamma = self.christoffel_symbols(pos)

            # Vectorized acceleration computation: -Γ^i_{jk} v^j v^k
            acceleration = -jnp.einsum("ijk,j,k->i", gamma, vel, vel)

            return jnp.concatenate([vel, acceleration])

        def rk4_step(state, dt):
            """Single Runge-Kutta 4th order step."""
            k1 = dt * geodesic_ode(state)
            k2 = dt * geodesic_ode(state + k1 / 2)
            k3 = dt * geodesic_ode(state + k2 / 2)
            k4 = dt * geodesic_ode(state + k3)

            return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Integration parameters
        dt = 0.01
        n_steps = int(1.0 / dt)

        # Initial state: [position, velocity]
        initial_state = jnp.concatenate([base, tangent])

        # Integrate using lax.fori_loop for JIT compatibility
        def integration_step(_, state):
            return rk4_step(state, dt)

        final_state = jax.lax.fori_loop(0, n_steps, integration_step, initial_state)

        # Extract final position
        return final_state[: self.dimension]

    def log_map(self, base: ManifoldPoint, point: ManifoldPoint) -> TangentVector:
        """Logarithmic map (inverse exponential map).

        Args:
            base: Base point on manifold
            point: Target point on manifold

        Returns:
            Tangent vector at base point
        """
        return self._jax_log_map_optimization(base, point)

    def _jax_log_map_optimization(
        self, base: ManifoldPoint, point: ManifoldPoint
    ) -> TangentVector:
        """JAX-native optimization for log map computation using gradient descent."""

        def objective(tangent_vec):
            """Objective function: ||exp_map(base, tangent_vec) - point||²."""
            result = self._jax_geodesic_integration(base, tangent_vec)
            return jnp.sum((result - point) ** 2)

        # Initial guess: straight line in embedding space
        tangent_vec = point - base

        # Gradient descent parameters
        learning_rate = 0.01
        max_iterations = 100

        def update_step(_, carry):
            tangent_vec, _ = carry

            # Compute gradient and loss
            loss, grad = jax.value_and_grad(objective)(tangent_vec)

            # Update tangent vector
            new_tangent_vec = tangent_vec - learning_rate * grad

            return new_tangent_vec, loss

        # Run optimization loop with fixed iterations (grad-compatible)
        initial_carry = (tangent_vec, jnp.inf)
        final_tangent_vec, _ = jax.lax.fori_loop(
            0, max_iterations, update_step, initial_carry
        )

        return final_tangent_vec

    def geodesic_distance(
        self, point1: ManifoldPoint, point2: ManifoldPoint
    ) -> jax.Array:
        """Compute geodesic distance between two points.

        Args:
            point1: First point
            point2: Second point

        Returns:
            Geodesic distance
        """
        # Distance is norm of tangent vector in log map
        tangent = self._jax_log_map_optimization(point1, point2)
        g = self.metric_tensor(point1)

        # Distance = sqrt(g_ij v^i v^j)
        return jnp.sqrt(jnp.einsum("i,ij,j->", tangent, g, tangent))

    def batch_geodesic_distance(
        self, points1: jax.Array, points2: jax.Array
    ) -> jax.Array:
        """Vectorized geodesic distance computation for batches of points.

        Args:
            points1: First batch of points [batch_size, manifold_dim]
            points2: Second batch of points [batch_size, manifold_dim]

        Returns:
            Batch of geodesic distances [batch_size]
        """
        return jax.vmap(self.geodesic_distance)(points1, points2)

    def batch_exp_map(self, bases: jax.Array, tangents: jax.Array) -> jax.Array:
        """Vectorized exponential map for batches.

        Args:
            bases: Batch of base points [batch_size, manifold_dim]
            tangents: Batch of tangent vectors [batch_size, manifold_dim]

        Returns:
            Batch of exponential map results [batch_size, manifold_dim]
        """
        return jax.vmap(self._jax_geodesic_integration)(bases, tangents)

    def batch_log_map(self, bases: jax.Array, points: jax.Array) -> jax.Array:
        """Vectorized logarithmic map for batches.

        Args:
            bases: Batch of base points [batch_size, manifold_dim]
            points: Batch of target points [batch_size, manifold_dim]

        Returns:
            Batch of tangent vectors [batch_size, manifold_dim]
        """
        return jax.vmap(self._jax_log_map_optimization)(bases, points)

    def parallel_transport(
        self, base: ManifoldPoint, tangent: TangentVector, vector: TangentVector
    ) -> TangentVector:
        """Parallel transport vector along geodesic.

        Args:
            base: Base point
            tangent: Direction of transport (tangent vector)
            vector: Vector to transport

        Returns:
            Parallel transported vector
        """
        return self._jax_parallel_transport(base, tangent, vector)

    def _jax_parallel_transport(
        self, base: ManifoldPoint, tangent: TangentVector, vector: TangentVector
    ) -> TangentVector:
        """JAX-native parallel transport integration."""

        def transport_ode(state):
            """Combined geodesic + parallel transport ODE."""
            # State: [position, velocity, transported_vector]
            pos = state[: self.dimension]
            vel = state[self.dimension : 2 * self.dimension]
            transport_vec = state[2 * self.dimension :]

            # Compute Christoffel symbols
            gamma = self.christoffel_symbols(pos)

            # Geodesic acceleration: -Γ^i_{jk} v^j v^k
            acceleration = -jnp.einsum("ijk,j,k->i", gamma, vel, vel)

            # Parallel transport derivative: -Γ^i_{jk} V^j v^k
            transport_derivative = -jnp.einsum("ijk,j,k->i", gamma, transport_vec, vel)

            return jnp.concatenate([vel, acceleration, transport_derivative])

        def rk4_step(state, dt):
            """Runge-Kutta 4th order step for combined system."""
            k1 = dt * transport_ode(state)
            k2 = dt * transport_ode(state + k1 / 2)
            k3 = dt * transport_ode(state + k2 / 2)
            k4 = dt * transport_ode(state + k3)

            return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Integration parameters
        dt = 0.01
        n_steps = int(1.0 / dt)

        # Initial state: [position, velocity, vector_to_transport]
        initial_state = jnp.concatenate([base, tangent, vector])

        # Integrate using lax.fori_loop
        def integration_step(_, state):
            return rk4_step(state, dt)

        final_state = jax.lax.fori_loop(0, n_steps, integration_step, initial_state)

        # Extract transported vector
        return final_state[2 * self.dimension :]

    def random_point(self, key: jax.Array, shape: tuple = ()) -> ManifoldPoint:
        """Generate random point on manifold.

        This is a basic implementation that generates points in the coordinate
        space. For manifolds with constraints, this should be overridden.

        Args:
            key: Random key
            shape: Shape of output (for batch generation)

        Returns:
            Random point(s) on manifold
        """
        full_shape = (*shape, self.dimension)

        # Generate random points in [-1, 1]^n
        # For general manifolds, this may not be appropriate
        return jax.random.uniform(key, full_shape, minval=-1.0, maxval=1.0)


# Predefined metric functions for common manifolds


def euclidean_metric(point: ManifoldPoint) -> MetricTensor:
    """Euclidean (flat) metric tensor."""
    dim = point.shape[-1]
    return jnp.eye(dim)


def hyperbolic_metric(_: float = -1.0):
    """Hyperbolic metric in Poincaré disk model."""

    def metric_fn(point: ManifoldPoint) -> MetricTensor:
        # Poincaré disk metric: g_ij = (4 / (1 - |x|²)²) δ_ij
        norm_sq = jnp.sum(point**2, axis=-1, keepdims=True)
        factor = 4.0 / (1.0 - norm_sq) ** 2
        dim = point.shape[-1]
        return factor * jnp.eye(dim)

    return metric_fn


def spherical_metric(radius: float = 1.0):
    """Spherical metric tensor."""

    def metric_fn(point: ManifoldPoint) -> MetricTensor:
        # For sphere embedded in R³, metric in spherical coordinates
        # This is a simplified version - full implementation would depend on coordinates
        dim = point.shape[-1]
        return (radius**2) * jnp.eye(dim)

    return metric_fn


def product_metric(*metrics):
    """Product metric for product manifolds."""

    def metric_fn(point: ManifoldPoint) -> MetricTensor:
        total_dim = 0
        metric_blocks = []

        start_idx = 0
        for metric in metrics:
            # Assume each metric function expects points of its own dimension
            # This is simplified - real implementation would need dimension info
            sub_point = point[..., start_idx : start_idx + 2]  # Assume 2D for now
            sub_metric = metric(sub_point)
            metric_blocks.append(sub_metric)
            start_idx += 2
            total_dim += 2

        # Block diagonal matrix
        return jax.scipy.linalg.block_diag(*metric_blocks)

    return metric_fn

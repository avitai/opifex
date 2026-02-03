"""Physics-Informed Neural Network (PINN) Solver.

This module implements the PINNSolver, which provides a high-level API for solving
PDE problems using physics-informed neural networks with Opifex's physics loss
infrastructure.

Design Principles:
- Composes with existing PhysicsLossConfig (no duplication)
- Accepts residual functions directly (no hidden string keys)
- Provides factory functions for common PDEs
- Clean separation of concerns
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from opifex.core.physics import AutoDiffEngine
from opifex.core.physics.losses import PhysicsLossComposer, PhysicsLossConfig


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Array, Float

    from opifex.geometry.base import Geometry


# =============================================================================
# Residual Function Factories
# =============================================================================
# These provide convenient ways to create residual functions for common PDEs.
# Users can also define their own or use PDEResidualRegistry directly.


def poisson_residual(
    source_fn: Callable[[Float[Array, "... d"]], Float[Array, ...]],
) -> Callable[[nnx.Module, Float[Array, "... d"]], Float[Array, ...]]:
    """Create a Poisson equation residual function.

    Creates a residual function for the standard Poisson equation:
        -∇²u = f(x)

    The residual is: -∇²u - f(x), which should be zero for the solution.

    Args:
        source_fn: Source term function f(x)

    Returns:
        Residual function with signature (model, x) -> residual

    Example:
        >>> source = lambda x: jnp.pi**2 * jnp.sin(jnp.pi * x)
        >>> residual_fn = poisson_residual(source)
        >>> # Use with PINNSolver
        >>> result = solver.solve(geometry, residual_fn, bc_fn, config)
    """

    def residual_fn(model: nnx.Module, x: Float[Array, "... d"]) -> Float[Array, ...]:
        laplacian = AutoDiffEngine.compute_laplacian(model, x)
        laplacian = jnp.real(laplacian)  # Ensure real output
        source = source_fn(x)
        if source.ndim > 1:
            source = source.squeeze(-1)
        # Standard Poisson: -∇²u = f => residual = -∇²u - f
        return -laplacian - source

    return residual_fn


def heat_residual(
    alpha: float = 1.0,
) -> Callable[[nnx.Module, Float[Array, "... d"]], Float[Array, ...]]:
    """Create a steady-state heat equation residual function.

    Creates a residual function for the steady-state heat equation:
        α∇²u = 0

    Args:
        alpha: Thermal diffusivity coefficient

    Returns:
        Residual function with signature (model, x) -> residual
    """

    def residual_fn(model: nnx.Module, x: Float[Array, "... d"]) -> Float[Array, ...]:
        laplacian = AutoDiffEngine.compute_laplacian(model, x)
        laplacian = jnp.real(laplacian)
        return alpha * laplacian

    return residual_fn


def helmholtz_residual(
    k: float,
    source_fn: Callable[[Float[Array, "... d"]], Float[Array, ...]] | None = None,
) -> Callable[[nnx.Module, Float[Array, "... d"]], Float[Array, ...]]:
    """Create a Helmholtz equation residual function.

    Creates a residual function for the Helmholtz equation:
        ∇²u + k²u = f(x)

    Args:
        k: Wave number
        source_fn: Optional source term function f(x). Defaults to zero.

    Returns:
        Residual function with signature (model, x) -> residual
    """

    def residual_fn(model: nnx.Module, x: Float[Array, "... d"]) -> Float[Array, ...]:
        u = model(x)  # type: ignore[reportCallIssue]
        if u.ndim > 1:
            u = u.squeeze(-1)
        laplacian = AutoDiffEngine.compute_laplacian(model, x)
        laplacian = jnp.real(laplacian)

        residual = laplacian + k**2 * u
        if source_fn is not None:
            source = source_fn(x)
            if source.ndim > 1:
                source = source.squeeze(-1)
            residual = residual - source
        return residual

    return residual_fn


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PINNConfig:
    """Configuration for PINN solver.

    Uses composition with PhysicsLossConfig for loss weights instead of
    duplicating fields. This ensures consistency with the rest of Opifex's
    physics loss infrastructure.

    Attributes:
        n_interior: Number of interior collocation points
        n_boundary: Number of boundary collocation points
        num_iterations: Number of training iterations
        learning_rate: Learning rate for optimizer
        print_every: Print loss every N iterations (0 to disable)
        seed: Random seed for reproducibility
        loss_config: PhysicsLossConfig for loss weights (composition, not duplication)
    """

    n_interior: int = 100
    n_boundary: int = 50
    num_iterations: int = 2000
    learning_rate: float = 1e-3
    print_every: int = 500
    seed: int = 42
    loss_config: PhysicsLossConfig = field(
        default_factory=lambda: PhysicsLossConfig(
            data_loss_weight=0.0,
            physics_loss_weight=1.0,
            boundary_loss_weight=100.0,
        )
    )


@dataclass
class PINNResult:
    """Result from PINN solver.

    Attributes:
        model: Trained PINN model
        losses: List of total loss values during training
        final_loss: Final training loss
        training_time: Total training time in seconds
        metrics: Additional metrics dictionary
    """

    model: nnx.Module
    losses: list[float]
    final_loss: float
    training_time: float
    metrics: dict[str, Any]


# =============================================================================
# Solver
# =============================================================================


class PINNSolver:
    """Solver for Physics-Informed Neural Networks.

    This solver provides a high-level API for solving PDEs using PINNs.
    It integrates with Opifex's physics loss infrastructure:
    - PhysicsLossComposer for loss weighting
    - AutoDiffEngine for automatic differentiation
    - Geometry classes for domain sampling

    The solver accepts residual functions directly rather than hiding them
    behind string keys. Use the provided factory functions (poisson_residual,
    heat_residual, etc.) or define your own.

    Example:
        >>> from opifex.neural.pinns import create_poisson_pinn
        >>> from opifex.geometry import Interval
        >>> from opifex.solvers import PINNSolver, PINNConfig, poisson_residual
        >>>
        >>> # Create geometry and model
        >>> geometry = Interval(-1.0, 1.0)
        >>> pinn = create_poisson_pinn(spatial_dim=1, rngs=nnx.Rngs(42))
        >>>
        >>> # Define source term and create residual function
        >>> source_fn = lambda x: jnp.pi**2 * jnp.sin(jnp.pi * x)
        >>> residual_fn = poisson_residual(source_fn)
        >>> bc_fn = lambda x: jnp.zeros_like(x[..., 0])
        >>>
        >>> # Solve
        >>> solver = PINNSolver(pinn)
        >>> result = solver.solve(geometry, residual_fn, bc_fn, PINNConfig())
    """

    def __init__(self, model: nnx.Module):
        """Initialize the PINN solver.

        Args:
            model: The neural network model to train (e.g., SimplePINN).
        """
        self.model = model

    def solve(
        self,
        geometry: Geometry,
        residual_fn: Callable[[nnx.Module, Float[Array, "... d"]], Float[Array, ...]],
        bc_fn: Callable[[Float[Array, "... d"]], Float[Array, ...]],
        config: PINNConfig | None = None,
    ) -> PINNResult:
        """Solve a PDE using physics-informed neural network.

        Args:
            geometry: Computational domain (e.g., Interval, Rectangle)
            residual_fn: PDE residual function with signature (model, x) -> residual.
                Use factory functions like poisson_residual() or define your own.
            bc_fn: Boundary condition function returning target values at
                boundary points.
            config: Solver configuration. Uses defaults if None.

        Returns:
            PINNResult containing trained model and metrics

        Example:
            >>> # Using factory function
            >>> residual_fn = poisson_residual(lambda x: jnp.ones(x.shape[0]))
            >>> result = solver.solve(geometry, residual_fn, bc_fn, config)

            >>> # Using custom residual
            >>> def custom_residual(model, x):
            ...     u = model(x).squeeze(-1)
            ...     laplacian = AutoDiffEngine.compute_laplacian(model, x)
            ...     return laplacian - u  # Example: ∇²u = u
            >>> result = solver.solve(geometry, custom_residual, bc_fn, config)
        """
        config = config or PINNConfig()

        # Create boundary loss function
        def bc_loss_fn(
            model: nnx.Module, x_bc: Float[Array, "... d"]
        ) -> Float[Array, ""]:
            """Compute boundary condition loss."""
            u_pred = model(x_bc)  # type: ignore[reportCallIssue]
            if u_pred.ndim > 1:
                u_pred = u_pred.squeeze(-1)
            u_target = bc_fn(x_bc)
            if u_target.ndim > 1:
                u_target = u_target.squeeze(-1)
            return jnp.mean((u_pred - u_target) ** 2)

        return self._train(
            geometry=geometry,
            pde_residual_fn=residual_fn,
            bc_loss_fn=bc_loss_fn,
            config=config,
        )

    def _train(
        self,
        geometry: Geometry,
        pde_residual_fn: Callable,
        bc_loss_fn: Callable,
        config: PINNConfig,
    ) -> PINNResult:
        """Internal training loop.

        Args:
            geometry: Computational domain
            pde_residual_fn: PDE residual function
            bc_loss_fn: Boundary condition loss function
            config: Solver configuration

        Returns:
            PINNResult containing trained model and metrics
        """
        # Use existing PhysicsLossComposer (composition, not duplication)
        loss_composer = PhysicsLossComposer(config.loss_config)

        # Generate collocation points
        key = jax.random.PRNGKey(config.seed)
        key_interior, key_boundary = jax.random.split(key)

        x_interior = geometry.sample_interior(config.n_interior, key_interior)
        x_boundary = geometry.sample_boundary(config.n_boundary, key_boundary)

        # Setup optimizer
        opt = nnx.Optimizer(self.model, optax.adam(config.learning_rate), wrt=nnx.Param)

        def total_loss_fn(model: nnx.Module) -> Float[Array, ""]:
            """Compute total physics-informed loss."""
            # PDE residual at interior points
            residual = pde_residual_fn(model, x_interior)
            physics_loss = jnp.mean(residual**2)

            # Boundary condition loss
            boundary_loss = bc_loss_fn(model, x_boundary)

            # Compose using PhysicsLossComposer
            return loss_composer.compose_loss(
                data_loss=jnp.array(0.0),
                physics_residual=physics_loss,
                boundary_residual=boundary_loss,
            )

        @nnx.jit
        def train_step(model: nnx.Module, opt: nnx.Optimizer) -> Float[Array, ""]:
            """Single training step."""
            loss, grads = nnx.value_and_grad(total_loss_fn)(model)
            opt.update(model, grads)
            return loss

        # Training loop
        losses = []
        start_time = time.time()

        for i in range(config.num_iterations):
            loss = train_step(self.model, opt)
            losses.append(float(loss))

            if config.print_every > 0 and (
                i % config.print_every == 0 or i == config.num_iterations - 1
            ):
                print(f"  Iteration {i:4d}: loss = {loss:.6e}")  # noqa: T201

        training_time = time.time() - start_time
        final_loss = losses[-1]

        # Compute metrics
        metrics = {
            "final_loss": final_loss,
            "initial_loss": losses[0],
            "training_time": training_time,
            "n_iterations": config.num_iterations,
            "n_interior": config.n_interior,
            "n_boundary": config.n_boundary,
        }

        return PINNResult(
            model=self.model,
            losses=losses,
            final_loss=final_loss,
            training_time=training_time,
            metrics=metrics,
        )

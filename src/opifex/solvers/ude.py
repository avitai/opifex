"""Universal Differential Equation (UDE) Solver.

Implements the UDE framework from:
    Rackauckas et al. "Universal Differential Equations for Scientific
    Machine Learning" (2020, arXiv:2001.04385)

Core idea: Grey-box modeling via du/dt = known(t, y) + NN(y), where
known physics and a neural network residual are combined and integrated
using diffrax ODE solvers.

References:
    - UDE concept: Rackauckas et al. (2020)
    - ODE integration: diffrax library (Kidger, 2022)
    - Neural ODE: Chen et al. (2018, NeurIPS)

Implementation follows the diffrax API patterns from:
    ../diffrax/benchmarks/small_neural_ode.py
"""

# ruff: noqa: F821  # jaxtyping dimension names (dim, n_times, etc.)

from collections.abc import Callable
from dataclasses import dataclass

import diffrax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


@dataclass(frozen=True)
class UDEConfig:
    """Configuration for UDE Solver.

    Attributes:
        dt0: Initial step size for adaptive solver.
        rtol: Relative tolerance for adaptive step control.
        atol: Absolute tolerance for adaptive step control.
        max_steps: Maximum number of integration steps.
    """

    dt0: float = 0.01
    rtol: float = 1e-5
    atol: float = 1e-5
    max_steps: int = 4096


class NeuralODE(nnx.Module):
    """Neural ODE: wraps a neural network as an ODE vector field.

    Integrates dy/dt = net(y) using diffrax. Compatible with JAX's
    automatic differentiation for training.

    The vector field signature follows diffrax convention:
        f(t, y, args) -> dy/dt
    """

    def __init__(
        self,
        net: nnx.Module,
        state_dim: int,
        *,
        config: UDEConfig | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize NeuralODE.

        Args:
            net: Neural network mapping state -> state derivative.
            state_dim: Dimensionality of the state vector.
            config: Solver configuration.
            rngs: Random number generators.
        """
        super().__init__()
        self.net = net
        self.state_dim = state_dim
        self.config = config or UDEConfig()

    def vector_field(self, t: float, y: Array, args) -> Array:
        """Evaluate dy/dt = net(y).

        Args:
            t: Current time (unused for autonomous systems, kept for API).
            y: Current state vector (state_dim,).
            args: Additional arguments (unused).

        Returns:
            State derivative (state_dim,).
        """
        # net expects (batch, dim) input, returns (batch, dim)
        return self.net(y[None, :])[0]  # type: ignore[reportCallIssue]

    def solve(
        self,
        y0: Float[Array, "dim"],
        ts: Float[Array, "n_times"],
        config: UDEConfig | None = None,
    ) -> Float[Array, "n_times dim"]:
        """Integrate the Neural ODE from y0 over time points ts.

        Args:
            y0: Initial state (state_dim,).
            ts: Time points to save at (n_times,).
            config: Optional override config.

        Returns:
            Trajectory at saved time points (n_times, state_dim).
        """
        cfg = config or self.config
        term = diffrax.ODETerm(self.vector_field)  # type: ignore[reportArgumentType]
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=cfg.rtol, atol=cfg.atol)
        saveat = diffrax.SaveAt(ts=ts)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=cfg.dt0,
            y0=y0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=cfg.max_steps,
        )
        return sol.ys  # type: ignore[reportReturnType]


class UDESolver(nnx.Module):
    """Universal Differential Equation Solver.

    Combines known physics with a neural network residual:

        du/dt = known(t, y) + neural(y)

    where `known` encodes prior knowledge (e.g., conservation laws,
    reaction kinetics) and `neural` learns unknown or missing terms
    from data.

    This implements the grey-box modeling paradigm from Rackauckas 2020.
    """

    def __init__(
        self,
        known_dynamics_fn: Callable,
        neural_residual: nnx.Module,
        state_dim: int,
        *,
        config: UDEConfig | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize UDE Solver.

        Args:
            known_dynamics_fn: Known physics function with signature
                (t, y) -> dy/dt. Must be a pure function (no side effects).
            neural_residual: Neural network module mapping state -> correction.
                Should accept (batch, state_dim) input.
            state_dim: Dimensionality of the state vector.
            config: Solver configuration.
            rngs: Random number generators.
        """
        super().__init__()
        self.known_dynamics_fn = known_dynamics_fn
        self.neural_residual = neural_residual
        self.state_dim = state_dim
        self.config = config or UDEConfig()

    def vector_field(self, t: float, y: Array, args) -> Array:
        """Evaluate du/dt = known(t, y) + neural(y).

        Args:
            t: Current time.
            y: Current state (state_dim,).
            args: Additional arguments (unused).

        Returns:
            Combined dynamics (state_dim,).
        """
        known = self.known_dynamics_fn(t, y)
        neural = self.neural_residual(y[None, :])[0]  # type: ignore[reportCallIssue]
        return known + neural

    def solve(
        self,
        y0: Float[Array, "dim"],
        ts: Float[Array, "n_times"],
        config: UDEConfig | None = None,
    ) -> Float[Array, "n_times dim"]:
        """Integrate the UDE from y0 over time points ts.

        Args:
            y0: Initial state (state_dim,).
            ts: Time points to save at (n_times,).
            config: Optional override config.

        Returns:
            Trajectory at saved time points (n_times, state_dim).
        """
        cfg = config or self.config
        term = diffrax.ODETerm(self.vector_field)  # type: ignore[reportArgumentType]
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=cfg.rtol, atol=cfg.atol)
        saveat = diffrax.SaveAt(ts=ts)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=cfg.dt0,
            y0=y0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=cfg.max_steps,
        )
        return sol.ys  # type: ignore[reportReturnType]

    def trajectory_loss(
        self,
        y0: Float[Array, "dim"],
        t_obs: Float[Array, "n_obs"],
        y_obs: Float[Array, "n_obs dim"],
    ) -> Float[Array, ""]:
        """Compute MSE loss between predicted and observed trajectory.

        This is the standard UDE training objective: minimize the
        discrepancy between the integrated trajectory and observations.

        Args:
            y0: Initial state (state_dim,).
            t_obs: Observation times (n_obs,).
            y_obs: Observed states (n_obs, state_dim).

        Returns:
            Scalar MSE loss.
        """
        y_pred = self.solve(y0, t_obs)
        return jnp.mean((y_pred - y_obs) ** 2)


def create_ude(
    known_dynamics_fn: Callable,
    state_dim: int,
    hidden_dims: list[int],
    *,
    config: UDEConfig | None = None,
    activation: Callable[[Array], Array] = jnp.tanh,
    rngs: nnx.Rngs,
) -> UDESolver:
    """Create a UDE solver with an MLP residual network.

    Args:
        known_dynamics_fn: Known physics function (t, y) -> dy/dt.
        state_dim: State vector dimensionality.
        hidden_dims: List of hidden layer sizes for the neural residual.
        config: Solver configuration.
        activation: Activation function for hidden layers.
        rngs: Random number generators.

    Returns:
        Configured UDESolver instance.
    """
    # Build MLP for neural residual
    layers = []
    in_features = state_dim
    for h in hidden_dims:
        layers.append(nnx.Linear(in_features, h, rngs=rngs))
        in_features = h
    layers.append(nnx.Linear(in_features, state_dim, rngs=rngs))

    class ResidualMLP(nnx.Module):
        """MLP for the neural residual term."""

        def __init__(self, layers, activation):
            super().__init__()
            self.layers = nnx.List(layers)
            self.activation = activation

        def __call__(self, x):
            h = x
            for layer in list(self.layers)[:-1]:
                h = self.activation(layer(h))
            return list(self.layers)[-1](h)

    net = ResidualMLP(layers, activation)

    return UDESolver(
        known_dynamics_fn=known_dynamics_fn,
        neural_residual=net,
        state_dim=state_dim,
        config=config,
        rngs=rngs,
    )

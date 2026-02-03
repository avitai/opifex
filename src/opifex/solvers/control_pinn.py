"""ControlPINN: PDE-Constrained Optimal Control via PINNs.

Implements simultaneous state and control optimization within a PINN
framework. A single neural network outputs both state variables u(x,t)
and control inputs c(x,t), trained with a combined loss:

    L = J(u, c) + lambda_pde * ||PDE(u, c)||^2 + lambda_ctrl * ||c||^2

where J is the cost objective, PDE constraint ensures physical
consistency, and control regularization prevents extreme solutions.

Reference: ComputationalScienceLaboratory/control-pinns (2023)
    - Single-stage framework for simultaneous state + control learning
    - Applied to heat equation, predator-prey, and analytical problems
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


@dataclass(frozen=True)
class ControlPINNConfig:
    """Configuration for ControlPINN.

    Attributes:
        n_state_outputs: Number of state variables to predict.
        n_control_outputs: Number of control variables to predict.
        pde_weight: Weight for PDE residual loss term.
        control_weight: Weight for control regularization (L2 norm).
    """

    n_state_outputs: int = 1
    n_control_outputs: int = 1
    pde_weight: float = 10.0
    control_weight: float = 0.01


class ControlPINN(nnx.Module):
    """Physics-Informed Neural Network for PDE-constrained optimal control.

    A single MLP outputs both state u(x) and control c(x), with the
    output split: first `n_state` outputs are state, last `n_control`
    outputs are control.

    This follows the single-stage framework from the ControlPINN paper:
    the network is trained to simultaneously satisfy PDE constraints
    and minimize a cost objective.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        config: ControlPINNConfig,
        *,
        activation: Callable[[Array], Array] = jnp.tanh,
        rngs: nnx.Rngs,
    ):
        """Initialize ControlPINN.

        Args:
            input_dim: Input dimension (e.g., spatial + temporal dims).
            hidden_dims: Hidden layer sizes for the shared MLP.
            config: ControlPINN configuration.
            activation: Activation function for hidden layers.
            rngs: Random number generators.
        """
        super().__init__()
        self.config = config
        self.activation = activation

        total_outputs = config.n_state_outputs + config.n_control_outputs

        # Build shared MLP
        layers = []
        in_features = input_dim
        for h in hidden_dims:
            layers.append(nnx.Linear(in_features, h, rngs=rngs))
            in_features = h
        layers.append(nnx.Linear(in_features, total_outputs, rngs=rngs))
        self.layers = nnx.List(layers)

    def _forward_raw(
        self, x: Float[Array, "batch input_dim"]
    ) -> Float[Array, "batch total_outputs"]:
        """Forward pass through the shared MLP.

        Args:
            x: Input tensor (batch, input_dim).

        Returns:
            Raw output (batch, n_state + n_control).
        """
        h = x
        for layer in list(self.layers)[:-1]:
            h = self.activation(layer(h))
        return list(self.layers)[-1](h)

    def __call__(
        self, x: Float[Array, "batch input_dim"]
    ) -> tuple[Float[Array, "batch n_state"], Float[Array, "batch n_ctrl"]]:
        """Forward pass returning (state, control) tuple.

        Args:
            x: Input tensor (batch, input_dim).

        Returns:
            Tuple of (state, control) tensors.
        """
        raw = self._forward_raw(x)
        n_state = self.config.n_state_outputs
        state = raw[:, :n_state]
        control = raw[:, n_state:]
        return state, control

    def state_output(
        self, x: Float[Array, "batch input_dim"]
    ) -> Float[Array, "batch n_state"]:
        """Return only state variables.

        Args:
            x: Input tensor (batch, input_dim).

        Returns:
            State output (batch, n_state_outputs).
        """
        state, _ = self(x)
        return state

    def control_output(
        self, x: Float[Array, "batch input_dim"]
    ) -> Float[Array, "batch n_ctrl"]:
        """Return only control variables.

        Args:
            x: Input tensor (batch, input_dim).

        Returns:
            Control output (batch, n_control_outputs).
        """
        _, control = self(x)
        return control

    def compute_objective(
        self,
        x: Float[Array, "batch input_dim"],
        objective_fn: Callable,
    ) -> Float[Array, ""]:
        """Evaluate the cost functional.

        Args:
            x: Collocation points (batch, input_dim).
            objective_fn: Cost function with signature (state, control) -> scalar.

        Returns:
            Scalar objective value.
        """
        state, control = self(x)
        return objective_fn(state, control)

    def pde_residual(
        self,
        x: Float[Array, "batch input_dim"],
        pde_fn: Callable,
    ) -> Float[Array, ""]:
        """Compute PDE constraint violation (MSE of residual).

        Args:
            x: Collocation points (batch, input_dim).
            pde_fn: PDE residual function with signature (model, x) -> residual.
                The function receives the ControlPINN model and can query
                both state and control outputs.

        Returns:
            Scalar MSE of PDE residual.
        """
        residual = pde_fn(self, x)
        return jnp.mean(residual**2)

    def control_loss(
        self,
        x: Float[Array, "batch input_dim"],
        pde_fn: Callable,
        objective_fn: Callable,
    ) -> Float[Array, ""]:
        """Compute the total ControlPINN loss.

        L = J(u, c) + lambda_pde * ||R||^2 + lambda_ctrl * ||c||^2

        where:
            J = objective (cost functional)
            R = PDE residual
            c = control variables

        Args:
            x: Collocation points (batch, input_dim).
            pde_fn: PDE residual function (model, x) -> residual.
            objective_fn: Cost function (state, control) -> scalar.

        Returns:
            Scalar total loss.
        """
        # Objective term
        state, control = self(x)
        objective = objective_fn(state, control)

        # PDE constraint term
        pde_res = self.pde_residual(x, pde_fn)

        # Control regularization (L2)
        ctrl_reg = jnp.mean(control**2)

        return (
            objective
            + self.config.pde_weight * pde_res
            + self.config.control_weight * ctrl_reg
        )


def create_control_pinn(
    input_dim: int,
    hidden_dims: list[int],
    *,
    config: ControlPINNConfig | None = None,
    activation: Callable[[Array], Array] = jnp.tanh,
    rngs: nnx.Rngs,
) -> ControlPINN:
    """Create a ControlPINN model.

    Args:
        input_dim: Input dimension (spatial + temporal).
        hidden_dims: Hidden layer sizes for the shared MLP.
        config: ControlPINN configuration.
        activation: Activation function for hidden layers.
        rngs: Random number generators.

    Returns:
        Configured ControlPINN instance.
    """
    cfg = config or ControlPINNConfig()
    return ControlPINN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        config=cfg,
        activation=activation,
        rngs=rngs,
    )

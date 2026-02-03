"""Fractional Physics-Informed Neural Network (fPINN).

Inspired by the fPINN framework introduced in:
    Pang et al. "fPINNs: Fractional Physics-Informed Neural Networks"
    (2019, SIAM Journal on Scientific Computing)

The original paper uses the Grünwald-Letnikov (GL) formula for
fractional derivative discretization. This implementation uses the
L1 scheme for the Caputo fractional derivative, which is widely used
in subsequent fPINN literature and provides O(τ^{2-α}) accuracy.

Key ideas:
    - Caputo fractional derivative via L1 discretization scheme
    - Non-local derivative: incorporates history via convolution
    - Applications: anomalous diffusion, viscoelastic materials,
      fractional advection-dispersion equations

References:
    - L1 scheme: Sun & Wu, "A fully discrete difference scheme for a
      diffusion-wave system" (2006, Applied Numerical Mathematics)
    - fPINN concept: Pang et al. (2019, SIAM J. Sci. Comput.)
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import gamma

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


def caputo_derivative_l1(
    f_vals: Array,
    t: Array,
    alpha: float,
) -> Array:
    """Compute Caputo fractional derivative using the L1 scheme.

    The Caputo derivative of order alpha (0 < alpha < 1) is:

        D^α f(t) = 1/Γ(1-α) ∫₀ᵗ f'(s) / (t-s)^α ds

    The L1 discretization on a (possibly non-uniform) mesh gives:

        D^α f(tₙ) ≈ 1/Γ(2-α) Σⱼ₌₀ⁿ⁻¹ bⱼ [f(t_{n-j}) - f(t_{n-j-1})] / τⱼ^α

    where bⱼ = (j+1)^{1-α} - j^{1-α} for uniform meshes, and the
    general non-uniform weights are computed from interval endpoints.

    For alpha in (1, 2), we reduce to D^{alpha-1}(f') using finite
    differences for f'.

    Args:
        f_vals: Function values at time points, shape (N,).
        t: Time points, shape (N,), must be monotonically increasing.
        alpha: Fractional order, 0 < alpha <= 2.

    Returns:
        Approximate Caputo derivative at each time point, shape (N,).
    """
    n = len(t)

    if alpha > 1.0:
        # For alpha in (1,2), reduce: D^a f = D^{a-1}(f')
        dt = t[1:] - t[:-1]
        df = (f_vals[1:] - f_vals[:-1]) / dt
        t_mid = (t[1:] + t[:-1]) / 2.0
        result_interior = caputo_derivative_l1(df, t_mid, alpha - 1.0)
        return jnp.concatenate([jnp.zeros(1), result_interior])

    # L1 scheme for 0 < alpha < 1
    # Caputo: D^a f(t) = 1/Gamma(1-a) * int_0^t f'(s)/(t-s)^a ds
    # L1 approximation uses piecewise-linear f' and exact integral of kernel
    coeff = 1.0 / gamma(1.0 - alpha)

    def compute_at_k(k):
        """Compute D^a f at t_k using L1 weights."""
        if k == 0:
            return 0.0

        # Finite differences: [f(t_{j+1}) - f(t_j)] / (t_{j+1} - t_j)
        dt_arr = t[1 : k + 1] - t[:k]
        df_arr = (f_vals[1 : k + 1] - f_vals[:k]) / dt_arr

        # Exact integral weights for non-uniform mesh:
        # w_j = int_{t_j}^{t_{j+1}} (t_k - s)^{-a} ds
        #     = [(t_k-t_j)^{1-a} - (t_k-t_{j+1})^{1-a}] / (1-a)
        j_indices = jnp.arange(k)
        w_left = (t[k] - t[j_indices]) ** (1.0 - alpha)
        w_right = (t[k] - t[j_indices + 1]) ** (1.0 - alpha)
        weights = (w_left - w_right) / (1.0 - alpha)

        return coeff * jnp.sum(weights * df_arr)

    results = []
    for k in range(n):
        results.append(compute_at_k(k))
    return jnp.array(results)


@dataclass(frozen=True)
class FPINNConfig:
    """Configuration for fPINN.

    Attributes:
        alpha: Fractional derivative order (0 < alpha <= 2).
        n_discretization: Number of points for L1 scheme discretization.
        hidden_dims: Default hidden layer dimensions.
    """

    alpha: float = 0.5
    n_discretization: int = 30
    hidden_dims: tuple[int, ...] = (64, 64, 64)


class FractionalPINN(nnx.Module):
    """Fractional Physics-Informed Neural Network.

    Solves fractional PDEs of the form:

        D_t^alpha u(x,t) = L[u](x,t) + f(x,t)

    where D_t^alpha is the Caputo fractional derivative in time
    and L is a spatial differential operator.

    The fractional time derivative is computed using the L1 discretization
    scheme, which evaluates the neural network at auxiliary time points
    and uses a history convolution.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        *,
        activation: Callable[[Array], Array] = jnp.tanh,
        config: FPINNConfig | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize fPINN.

        Args:
            input_dim: Input dimensionality (spatial coords + time).
            output_dim: Output dimensionality.
            hidden_dims: Hidden layer dimensions.
            activation: Activation function.
            config: fPINN configuration.
            rngs: Random number generators.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or FPINNConfig()
        self.activation = activation

        dims = list(hidden_dims or self.config.hidden_dims)

        # Build MLP
        layers = []
        in_features = input_dim
        for h in dims:
            layers.append(nnx.Linear(in_features, h, rngs=rngs))
            in_features = h
        layers.append(nnx.Linear(in_features, output_dim, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, x: Array) -> Array:
        """Forward pass: x -> u(x).

        Args:
            x: Input coordinates (batch_size, input_dim).

        Returns:
            Solution prediction (batch_size, output_dim).
        """
        h = x
        for layer in list(self.layers)[:-1]:
            h = self.activation(layer(h))
        return list(self.layers)[-1](h)

    def fractional_residual(
        self,
        x_colloc: Array,
        t_grid: Array,
    ) -> Array:
        """Compute fractional PDE residual at collocation points.

        Evaluates D_t^alpha u(x, t) at each collocation point using the
        L1 scheme. The residual is MSE of the fractional derivative
        (for the simple case D_t^alpha u = 0 as a baseline).

        Args:
            x_colloc: Spatial collocation points (batch, input_dim).
                Last column is treated as time coordinate.
            t_grid: Time grid for L1 scheme evaluation (n_t,).

        Returns:
            Scalar residual loss.
        """
        batch_size = x_colloc.shape[0]
        alpha = self.config.alpha

        residuals = []
        for i in range(batch_size):
            spatial = x_colloc[i, :-1]  # (input_dim - 1,)

            # Evaluate u at (spatial, t) for each t in t_grid
            def eval_at_t(t_val, sp=spatial):
                point = jnp.concatenate([sp, t_val[None]])
                return self(point[None, :])[0, 0]

            u_vals = jnp.array([eval_at_t(t) for t in t_grid])

            # Compute Caputo derivative
            d_alpha_u = caputo_derivative_l1(u_vals, t_grid, alpha)

            # Residual: D^alpha u should equal some RHS (0 for baseline)
            residuals.append(jnp.mean(d_alpha_u**2))

        return jnp.mean(jnp.array(residuals))


def create_fpinn(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int] | None = None,
    *,
    config: FPINNConfig | None = None,
    activation: Callable[[Array], Array] = jnp.tanh,
    rngs: nnx.Rngs,
) -> FractionalPINN:
    """Create an fPINN model.

    Args:
        input_dim: Input dimensionality.
        output_dim: Output dimensionality.
        hidden_dims: Hidden layer dimensions.
        config: fPINN configuration.
        activation: Activation function.
        rngs: Random number generators.

    Returns:
        Configured FractionalPINN instance.
    """
    return FractionalPINN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        config=config,
        activation=activation,
        rngs=rngs,
    )

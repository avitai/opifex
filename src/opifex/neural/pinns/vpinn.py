"""Variational Physics-Informed Neural Network (VPINN).

Implements the VPINN/hp-VPINN framework from:
    Kharazmi et al. "hp-VPINNs: Variational Physics-Informed Neural Networks
    With Domain Decomposition" (2021, CMAME)

Key ideas:
    - Weak form: multiply PDE by test functions, integrate by parts
    - Test functions: Legendre polynomials (orthogonal, good for spectral accuracy)
    - Integration: Gauss-Legendre quadrature for accurate numerical integrals
    - Advantage: reduces derivative order needed; better for non-smooth solutions
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


def gauss_legendre_quadrature(n: int) -> tuple[Array, Array]:
    """Compute Gauss-Legendre quadrature points and weights on [-1, 1].

    Uses the eigenvalue method for computing nodes and weights of
    the n-point Gauss-Legendre quadrature rule.

    Args:
        n: Number of quadrature points.

    Returns:
        Tuple of (points, weights), each of shape (n,).
    """
    # Companion matrix (Golub-Welsch algorithm)
    i = jnp.arange(1, n, dtype=jnp.float32)
    beta = i / jnp.sqrt(4.0 * i**2 - 1.0)

    # Build symmetric tridiagonal matrix
    companion = jnp.diag(beta, -1) + jnp.diag(beta, 1)

    # Eigenvalues = nodes, eigenvectors -> weights
    eigenvalues, eigenvectors = jnp.linalg.eigh(companion)

    points = eigenvalues
    weights = 2.0 * eigenvectors[0, :] ** 2

    # Sort by point location
    sort_idx = jnp.argsort(points)
    return points[sort_idx], weights[sort_idx]


def _legendre_polynomial(x: Array, n: int) -> Array:
    """Evaluate nth Legendre polynomial at points x.

    Uses the three-term recurrence relation:
        (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)

    Args:
        x: Evaluation points, arbitrary shape.
        n: Degree of the polynomial.

    Returns:
        P_n(x), same shape as x.
    """
    if n == 0:
        return jnp.ones_like(x)
    if n == 1:
        return x

    p_prev = jnp.ones_like(x)
    p_curr = x
    for k in range(1, n):
        p_next = ((2 * k + 1) * x * p_curr - k * p_prev) / (k + 1)
        p_prev = p_curr
        p_curr = p_next
    return p_curr


@dataclass(frozen=True)
class VPINNConfig:
    """Configuration for VPINN.

    Attributes:
        n_test_functions: Number of test functions (Legendre polynomials).
        n_quadrature_points: Number of Gauss-Legendre quadrature points.
        hidden_dims: Default hidden layer dimensions.
    """

    n_test_functions: int = 5
    n_quadrature_points: int = 15
    hidden_dims: tuple[int, ...] = (64, 64, 64)


class VPINN(nnx.Module):
    """Variational Physics-Informed Neural Network.

    Instead of enforcing the PDE residual at collocation points (strong form),
    VPINN enforces the weak form:

        integral[ L[u] * v_k dx ] = integral[ f * v_k dx ]

    where L is the PDE operator, u is the neural network, and v_k are test
    functions (Legendre polynomials). Integration is performed using
    Gauss-Legendre quadrature.

    Advantages over standard PINN:
    - Reduces derivative order by integration by parts
    - Better handles non-smooth solutions
    - Provides global consistency through integral formulation
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        *,
        activation: Callable[[Array], Array] = jnp.tanh,
        config: VPINNConfig | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize VPINN.

        Args:
            input_dim: Input dimensionality.
            output_dim: Output dimensionality.
            hidden_dims: Hidden layer dimensions.
            activation: Activation function.
            config: VPINN configuration.
            rngs: Random number generators.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or VPINNConfig()
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

    def variational_residual(
        self,
        pde_lhs_fn: Callable[["VPINN", Array], Array],
        *,
        domain: tuple[float, float] = (-1.0, 1.0),
        rhs_fn: Callable[[Array], Array] | None = None,
    ) -> Array:
        """Compute variational residuals for each test function.

        Evaluates: R_k = integral[ L[u](x) * v_k(x) dx ] - integral[ f(x) * v_k(x) dx ]

        for k = 1, ..., n_test_functions.

        Args:
            pde_lhs_fn: Function(model, x) -> L[u](x) at quadrature points.
                x has shape (n_quad, input_dim), returns (n_quad,).
            domain: Integration domain (a, b) for 1D problems.
            rhs_fn: Optional RHS forcing function f(x). If None, assumed zero.

        Returns:
            Variational residuals, shape (n_test_functions,).
        """
        n_quad = self.config.n_quadrature_points
        n_test = self.config.n_test_functions

        # Get quadrature points on [-1, 1] and map to [a, b]
        ref_pts, ref_wts = gauss_legendre_quadrature(n_quad)
        a, b = domain
        # Affine map: x = (b-a)/2 * xi + (a+b)/2
        scale = (b - a) / 2.0
        shift = (a + b) / 2.0
        quad_pts = scale * ref_pts + shift  # (n_quad,)
        quad_wts = scale * ref_wts  # scaled weights

        # Evaluate PDE LHS at quadrature points
        x_quad = quad_pts[:, None]  # (n_quad, 1) for 1D
        lhs_vals = pde_lhs_fn(self, x_quad)  # (n_quad,)

        rhs_vals = rhs_fn(x_quad) if rhs_fn is not None else jnp.zeros(n_quad)

        # Compute variational residuals for each test function
        residuals = []
        for k in range(n_test):
            # Evaluate test function v_k at reference points
            v_k = _legendre_polynomial(ref_pts, k + 1)  # (n_quad,)
            # Weighted integral: sum_j w_j * (L[u] - f)(x_j) * v_k(x_j)
            integrand = (lhs_vals - rhs_vals) * v_k
            residual_k = jnp.sum(quad_wts * integrand)
            residuals.append(residual_k)

        return jnp.array(residuals)

    def variational_loss(
        self,
        pde_lhs_fn: Callable[["VPINN", Array], Array],
        *,
        domain: tuple[float, float] = (-1.0, 1.0),
        rhs_fn: Callable[[Array], Array] | None = None,
    ) -> Array:
        """Compute variational loss: sum of squared variational residuals.

        L = sum_k R_k^2

        Args:
            pde_lhs_fn: PDE left-hand side function.
            domain: Integration domain.
            rhs_fn: Optional RHS forcing function.

        Returns:
            Scalar variational loss.
        """
        residuals = self.variational_residual(pde_lhs_fn, domain=domain, rhs_fn=rhs_fn)
        return jnp.sum(residuals**2)


def create_vpinn(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int] | None = None,
    *,
    config: VPINNConfig | None = None,
    activation: Callable[[Array], Array] = jnp.tanh,
    rngs: nnx.Rngs,
) -> VPINN:
    """Create a VPINN model.

    Args:
        input_dim: Input dimensionality.
        output_dim: Output dimensionality.
        hidden_dims: Hidden layer dimensions.
        config: VPINN configuration.
        activation: Activation function.
        rngs: Random number generators.

    Returns:
        Configured VPINN instance.
    """
    return VPINN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        config=config,
        activation=activation,
        rngs=rngs,
    )

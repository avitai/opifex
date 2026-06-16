r"""Diagonal Extended Kalman (EK1) probabilistic ODE step.

Carries a per-dimension Cholesky factor of the marginal state covariance
and uses the diagonal of the ODE-RHS Jacobian as a structured
linearisation. Each output dimension is updated independently, so the
solver scales linearly in the ODE dimension and remains JAX-friendly.

The unit step extrapolates the state through a (typically IBM) transition
matrix, evaluates the ODE residual ``z = m_pred[1] - f(m_pred[0])``,
estimates the local error and a per-dimension scale ``sigma``, and
applies a Kalman correction with the linearisation
``H = e_1^T - diag(df/dx) e_0^T`` per dimension.

Canonical reference (line-by-line port):
* ``../tornadox/tornadox/ek1.py`` — ``DiagonalEK1.attempt_unit_step``
  (line 274), ``evaluate_ode`` (line 304), ``estimate_error`` (line 316),
  ``observe_cov_sqrtm`` (line 335), ``correct_cov_sqrtm`` (line 353),
  ``correct_mean`` (line 366).

References
----------
* Krämer, Schmidt, Hennig 2022 — *Probabilistic ODE Solutions in Millions
  of Dimensions*, arXiv:2110.11812.
* Bosch, Tronarp, Hennig 2021 — *Pick-and-Mix Information Operators for
  Probabilistic ODE Solvers*, arXiv:2110.10770.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax
import jax.numpy as jnp


def _propagate_cholesky_factor(
    transition_times_factor: jax.Array, process_sqrt: jax.Array
) -> jax.Array:
    r"""Square-root propagate per-dimension Cholesky factors.

    Stacks ``[(transition L)^T; Q_sqrt^T]`` and reads off the upper
    triangle of a thin QR factorisation — the canonical
    ``batched_propagate_cholesky_factor`` from tornadox.
    """
    stacked = jnp.concatenate(
        [transition_times_factor.swapaxes(-1, -2), process_sqrt.swapaxes(-1, -2)], axis=-2
    )
    upper = jax.vmap(lambda block: jnp.linalg.qr(block, mode="reduced")[1])(stacked)
    return upper.swapaxes(-1, -2)


def diagonal_ek1_step(
    *,
    mean: jax.Array,
    cov_sqrt: jax.Array,
    transition: jax.Array,
    process_sqrt: jax.Array,
    drift: Callable[[jax.Array], jax.Array],
    jacobian_diagonal: Callable[[jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""One unit step of the diagonal Extended Kalman (EK1) ODE solver.

    Args:
        mean: State mean of shape ``(num_derivatives + 1, ode_dim)``. Row
            ``0`` holds the function value, row ``1`` the first derivative,
            and so on.
        cov_sqrt: Per-dimension lower-triangular Cholesky factor of the
            marginal covariance, shape
            ``(ode_dim, num_derivatives + 1, num_derivatives + 1)`` with
            ``P_d = L_d L_d^T``.
        transition: Discrete-time transition matrix ``Φ`` of shape
            ``(num_derivatives + 1, num_derivatives + 1)`` (typically the
            IBM(n) transition over ``dt``).
        process_sqrt: Cholesky factor of the discrete process noise,
            shape ``(num_derivatives + 1, num_derivatives + 1)``.
        drift: ODE right-hand side ``f(x)`` mapping ``(ode_dim,)`` to
            ``(ode_dim,)``. Time is assumed absorbed into ``drift``.
        jacobian_diagonal: Diagonal of ``df/dx`` mapping ``(ode_dim,)`` to
            ``(ode_dim,)``.

    Returns:
        ``(new_mean, new_cov_sqrt, error_estimate, sigma)``:
        - ``new_mean`` with shape ``(num_derivatives + 1, ode_dim)``;
        - ``new_cov_sqrt`` with shape ``(ode_dim, n + 1, n + 1)``;
        - ``error_estimate``/``sigma``: per-dimension ``(ode_dim,)`` arrays for step-size control.
    """
    state_dim = mean.shape[0]
    ode_dim = mean.shape[1]

    predicted_mean = transition @ mean
    transition_times_factor = jnp.einsum("ij,djk->dik", transition, cov_sqrt)
    process_sqrt_batch = jnp.broadcast_to(process_sqrt, (ode_dim, state_dim, state_dim))

    function_at = predicted_mean[0]
    derivative_at = predicted_mean[1]
    rhs_value = drift(function_at)
    residual = derivative_at - rhs_value
    jacobian_diag = jacobian_diagonal(function_at)

    # Predicted covariance sqrt before the correction (used both for error
    # estimate and for the corrected sqrt factor).
    predicted_cov_sqrt = _propagate_cholesky_factor(transition_times_factor, process_sqrt_batch)

    # Observation row H_d = [-J_d, 1, 0, ..., 0] selects (m_pred[1] - J_d m_pred[0]).
    sc_0 = predicted_cov_sqrt[:, 0, :]
    sc_1 = predicted_cov_sqrt[:, 1, :]
    h_sc = sc_1 - jacobian_diag[:, None] * sc_0

    innovation_variance = jnp.einsum("dn,dn->d", h_sc, h_sc) + 1e-16
    sigma_squared = residual**2 / innovation_variance
    sigma = jnp.sqrt(sigma_squared)
    error_estimate = sigma * jnp.sqrt(innovation_variance)

    gain = jnp.einsum("dij,dj->di", predicted_cov_sqrt, h_sc) / innovation_variance[:, None]

    correction = gain * residual[:, None]
    new_mean = predicted_mean - correction.T

    new_cov_sqrt = predicted_cov_sqrt - jnp.einsum("di,dj->dij", gain, h_sc)
    return new_mean, new_cov_sqrt, error_estimate, sigma

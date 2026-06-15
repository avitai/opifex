"""Tests for the diagonal Extended Kalman (EK1) probabilistic ODE step.

The diagonal EK1 solver runs a Kalman correction with a per-dimension
Jacobian diagonal, exploiting the structure of separable or
diagonal-Jacobian ODEs. Each output dimension carries an independent
Cholesky factor of the marginal covariance.

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

import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace import diagonal_ek1_step


def test_diagonal_ek1_step_shapes_match_inputs() -> None:
    """Returns mean ``(n+1, d)``, sqrt factor ``(d, n+1, n+1)``, errors ``(d,)``."""
    ode_dim = 3
    num_derivatives = 1
    state_dim = num_derivatives + 1
    mean = jnp.zeros((state_dim, ode_dim))
    cov_sqrt = jnp.broadcast_to(jnp.eye(state_dim), (ode_dim, state_dim, state_dim))
    transition = jnp.eye(state_dim)
    process_sqrt = 0.1 * jnp.eye(state_dim)

    def drift(state: jax.Array) -> jax.Array:
        return -state

    def jacobian_diagonal(state: jax.Array) -> jax.Array:
        del state
        return -jnp.ones(ode_dim)

    new_mean, new_cov_sqrt, error_estimate, sigma = diagonal_ek1_step(
        mean=mean,
        cov_sqrt=cov_sqrt,
        transition=transition,
        process_sqrt=process_sqrt,
        drift=drift,
        jacobian_diagonal=jacobian_diagonal,
    )
    assert new_mean.shape == (state_dim, ode_dim)
    assert new_cov_sqrt.shape == (ode_dim, state_dim, state_dim)
    assert error_estimate.shape == (ode_dim,)
    assert sigma.shape == (ode_dim,)


def test_diagonal_ek1_step_correction_drives_residual_to_zero() -> None:
    """After one step, the residual ``z = m'[1] - f(m'[0])`` is reduced.

    The EK1 correction Kalman-updates the state using ``z`` as innovation, so
    the corrected mean must reduce the residual norm to machine precision
    for a linear ODE (perfect Jacobian).
    """
    ode_dim = 2
    state_dim = 2
    initial_mean = jnp.asarray([[1.0, 2.0], [0.0, 0.0]])  # (n+1, d)
    cov_sqrt = jnp.broadcast_to(jnp.eye(state_dim), (ode_dim, state_dim, state_dim))
    transition = jnp.asarray([[1.0, 0.1], [0.0, 1.0]])
    process_sqrt = 0.01 * jnp.eye(state_dim)
    decay_rate = jnp.asarray([-0.5, -1.0])

    def drift(state: jax.Array) -> jax.Array:
        return decay_rate * state

    def jacobian_diagonal(state: jax.Array) -> jax.Array:
        del state
        return decay_rate

    new_mean, _, _, _ = diagonal_ek1_step(
        mean=initial_mean,
        cov_sqrt=cov_sqrt,
        transition=transition,
        process_sqrt=process_sqrt,
        drift=drift,
        jacobian_diagonal=jacobian_diagonal,
    )
    residual_after = new_mean[1] - drift(new_mean[0])
    assert jnp.max(jnp.abs(residual_after)) < 1e-5


def test_diagonal_ek1_step_posterior_sqrt_is_psd_reconstruction() -> None:
    """The implicit posterior covariance ``L_i L_i^T`` is PSD per dimension."""
    ode_dim = 2
    state_dim = 2
    initial_mean = jnp.asarray([[1.0, 0.5], [-0.1, 0.2]])
    cov_sqrt = jnp.broadcast_to(jnp.eye(state_dim), (ode_dim, state_dim, state_dim))
    transition = jnp.asarray([[1.0, 0.05], [0.0, 1.0]])
    process_sqrt = 0.02 * jnp.eye(state_dim)

    def drift(state: jax.Array) -> jax.Array:
        return -(state**2)

    def jacobian_diagonal(state: jax.Array) -> jax.Array:
        return -2.0 * state

    _, new_cov_sqrt, _, _ = diagonal_ek1_step(
        mean=initial_mean,
        cov_sqrt=cov_sqrt,
        transition=transition,
        process_sqrt=process_sqrt,
        drift=drift,
        jacobian_diagonal=jacobian_diagonal,
    )
    for dim in range(ode_dim):
        block = new_cov_sqrt[dim]
        reconstructed = block @ block.T
        symmetric = 0.5 * (reconstructed + reconstructed.T)
        eigenvalues = jnp.linalg.eigvalsh(symmetric)
        assert jnp.all(eigenvalues >= -1e-5), f"dim {dim}"


def test_diagonal_ek1_step_jit_compatible_and_differentiable() -> None:
    """``diagonal_ek1_step`` compiles under jit and is differentiable.

    The step is a building block for ODE-based hyperparameter learning;
    gradients must flow through ``drift`` and ``jacobian_diagonal``.
    """
    ode_dim = 2
    state_dim = 2
    transition = jnp.asarray([[1.0, 0.1], [0.0, 1.0]])
    process_sqrt = 0.01 * jnp.eye(state_dim)
    cov_sqrt = jnp.broadcast_to(jnp.eye(state_dim), (ode_dim, state_dim, state_dim))

    def step(parameters: jax.Array) -> jax.Array:
        def drift(state: jax.Array) -> jax.Array:
            return parameters * state

        def jacobian_diagonal(state: jax.Array) -> jax.Array:
            del state
            return parameters

        mean = jnp.asarray([[1.0, 1.0], [parameters[0], parameters[1]]])
        new_mean, _, _, _ = diagonal_ek1_step(
            mean=mean,
            cov_sqrt=cov_sqrt,
            transition=transition,
            process_sqrt=process_sqrt,
            drift=drift,
            jacobian_diagonal=jacobian_diagonal,
        )
        return jnp.sum(new_mean[0] ** 2)

    parameters = jnp.asarray([-0.5, -1.0])
    grad_value = jax.jit(jax.grad(step))(parameters)
    assert jnp.all(jnp.isfinite(grad_value))


def test_diagonal_ek1_step_error_estimate_is_nonnegative() -> None:
    """The local error estimate is a per-dimension non-negative array."""
    ode_dim = 3
    state_dim = 2
    initial_mean = jnp.zeros((state_dim, ode_dim)).at[0].set(jnp.asarray([0.5, -0.3, 1.0]))
    cov_sqrt = jnp.broadcast_to(jnp.eye(state_dim), (ode_dim, state_dim, state_dim))
    transition = jnp.asarray([[1.0, 0.1], [0.0, 1.0]])
    process_sqrt = 0.05 * jnp.eye(state_dim)

    def drift(state: jax.Array) -> jax.Array:
        return -2.0 * state

    def jacobian_diagonal(state: jax.Array) -> jax.Array:
        del state
        return -2.0 * jnp.ones(ode_dim)

    _, _, error_estimate, sigma = diagonal_ek1_step(
        mean=initial_mean,
        cov_sqrt=cov_sqrt,
        transition=transition,
        process_sqrt=process_sqrt,
        drift=drift,
        jacobian_diagonal=jacobian_diagonal,
    )
    assert jnp.all(error_estimate >= 0.0)
    assert jnp.all(sigma >= 0.0)


def test_diagonal_ek1_step_handles_independent_dimensions() -> None:
    """Each ODE dimension is updated independently — different rates,
    different posteriors.
    """
    ode_dim = 2
    state_dim = 2
    initial_mean = jnp.asarray([[1.0, 1.0], [0.0, 0.0]])
    cov_sqrt = jnp.broadcast_to(jnp.eye(state_dim), (ode_dim, state_dim, state_dim))
    transition = jnp.asarray([[1.0, 0.1], [0.0, 1.0]])
    process_sqrt = 0.01 * jnp.eye(state_dim)

    def drift(state: jax.Array) -> jax.Array:
        # Different rates per dimension: dimension 0 decays slowly,
        # dimension 1 decays fast.
        return jnp.asarray([-0.1, -10.0]) * state

    def jacobian_diagonal(state: jax.Array) -> jax.Array:
        del state
        return jnp.asarray([-0.1, -10.0])

    new_mean, new_cov_sqrt, _, _ = diagonal_ek1_step(
        mean=initial_mean,
        cov_sqrt=cov_sqrt,
        transition=transition,
        process_sqrt=process_sqrt,
        drift=drift,
        jacobian_diagonal=jacobian_diagonal,
    )
    # The two dimensions must produce different posteriors (different rates).
    assert not jnp.allclose(new_mean[0, 0], new_mean[0, 1])
    assert not jnp.allclose(new_cov_sqrt[0], new_cov_sqrt[1])

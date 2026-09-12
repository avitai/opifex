"""Tests for the continuous-time SDE → discrete-time transition utilities.

Coverage:

* zero-drift recovers the identity transition and ``Q(dt) = Q_c dt``;
* scalar OU process matches the analytic closed form;
* dispersion ``L`` correctly maps a low-rank Wiener process into the state;
* default-identity diffusion matches an explicit identity matrix;
* the joint matrix exponential is mathematically equivalent to direct
  computation of ``A = expm(F dt)`` for the transition block;
* the integrated process noise is symmetric positive semi-definite;
* the operator is jit-compatible and differentiable w.r.t. ``dt`` (needed
  for SDE hyperparameter learning);
* the result is vmap-compatible across an array of step sizes (the typical
  multi-step pattern in Kalman filtering).

Canonical reference (line-by-line port):
* ``../probnum/src/probnum/randprocs/markov/continuous/_mfd.py``
  ``matrix_fraction_decomposition`` — Van Loan (1978) Theorem 1.

References
----------
* Van Loan, C. F. 1978 — *Computing integrals involving the matrix
  exponential*, IEEE TAC 23(3).
* Särkkä & Solin 2019 — *Applied Stochastic Differential Equations*
  §6.2 eqn 6.18.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import expm

from opifex.uncertainty.statespace import (
    discretize_lti_sde,
    matern12_kernel,
    matern32_kernel,
    matern52_kernel,
    matern72_kernel,
)


def test_zero_drift_recovers_identity_transition_and_qc_dt() -> None:
    """For ``F = 0`` the transition is the identity and ``Q(dt) = L Qc L^T dt``."""
    n = 3
    drift = jnp.zeros((n, n))
    dispersion = jnp.eye(n)
    diffusion = jnp.diag(jnp.asarray([1.0, 0.5, 0.25]))
    dt = jnp.asarray(0.1)

    transition, process_noise = discretize_lti_sde(
        drift_matrix=drift,
        dispersion_matrix=dispersion,
        diffusion=diffusion,
        dt=dt,
    )
    assert jnp.allclose(transition, jnp.eye(n), atol=1e-6)
    assert jnp.allclose(process_noise, diffusion * dt, atol=1e-6)


def test_scalar_decay_sde_matches_closed_form() -> None:
    """For ``dx = -a x dt + sigma dw`` the closed form is
    ``A = exp(-a dt)`` and ``Q = sigma^2 (1 - exp(-2 a dt)) / (2 a)``.
    """
    decay = 0.7
    sigma = 0.5
    dt = jnp.asarray(0.3)
    drift = jnp.asarray([[-decay]])
    dispersion = jnp.asarray([[sigma]])
    diffusion = jnp.asarray([[1.0]])

    transition, process_noise = discretize_lti_sde(
        drift_matrix=drift,
        dispersion_matrix=dispersion,
        diffusion=diffusion,
        dt=dt,
    )
    expected_transition = jnp.exp(-decay * dt)
    expected_q = sigma**2 * (1.0 - jnp.exp(-2.0 * decay * dt)) / (2.0 * decay)
    assert jnp.allclose(transition, jnp.asarray([[expected_transition]]), atol=1e-6)
    assert jnp.allclose(process_noise, jnp.asarray([[expected_q]]), atol=1e-6)


def test_transition_matches_direct_expm() -> None:
    """The block-exponential's top-left block equals ``expm(F dt)`` directly."""
    drift = jnp.asarray([[-0.5, 1.0], [-0.2, -0.7]])
    dispersion = jnp.eye(2)
    dt = jnp.asarray(0.4)
    transition, _ = discretize_lti_sde(drift_matrix=drift, dispersion_matrix=dispersion, dt=dt)
    assert jnp.allclose(transition, expm(drift * dt), atol=1e-5)


def test_default_diffusion_is_identity() -> None:
    """Calling without ``diffusion`` is equivalent to ``Qc = I``."""
    drift = jnp.asarray([[-0.5, 0.0], [0.0, -0.5]])
    dispersion = jnp.eye(2)
    dt = jnp.asarray(0.1)

    transition_default, q_default = discretize_lti_sde(
        drift_matrix=drift, dispersion_matrix=dispersion, dt=dt
    )
    transition_explicit, q_explicit = discretize_lti_sde(
        drift_matrix=drift,
        dispersion_matrix=dispersion,
        diffusion=jnp.eye(2),
        dt=dt,
    )
    assert jnp.allclose(transition_default, transition_explicit, atol=1e-7)
    assert jnp.allclose(q_default, q_explicit, atol=1e-7)


def test_dispersion_routes_low_rank_noise() -> None:
    """``L`` selects which state components receive Brownian noise."""
    drift = jnp.zeros((3, 3))
    dispersion = jnp.asarray([[0.0], [1.0], [0.0]])  # noise enters only x_1
    dt = jnp.asarray(0.5)
    _, process_noise = discretize_lti_sde(drift_matrix=drift, dispersion_matrix=dispersion, dt=dt)
    expected = jnp.zeros((3, 3)).at[1, 1].set(0.5)
    assert jnp.allclose(process_noise, expected, atol=1e-6)


def test_process_noise_is_symmetric_positive_semidefinite() -> None:
    """``Q(dt)`` must be symmetric and PSD for any well-posed LTI SDE."""
    drift = jnp.asarray([[-1.0, 0.5], [-0.2, -0.8]])
    dispersion = jnp.eye(2)
    diffusion = jnp.asarray([[2.0, 0.3], [0.3, 1.5]])
    dt = jnp.asarray(0.2)

    _, process_noise = discretize_lti_sde(
        drift_matrix=drift,
        dispersion_matrix=dispersion,
        diffusion=diffusion,
        dt=dt,
    )
    symmetrized = 0.5 * (process_noise + process_noise.T)
    assert jnp.allclose(process_noise, symmetrized, atol=1e-5)
    eigenvalues = jnp.linalg.eigvalsh(symmetrized)
    assert jnp.all(eigenvalues >= -1e-6)


def test_zero_dt_returns_identity_and_zero_noise() -> None:
    """At ``dt = 0`` the transition is the identity and process noise vanishes."""
    drift = jnp.asarray([[-1.0, 0.0], [0.0, -1.0]])
    dispersion = jnp.eye(2)
    transition, process_noise = discretize_lti_sde(
        drift_matrix=drift, dispersion_matrix=dispersion, dt=jnp.asarray(0.0)
    )
    assert jnp.allclose(transition, jnp.eye(2), atol=1e-7)
    assert jnp.allclose(process_noise, jnp.zeros((2, 2)), atol=1e-7)


def test_brownian_motion_recovers_qc_dt() -> None:
    """For ``dx = dW`` (F = 0, L = I) we get ``Q(dt) = I dt``."""
    n = 4
    drift = jnp.zeros((n, n))
    dispersion = jnp.eye(n)
    dt = jnp.asarray(0.37)
    transition, process_noise = discretize_lti_sde(
        drift_matrix=drift, dispersion_matrix=dispersion, dt=dt
    )
    assert jnp.allclose(transition, jnp.eye(n), atol=1e-7)
    assert jnp.allclose(process_noise, jnp.eye(n) * dt, atol=1e-7)


def test_discretize_lti_sde_jit_compatible_and_differentiable() -> None:
    """``discretize_lti_sde`` works under ``jax.jit`` and is differentiable
    w.r.t. ``dt`` (critical for SDE-based hyperparameter learning).
    """
    drift = jnp.asarray([[-0.4]])
    dispersion = jnp.asarray([[0.7]])
    diffusion = jnp.asarray([[1.0]])

    def transition_norm(dt: jax.Array) -> jax.Array:
        transition, _ = discretize_lti_sde(
            drift_matrix=drift,
            dispersion_matrix=dispersion,
            diffusion=diffusion,
            dt=dt,
        )
        return jnp.sum(transition**2)

    dt = jnp.asarray(0.25)
    jitted = jax.jit(transition_norm)
    grad_fn = jax.jit(jax.grad(transition_norm))
    value = jitted(dt)
    gradient = grad_fn(dt)
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_discretize_lti_sde_vmap_over_dt_array() -> None:
    """``vmap`` over a sequence of step sizes gives a stack of transitions."""
    drift = jnp.asarray([[-0.4, 0.1], [0.0, -0.6]])
    dispersion = jnp.eye(2)
    dt_array = jnp.linspace(0.05, 0.5, 6)

    def step(dt: jax.Array) -> tuple[jax.Array, jax.Array]:
        return discretize_lti_sde(drift_matrix=drift, dispersion_matrix=dispersion, dt=dt)

    transitions, process_noises = jax.vmap(step)(dt_array)
    assert transitions.shape == (6, 2, 2)
    assert process_noises.shape == (6, 2, 2)
    # At dt[0] (small), transition close to identity; at dt[-1] (larger), decay.
    assert jnp.linalg.norm(transitions[0] - jnp.eye(2)) < jnp.linalg.norm(
        transitions[-1] - jnp.eye(2)
    )


def test_discretize_lti_sde_differentiable_through_drift() -> None:
    """The transition is differentiable w.r.t. the drift matrix."""
    dispersion = jnp.eye(2)
    dt = jnp.asarray(0.2)

    def transition_trace(drift: jax.Array) -> jax.Array:
        transition, _ = discretize_lti_sde(drift_matrix=drift, dispersion_matrix=dispersion, dt=dt)
        return jnp.trace(transition)

    grad_value = jax.grad(transition_trace)(jnp.asarray([[-0.5, 0.0], [0.0, -0.7]]))
    assert jnp.all(jnp.isfinite(grad_value))


# ---------------------------------------------------------------------------
# Coarse steps: the block exponential must not overflow.
# ---------------------------------------------------------------------------

# (factory, variance, lengthscale) for stationary Matern SDEs, whose closed-form discretisation
# (``StateSpaceKernel.discretize``) is the float32 reference.
_MATERN_SDES = {
    "matern12": (matern12_kernel, 1.7, 0.6),
    "matern32": (matern32_kernel, 1.2, 0.9),
    "matern52": (matern52_kernel, 0.8, 1.1),
    "matern72": (matern72_kernel, 1.4, 0.7),
}
# Steps in lengthscales, from far below to far above the correlation time.
_COARSE_STEP_RATIOS = (1e-4, 1e-2, 1.0, 10.0, 100.0, 1e3)
# Float32 errors of the doubled block exponential, measured over these kernels and steps: process
# noise at most 3.6e-6 relative, transition at most 2.0e-6 of max(1, |A|). The single-step block
# exponential is non-finite at 100 lengthscales and off by 0.89 for Matern-5/2 at 10.
_MATERN_TOLERANCE = 5e-5
# Measured for integrated Wiener processes of order 1 to 3 at steps 1e-3 to 100: at most 1.0e-7 for
# the process noise and 6.1e-8 for the transition. The single-step block reaches 5.0e-4 at order 3.
_INTEGRATED_WIENER_TOLERANCE = 1e-5


def _integrated_wiener_process(
    order: int, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(F, L L^T, A, Q)`` for the order-``q`` integrated Wiener process in closed form.

    ``A_ij = dt^(j-i) / (j-i)!`` and ``Q_ij = dt^(2q+1-i-j) / ((2q+1-i-j) (q-i)! (q-j)!)``.
    """
    size = order + 1
    drift = np.diag(np.ones(size - 1), k=1)
    dispersion = np.zeros((size, 1))
    dispersion[-1, 0] = 1.0
    transition = np.zeros((size, size))
    process_noise = np.zeros((size, size))
    for row in range(size):
        for column in range(size):
            if column >= row:
                transition[row, column] = dt ** (column - row) / math.factorial(column - row)
            power = 2 * order + 1 - row - column
            process_noise[row, column] = dt**power / (
                power * math.factorial(order - row) * math.factorial(order - column)
            )
    return drift, dispersion @ dispersion.T, transition, process_noise


@pytest.mark.parametrize("name", list(_MATERN_SDES))
def test_matern_discretisation_is_accurate_from_fine_to_coarse_steps(name: str) -> None:
    """Transition and process noise match the closed-form Matern discretisation at every step."""
    factory, variance, lengthscale = _MATERN_SDES[name]
    kernel = factory(variance=variance, lengthscale=lengthscale)
    for ratio in _COARSE_STEP_RATIOS:
        dt = jnp.asarray(ratio * lengthscale)
        transition, process_noise = discretize_lti_sde(
            drift_matrix=kernel.feedback,
            dispersion_matrix=kernel.noise_effect,
            dt=dt,
            diffusion=kernel.diffusion,
        )
        reference_transition, reference_noise = kernel.discretize(dt)
        assert bool(jnp.all(jnp.isfinite(transition))), (name, ratio)
        assert bool(jnp.all(jnp.isfinite(process_noise))), (name, ratio)
        transition_scale = max(1.0, float(jnp.max(jnp.abs(reference_transition))))
        transition_error = (
            float(jnp.max(jnp.abs(transition - reference_transition))) / transition_scale
        )
        noise_error = float(
            jnp.linalg.norm(process_noise - reference_noise) / jnp.linalg.norm(reference_noise)
        )
        assert transition_error <= _MATERN_TOLERANCE, (name, ratio, transition_error)
        assert noise_error <= _MATERN_TOLERANCE, (name, ratio, noise_error)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_integrated_wiener_process_matches_its_closed_form(order: int) -> None:
    """A non-stationary SDE, with no stationary covariance to fall back on, stays exact."""
    for step in (1e-3, 1.0, 10.0, 100.0):
        drift, noise, reference_transition, reference_noise = _integrated_wiener_process(
            order, step
        )
        transition, process_noise = discretize_lti_sde(
            drift_matrix=jnp.asarray(drift, dtype=jnp.float32),
            dispersion_matrix=jnp.eye(order + 1),
            dt=jnp.asarray(step),
            diffusion=jnp.asarray(noise, dtype=jnp.float32),
        )
        transition_error = float(
            np.abs(np.asarray(transition, np.float64) - reference_transition).max()
            / np.abs(reference_transition).max()
        )
        noise_error = float(
            np.linalg.norm(np.asarray(process_noise, np.float64) - reference_noise)
            / np.linalg.norm(reference_noise)
        )
        assert transition_error <= _INTEGRATED_WIENER_TOLERANCE, (order, step, transition_error)
        assert noise_error <= _INTEGRATED_WIENER_TOLERANCE, (order, step, noise_error)


def test_discretisation_is_differentiable_at_coarse_steps() -> None:
    """Gradients with respect to the step and the drift stay finite far beyond the lengthscale."""
    kernel = matern52_kernel(variance=0.8, lengthscale=1.1)

    def total_noise(dt: jax.Array, drift: jax.Array) -> jax.Array:
        _, process_noise = discretize_lti_sde(
            drift_matrix=drift,
            dispersion_matrix=kernel.noise_effect,
            dt=dt,
            diffusion=kernel.diffusion,
        )
        return jnp.sum(process_noise)

    for step in (1e-4, 110.0, 1100.0):
        gradient_dt, gradient_drift = jax.grad(total_noise, argnums=(0, 1))(
            jnp.asarray(step), kernel.feedback
        )
        assert bool(jnp.isfinite(gradient_dt)), step
        assert bool(jnp.all(jnp.isfinite(gradient_drift))), step


def test_steps_beyond_the_doubling_limit_return_nan() -> None:
    """A step needing more than the supported number of doublings is reported as NaN."""
    kernel = matern52_kernel(variance=0.8, lengthscale=1.1)
    transition, process_noise = discretize_lti_sde(
        drift_matrix=kernel.feedback,
        dispersion_matrix=kernel.noise_effect,
        dt=jnp.asarray(1e12),
        diffusion=kernel.diffusion,
    )
    assert bool(jnp.all(jnp.isnan(transition)))
    assert bool(jnp.all(jnp.isnan(process_noise)))

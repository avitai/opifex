"""Tests for the continuous-time SDE → discrete-time transition utilities.

Coverage:

* closed forms: zero drift, a scalar Ornstein-Uhlenbeck process, Brownian motion and a zero step;
* the dispersion routes low-rank noise, and an omitted diffusion means the identity;
* the transition equals ``scipy.linalg.expm(F dt)``;
* per-component accuracy of the transition and the process noise against extended-precision
  references for Matern, integrated Wiener and integrated Ornstein-Uhlenbeck SDEs, at steps from
  1e-4 to 1e4, in float32 and float64;
* a singular positive semi-definite diffusion, an isotropic diffusion under ``jax.grad``, and an
  empty dispersion;
* jit, vmap and reverse-mode gradients, including coarse steps and unstable drifts, with one
  compilation per precision;
* steps beyond the supported number of doublings return NaN.

The discretisation is the exponential-and-Gramian doubling of Stillfjord & Tronarp
(arXiv:2310.13462).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm as scipy_expm

from opifex.uncertainty.statespace import discretize_lti_sde, matern52_kernel
from tests.uncertainty.statespace._accuracy import scaled_error, transition_error
from tests.uncertainty.statespace._process_noise_references import REFERENCES


_PRECISIONS = ("float32", "float64")
# Worst scaled process-noise error max |Q_ij - R_ij| / sqrt(R_ii R_jj) of the Stillfjord-Tronarp
# order-9 discretisation over the references: 1.7e-5 in float32 and 1.1e-13 in float64, both for
# Matern-7/2 at lambda dt = 1e4.
_NOISE_TOLERANCE = {"float32": 5e-5, "float64": 1e-12}
# Worst transition error max |A_ij - R_ij| / max(1, max |R|), measured the same way: 2.2e-6 in
# float32 and 4.9e-14 in float64, both for Matern-7/2 at lambda dt = 1.
_TRANSITION_TOLERANCE = {"float32": 1e-4, "float64": 1e-12}
# The order-4 integrated Wiener process at dt = 1e-4 has Q_00 = dt^9 / (9 * 4!^2), about 1.9e-40,
# below the smallest normal float32 number (1.2e-38); no float32 method can represent it.
_UNREPRESENTABLE = {("iwp4", 1e-4, "float32")}


def _dtype_for(request: pytest.FixtureRequest, precision: str) -> jnp.dtype:
    """Enable x64 for the float64 cases and return the dtype under test."""
    if precision == "float64":
        request.getfixturevalue("float64")
        return jnp.float64
    return jnp.float32


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


def test_transition_matches_scipy_expm() -> None:
    """The transition equals ``scipy.linalg.expm(F dt)``."""
    drift = np.asarray([[-0.5, 1.0], [-0.2, -0.7]])
    dt = 0.4
    transition, _ = discretize_lti_sde(
        drift_matrix=jnp.asarray(drift), dispersion_matrix=jnp.eye(2), dt=jnp.asarray(dt)
    )
    assert jnp.allclose(transition, jnp.asarray(scipy_expm(drift * dt)), atol=1e-6)


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
    assert jnp.array_equal(process_noise, process_noise.T)
    eigenvalues = jnp.linalg.eigvalsh(process_noise)
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
    assert jnp.allclose(process_noise, jnp.eye(n) * dt, atol=1e-6)


@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_discretisation_matches_extended_precision_references(
    request: pytest.FixtureRequest, name: str, precision: str
) -> None:
    """Every component of ``A`` and ``Q`` matches its reference, from 1e-4 to 1e4."""
    dtype = _dtype_for(request, precision)
    reference = REFERENCES[name]
    discretize = jax.jit(discretize_lti_sde)
    for step in reference.steps:
        if (name, step.scale, precision) in _UNREPRESENTABLE:
            continue
        transition, process_noise = discretize(
            drift_matrix=jnp.asarray(reference.drift, dtype=dtype),
            dispersion_matrix=jnp.asarray(reference.dispersion, dtype=dtype),
            dt=jnp.asarray(step.dt, dtype=dtype),
            diffusion=jnp.asarray(reference.diffusion, dtype=dtype),
        )
        assert transition.dtype == dtype
        assert process_noise.dtype == dtype
        transition_error_value = transition_error(transition, step.transition)
        noise_error = scaled_error(process_noise, step.process_noise)
        assert transition_error_value <= _TRANSITION_TOLERANCE[precision], (
            name,
            step.scale,
            transition_error_value,
        )
        assert noise_error <= _NOISE_TOLERANCE[precision], (name, step.scale, noise_error)


def test_singular_diffusion_matches_the_reference() -> None:
    """A rank-deficient positive semi-definite ``Q_c`` gives the same process noise as its factor.

    ``L = I`` with ``Q_c = e_n e_n^T`` describes the same SDE as ``L = e_n`` with ``Q_c = 1``.
    """
    reference = REFERENCES["iwp2"]
    dispersion = np.asarray(reference.dispersion)
    for step in reference.steps:
        _, process_noise = discretize_lti_sde(
            drift_matrix=jnp.asarray(reference.drift, dtype=jnp.float32),
            dispersion_matrix=jnp.eye(dispersion.shape[0]),
            dt=jnp.asarray(step.dt, dtype=jnp.float32),
            diffusion=jnp.asarray(dispersion @ dispersion.T, dtype=jnp.float32),
        )
        noise_error = scaled_error(process_noise, step.process_noise)
        assert noise_error <= _NOISE_TOLERANCE["float32"], (step.scale, noise_error)


@pytest.mark.usefixtures("float64")
def test_gradient_with_respect_to_an_isotropic_diffusion_is_exact() -> None:
    """``Q`` is linear in ``Q_c``, so ``d sum(Q(s I)) / ds`` is ``sum(Q(I))``.

    A repeated-eigenvalue diffusion is where an eigendecomposition-based factor loses its gradient.
    """
    drift = jnp.asarray([[-0.4, 0.3], [-0.2, -0.9]])
    dt = jnp.asarray(0.7)

    def total_noise(scale: jax.Array) -> jax.Array:
        _, process_noise = discretize_lti_sde(
            drift_matrix=drift, dispersion_matrix=jnp.eye(2), dt=dt, diffusion=scale * jnp.eye(2)
        )
        return jnp.sum(process_noise)

    gradient = jax.grad(total_noise)(jnp.asarray(2.5))
    expected = total_noise(jnp.asarray(1.0))
    assert float(abs(gradient - expected)) <= 1e-12 * float(abs(expected))


def test_empty_dispersion_gives_zero_process_noise() -> None:
    """A conservative SDE with no Wiener process has ``Q = 0`` and a rotation for ``A``."""
    frequency = 1.3
    drift = np.asarray([[0.0, -frequency], [frequency, 0.0]])
    dt = 0.9
    transition, process_noise = discretize_lti_sde(
        drift_matrix=jnp.asarray(drift),
        dispersion_matrix=jnp.zeros((2, 0)),
        dt=jnp.asarray(dt),
        diffusion=jnp.zeros((0, 0)),
    )
    assert jnp.allclose(transition, jnp.asarray(scipy_expm(drift * dt)), atol=1e-6)
    assert jnp.array_equal(process_noise, jnp.zeros((2, 2)))


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


def test_one_compilation_per_precision(request: pytest.FixtureRequest) -> None:
    """New step values reuse the compiled program; only a new dtype compiles again."""
    traced_dtypes: list[str] = []

    def discretize(dt: jax.Array) -> tuple[jax.Array, jax.Array]:
        traced_dtypes.append(str(dt.dtype))
        return discretize_lti_sde(
            drift_matrix=jnp.asarray([[-0.5, 1.0], [0.0, -0.3]], dtype=dt.dtype),
            dispersion_matrix=jnp.eye(2, dtype=dt.dtype),
            dt=dt,
        )

    jitted = jax.jit(discretize)
    for value in (0.1, 2.0, 500.0):
        jitted(jnp.asarray(value, dtype=jnp.float32))
    request.getfixturevalue("float64")
    for value in (0.1, 2.0):
        jitted(jnp.asarray(value, dtype=jnp.float64))
    assert traced_dtypes == ["float32", "float64"]


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


@pytest.mark.parametrize(
    ("rate", "dt"),
    [(1.0, 30.0), (1.0, 40.0), (100.0, 0.463)],
    ids=["rate1-dt30", "rate1-dt40", "rate100-overflowing-discarded-doubling"],
)
def test_gradients_stay_finite_for_unstable_drift(rate: float, dt: float) -> None:
    """A growing drift keeps finite values and gradients in float32.

    At rate 100 and dt 0.463 the process noise still fits in float32 (``Q_22`` about 8e37), but a
    doubling past the needed count would multiply the final factor by ``exp(46.3)`` and overflow.
    """

    def log_noise(drift_rate: jax.Array) -> jax.Array:
        drift = jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        drift = drift.at[2, 2].set(drift_rate)
        _, process_noise = discretize_lti_sde(
            drift_matrix=drift,
            dispersion_matrix=jnp.asarray([[0.0], [0.0], [1.0]]),
            dt=jnp.asarray(dt),
        )
        return jnp.sum(jnp.log(jnp.diag(process_noise)))

    value, gradient = jax.value_and_grad(log_noise)(jnp.asarray(rate))
    assert bool(jnp.isfinite(value)), value
    assert bool(jnp.isfinite(gradient)), gradient


def test_steps_beyond_the_doubling_limit_return_nan() -> None:
    """A step needing more than 32 doublings is reported as NaN."""
    kernel = matern52_kernel(variance=0.8, lengthscale=1.1)
    transition, process_noise = discretize_lti_sde(
        drift_matrix=kernel.feedback,
        dispersion_matrix=kernel.noise_effect,
        dt=jnp.asarray(1e12),
        diffusion=kernel.diffusion,
    )
    assert bool(jnp.all(jnp.isnan(transition)))
    assert bool(jnp.all(jnp.isnan(process_noise)))

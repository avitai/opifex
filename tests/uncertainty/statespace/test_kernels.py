"""Tests for state-space kernels.

State-space kernels expose the continuous-time linear SDE
``(F, L, Q_c, H, P_inf)`` of a temporal GP prior and the closed-form
discrete-time state transition ``A(dt) = exp(F dt)`` per Särkkä & Solin
2019 Table 12.2.

Coverage targets every Matern kernel (Matern12/32/52/72) on:

* shape/state-dim consistency;
* closed-form ``A(dt)`` matches the dense ``expm(F dt)`` reference;
* the semigroup property ``A(dt1 + dt2) = A(dt2) @ A(dt1)``;
* the boundary ``A(0) = I``;
* the Lyapunov equation ``F P_inf + P_inf F^T + L Q_c L^T = 0``;
* the measurement matrix selects the function value;
* jit + grad compatibility w.r.t. ``dt`` and hyperparameters.

For ``Cosine`` and ``Periodic`` we additionally exercise orthogonality of
the closed-form transition and its periodicity. ``QuasiPeriodicMatern12``
exercises the Kronecker-product structure.

Canonical reference (line-by-line port):
* ``../bayesnewton/bayesnewton/kernels.py`` — ``Matern12`` (line 141),
  ``Matern32`` (line 200), ``Matern52`` (line 253), ``Matern72`` (line
  321), ``Cosine`` (line 770), ``Periodic`` (line 802),
  ``QuasiPeriodicMatern12`` (line 882).

References
----------
* Särkkä & Solin 2019 — *Applied Stochastic Differential Equations* §12.3.
* Hartikainen & Särkkä 2010 — *Kalman filtering and smoothing solutions to
  temporal Gaussian process regression models*, MLSP.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import expm

from opifex.uncertainty.statespace import (
    cosine_kernel,
    matern12_kernel,
    matern32_kernel,
    matern52_kernel,
    matern72_kernel,
    periodic_kernel,
    quasi_periodic_matern12_kernel,
    StateSpaceKernel,
)


# Each entry: (factory, expected_state_dim, kwargs).
MATERN_KERNELS: list[tuple[str, object, int, dict[str, float]]] = [
    ("matern12", matern12_kernel, 1, {"variance": 1.7, "lengthscale": 0.6}),
    ("matern32", matern32_kernel, 2, {"variance": 1.2, "lengthscale": 0.9}),
    ("matern52", matern52_kernel, 3, {"variance": 0.8, "lengthscale": 1.1}),
    ("matern72", matern72_kernel, 4, {"variance": 1.4, "lengthscale": 0.7}),
]


def _is_psd(matrix: jax.Array, atol: float) -> bool:
    """Return whether ``matrix`` is symmetric positive semi-definite."""
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = jnp.linalg.eigvalsh(symmetric)
    return bool(jnp.all(eigenvalues >= -atol))


def _lyapunov_residual(kernel: StateSpaceKernel) -> jax.Array:
    """Compute ``F P_inf + P_inf F^T + L Q_c L^T`` (zero at stationarity)."""
    return (
        kernel.feedback @ kernel.stationary_cov
        + kernel.stationary_cov @ kernel.feedback.T
        + kernel.noise_effect @ kernel.diffusion @ kernel.noise_effect.T
    )


# ---------------------------------------------------------------------------
# Matern kernels — shared structural assertions parametrised across smoothness.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "factory", "expected_dim", "params"), MATERN_KERNELS)
def test_matern_kernel_shapes(name, factory, expected_dim, params) -> None:
    """Each Matern kernel exposes the expected ``(F, L, Qc, H, P_inf)`` shapes."""
    kernel = factory(**params)
    assert kernel.state_dim == expected_dim, name
    assert kernel.feedback.shape == (expected_dim, expected_dim)
    assert kernel.noise_effect.shape == (expected_dim, 1)
    assert kernel.diffusion.shape == (1, 1)
    assert kernel.measurement.shape == (1, expected_dim)
    assert kernel.stationary_cov.shape == (expected_dim, expected_dim)


@pytest.mark.parametrize(("name", "factory", "expected_dim", "params"), MATERN_KERNELS)
def test_matern_state_transition_closed_form_matches_expm(
    name, factory, expected_dim, params
) -> None:
    """``A(dt)`` from closed form equals ``expm(F dt)`` for each Matern kernel."""
    kernel = factory(**params)
    for dt_value in (0.05, 0.3, 1.0):
        dt = jnp.asarray(dt_value)
        closed_form = kernel.state_transition(dt)
        reference = expm(kernel.feedback * dt)
        assert closed_form.shape == (expected_dim, expected_dim)
        assert jnp.allclose(closed_form, reference, atol=1e-4), (name, dt_value)


@pytest.mark.parametrize(("name", "factory", "expected_dim", "params"), MATERN_KERNELS)
def test_matern_state_transition_identity_at_zero_dt(name, factory, expected_dim, params) -> None:
    """At ``dt = 0`` the state transition is the identity matrix."""
    kernel = factory(**params)
    assert jnp.allclose(kernel.state_transition(jnp.asarray(0.0)), jnp.eye(expected_dim), atol=1e-6)


@pytest.mark.parametrize(("name", "factory", "expected_dim", "params"), MATERN_KERNELS)
def test_matern_state_transition_semigroup_property(name, factory, expected_dim, params) -> None:
    """``A(dt1 + dt2) = A(dt2) @ A(dt1)`` (semigroup / Markov property)."""
    kernel = factory(**params)
    dt1 = jnp.asarray(0.2)
    dt2 = jnp.asarray(0.35)
    combined = kernel.state_transition(dt1 + dt2)
    composed = kernel.state_transition(dt2) @ kernel.state_transition(dt1)
    assert jnp.allclose(combined, composed, atol=1e-4), name


@pytest.mark.parametrize(("name", "factory", "expected_dim", "params"), MATERN_KERNELS)
def test_matern_state_transition_decays_for_large_dt(name, factory, expected_dim, params) -> None:
    """Stable Matern dynamics decay to zero for large step sizes."""
    kernel = factory(**params)
    far_future = kernel.state_transition(jnp.asarray(50.0))
    assert jnp.all(jnp.isfinite(far_future))
    assert jnp.max(jnp.abs(far_future)) < 1e-3, name


@pytest.mark.parametrize(("name", "factory", "expected_dim", "params"), MATERN_KERNELS)
def test_matern_stationary_covariance_solves_lyapunov(name, factory, expected_dim, params) -> None:
    """``F P_inf + P_inf F^T + L Q_c L^T = 0`` — Lyapunov equation residual scales
    inversely with ``lengthscale**(2 m + 1)`` for Matern-(m+1/2), so test the
    relative residual rather than an absolute tolerance.
    """
    kernel = factory(**params)
    residual = _lyapunov_residual(kernel)
    scale = jnp.linalg.norm(kernel.feedback) * jnp.linalg.norm(kernel.stationary_cov)
    relative = jnp.linalg.norm(residual) / (scale + 1e-12)
    assert relative < 1e-5, (name, float(relative))


@pytest.mark.parametrize(("name", "factory", "expected_dim", "params"), MATERN_KERNELS)
def test_matern_stationary_covariance_is_psd(name, factory, expected_dim, params) -> None:
    """``P_inf`` is symmetric positive semi-definite."""
    kernel = factory(**params)
    assert _is_psd(kernel.stationary_cov, atol=1e-6), name


@pytest.mark.parametrize(("name", "factory", "expected_dim", "params"), MATERN_KERNELS)
def test_matern_measurement_extracts_function_value(name, factory, expected_dim, params) -> None:
    """``H x`` selects the function value (first state component)."""
    kernel = factory(**params)
    expected = jnp.zeros((1, expected_dim)).at[0, 0].set(1.0)
    assert jnp.allclose(kernel.measurement, expected, atol=1e-7), name


@pytest.mark.parametrize(("name", "factory", "expected_dim", "params"), MATERN_KERNELS)
def test_matern_stationary_function_variance_equals_variance_hyperparameter(
    name, factory, expected_dim, params
) -> None:
    """``H P_inf H^T = sigma^2`` recovers the kernel's marginal variance."""
    kernel = factory(**params)
    marginal = (kernel.measurement @ kernel.stationary_cov @ kernel.measurement.T)[0, 0]
    assert jnp.allclose(marginal, params["variance"], atol=1e-5), name


@pytest.mark.parametrize(("name", "factory", "expected_dim", "params"), MATERN_KERNELS)
def test_matern_state_transition_is_jit_grad_compatible(
    name, factory, expected_dim, params
) -> None:
    """``state_transition`` is jit-compilable and differentiable w.r.t. ``dt``."""
    kernel = factory(**params)

    def norm_squared(dt: jax.Array) -> jax.Array:
        return jnp.sum(kernel.state_transition(dt) ** 2)

    value = jax.jit(norm_squared)(jnp.asarray(0.3))
    grad_value = jax.jit(jax.grad(norm_squared))(jnp.asarray(0.3))
    assert jnp.isfinite(value)
    assert jnp.isfinite(grad_value)


@pytest.mark.parametrize(("name", "factory", "expected_dim", "params"), MATERN_KERNELS)
def test_matern_kernel_differentiable_through_lengthscale(
    name, factory, expected_dim, params
) -> None:
    """Closed-form ``A(dt)`` is differentiable w.r.t. the lengthscale."""

    def lengthscale_objective(lengthscale: jax.Array) -> jax.Array:
        kernel_local = factory(variance=params["variance"], lengthscale=lengthscale)
        return jnp.sum(kernel_local.state_transition(jnp.asarray(0.2)) ** 2)

    grad_value = jax.grad(lengthscale_objective)(jnp.asarray(params["lengthscale"]))
    assert jnp.isfinite(grad_value)


@pytest.mark.parametrize(("name", "factory", "expected_dim", "params"), MATERN_KERNELS)
def test_matern_kernel_smaller_lengthscale_decays_faster(
    name, factory, expected_dim, params
) -> None:
    """A shorter lengthscale produces a faster-decaying first-row first-column."""
    kernel_short = factory(variance=params["variance"], lengthscale=0.1)
    kernel_long = factory(variance=params["variance"], lengthscale=2.0)
    dt = jnp.asarray(1.0)
    short_decay = jnp.abs(kernel_short.state_transition(dt)[0, 0])
    long_decay = jnp.abs(kernel_long.state_transition(dt)[0, 0])
    assert short_decay < long_decay, name


# ---------------------------------------------------------------------------
# Matern12 specific — exact closed form.
# ---------------------------------------------------------------------------


def test_matern12_state_transition_closed_form_exact() -> None:
    """Matern-1/2: ``A(dt) = exp(-dt/ell)`` to machine precision."""
    kernel = matern12_kernel(variance=2.0, lengthscale=0.4)
    dt = jnp.asarray(0.7)
    assert jnp.allclose(
        kernel.state_transition(dt),
        jnp.asarray([[jnp.exp(-dt / 0.4)]]),
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# Cosine kernel — orthogonality and periodicity.
# ---------------------------------------------------------------------------


def test_cosine_kernel_shape_and_state_dim() -> None:
    """Cosine kernel has state dim 2 and the expected feedback matrix."""
    kernel = cosine_kernel(frequency=1.5)
    assert kernel.state_dim == 2
    assert jnp.allclose(kernel.feedback, jnp.asarray([[0.0, -1.5], [1.5, 0.0]]))
    assert jnp.allclose(kernel.measurement, jnp.asarray([[1.0, 0.0]]))


def test_cosine_state_transition_is_orthogonal_rotation() -> None:
    """``A(dt) A(dt)^T = I`` — rotation matrix."""
    kernel = cosine_kernel(frequency=2.3)
    transition = kernel.state_transition(jnp.asarray(0.4))
    assert jnp.allclose(transition @ transition.T, jnp.eye(2), atol=1e-6)


def test_cosine_state_transition_identity_at_zero_dt() -> None:
    """``A(0) = I`` for the cosine kernel."""
    kernel = cosine_kernel(frequency=1.0)
    assert jnp.allclose(kernel.state_transition(jnp.asarray(0.0)), jnp.eye(2), atol=1e-7)


def test_cosine_state_transition_periodic() -> None:
    """``A(2 pi / omega) = I`` — full period returns to identity."""
    omega = 1.7
    kernel = cosine_kernel(frequency=omega)
    period_dt = jnp.asarray(2.0 * jnp.pi / omega)
    assert jnp.allclose(kernel.state_transition(period_dt), jnp.eye(2), atol=1e-5)


def test_cosine_state_transition_matches_expm() -> None:
    """Closed-form rotation matches ``expm(F dt)``."""
    omega = 1.2
    kernel = cosine_kernel(frequency=omega)
    dt = jnp.asarray(0.3)
    assert jnp.allclose(kernel.state_transition(dt), expm(kernel.feedback * dt), atol=1e-6)


# ---------------------------------------------------------------------------
# Periodic kernel — block structure, periodicity, PSD.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 3, 5])
def test_periodic_state_transition_block_diagonal_rotations(order: int) -> None:
    """``A(dt)`` is block-diagonal of harmonic rotation matrices."""
    period = 1.5
    kernel = periodic_kernel(variance=1.0, lengthscale=0.6, period=period, order=order)
    dt = jnp.asarray(0.2)
    transition = kernel.state_transition(dt)
    state_size = 2 * (order + 1)
    assert transition.shape == (state_size, state_size)
    omega = 2.0 * jnp.pi / period
    for j in range(order + 1):
        angle = j * omega * dt
        block = transition[2 * j : 2 * j + 2, 2 * j : 2 * j + 2]
        expected = jnp.asarray(
            [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
        )
        assert jnp.allclose(block, expected, atol=1e-6)
        # Off-block entries vanish.
        if j > 0:
            off_block = transition[2 * j : 2 * j + 2, 0:2]
            assert jnp.allclose(off_block, jnp.zeros((2, 2)), atol=1e-6)


def test_periodic_state_transition_periodic_at_full_period() -> None:
    """``A(period) = I`` — every harmonic returns to identity."""
    period = 0.9
    order = 3
    kernel = periodic_kernel(variance=1.0, lengthscale=0.5, period=period, order=order)
    assert jnp.allclose(
        kernel.state_transition(jnp.asarray(period)), jnp.eye(2 * (order + 1)), atol=1e-5
    )


def test_periodic_stationary_covariance_is_psd() -> None:
    """Periodic stationary cov (Bessel-weighted) is PSD."""
    kernel = periodic_kernel(variance=1.3, lengthscale=0.8, period=1.0, order=4)
    assert _is_psd(kernel.stationary_cov, atol=1e-6)


def test_periodic_state_transition_is_orthogonal() -> None:
    """The full periodic transition is orthogonal (block-diag of rotations)."""
    kernel = periodic_kernel(variance=1.0, lengthscale=0.7, period=1.0, order=4)
    dt = jnp.asarray(0.3)
    transition = kernel.state_transition(dt)
    state_size = transition.shape[0]
    assert jnp.allclose(transition @ transition.T, jnp.eye(state_size), atol=1e-5)


def test_periodic_kernel_jit_compatible() -> None:
    """Periodic ``state_transition`` works under ``jax.jit``."""
    kernel = periodic_kernel(variance=1.0, lengthscale=0.7, period=1.0, order=3)
    jitted = jax.jit(kernel.state_transition)
    transition = jitted(jnp.asarray(0.2))
    assert jnp.all(jnp.isfinite(transition))


# ---------------------------------------------------------------------------
# QuasiPeriodicMatern12 — Kronecker structure.
# ---------------------------------------------------------------------------


def test_quasi_periodic_state_dim_and_kronecker_structure() -> None:
    """QPM12 transition is the Kronecker of Matern12 and Periodic transitions."""
    order = 3
    variance = 1.0
    lengthscale_matern = 1.2
    lengthscale_periodic = 0.7
    period = 1.0
    kernel = quasi_periodic_matern12_kernel(
        variance=variance,
        lengthscale_periodic=lengthscale_periodic,
        period=period,
        lengthscale_matern=lengthscale_matern,
        order=order,
    )
    matern = matern12_kernel(variance=variance, lengthscale=lengthscale_matern)
    periodic = periodic_kernel(
        variance=1.0, lengthscale=lengthscale_periodic, period=period, order=order
    )

    dt = jnp.asarray(0.25)
    expected = jnp.kron(matern.state_transition(dt), periodic.state_transition(dt))
    assert jnp.allclose(kernel.state_transition(dt), expected, atol=1e-6)
    assert kernel.state_dim == 2 * (order + 1)


def test_quasi_periodic_identity_at_zero_dt() -> None:
    """QPM12 ``A(0) = I``."""
    kernel = quasi_periodic_matern12_kernel(
        variance=1.0,
        lengthscale_periodic=0.5,
        period=1.0,
        lengthscale_matern=2.0,
        order=2,
    )
    assert jnp.allclose(
        kernel.state_transition(jnp.asarray(0.0)),
        jnp.eye(2 * (2 + 1)),
        atol=1e-6,
    )


def test_quasi_periodic_state_transition_jit_compatible() -> None:
    """QPM12 ``state_transition`` works under ``jax.jit``."""
    kernel = quasi_periodic_matern12_kernel(
        variance=1.0,
        lengthscale_periodic=0.6,
        period=1.0,
        lengthscale_matern=1.0,
        order=2,
    )
    transition = jax.jit(kernel.state_transition)(jnp.asarray(0.2))
    assert jnp.all(jnp.isfinite(transition))


# ---------------------------------------------------------------------------
# StateSpaceKernel dataclass — invariants.
# ---------------------------------------------------------------------------


def test_state_space_kernel_is_immutable() -> None:
    """``StateSpaceKernel`` instances are frozen — attributes cannot be reassigned."""
    kernel = matern12_kernel(variance=1.0, lengthscale=1.0)
    with pytest.raises((AttributeError, TypeError)):
        kernel.feedback = jnp.zeros((1, 1))  # type: ignore[misc]

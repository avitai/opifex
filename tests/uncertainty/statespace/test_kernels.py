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

Every constructor satisfies the Lyapunov equation. The periodic and quasi-periodic
covariances ``H A(tau) P_inf H^T`` match their closed forms up to the dropped harmonics, and
the exponentially scaled modified Bessel functions behind the harmonic weights match
``scipy.special.ive`` in value and derivative.

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

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import expm
from scipy.linalg import expm as scipy_expm
from scipy.special import ive

from opifex.uncertainty.statespace import (
    cosine_kernel,
    discretize_lti_sde,
    matern12_kernel,
    matern32_kernel,
    matern52_kernel,
    matern72_kernel,
    periodic_kernel,
    quasi_periodic_matern12_kernel,
    StateSpaceKernel,
)
from opifex.uncertainty.statespace.kernels import i0e_vector, scaled_modified_bessel_i


if TYPE_CHECKING:
    from collections.abc import Callable


# Each entry: (factory, expected_state_dim, kwargs).
MATERN_KERNELS: list[tuple[str, object, int, dict[str, float]]] = [
    ("matern12", matern12_kernel, 1, {"variance": 1.7, "lengthscale": 0.6}),
    ("matern32", matern32_kernel, 2, {"variance": 1.2, "lengthscale": 0.9}),
    ("matern52", matern52_kernel, 3, {"variance": 0.8, "lengthscale": 1.1}),
    ("matern72", matern72_kernel, 4, {"variance": 1.4, "lengthscale": 0.7}),
]

# Every state-space kernel constructor.
ALL_KERNELS: list[tuple[str, Callable[[], StateSpaceKernel]]] = [
    ("matern12", lambda: matern12_kernel(variance=1.7, lengthscale=0.6)),
    ("matern32", lambda: matern32_kernel(variance=1.2, lengthscale=0.9)),
    ("matern52", lambda: matern52_kernel(variance=0.8, lengthscale=1.1)),
    ("matern72", lambda: matern72_kernel(variance=1.4, lengthscale=0.7)),
    ("cosine", lambda: cosine_kernel(frequency=1.3)),
    ("periodic", lambda: periodic_kernel(variance=1.3, lengthscale=1.5, period=1.1, order=8)),
    (
        "quasi_periodic_matern12",
        lambda: quasi_periodic_matern12_kernel(
            variance=0.9,
            lengthscale_periodic=1.2,
            period=0.8,
            lengthscale_matern=1.6,
            order=8,
        ),
    ),
]
ALL_KERNEL_IDS = [name for name, _ in ALL_KERNELS]


def _time_scale(kernel: StateSpaceKernel) -> float:
    """Return the reciprocal spectral radius of the drift, the kernel's own time scale."""
    return float(1.0 / jnp.max(jnp.abs(jnp.linalg.eigvals(kernel.feedback))))


def _decay_time_scale(kernel: StateSpaceKernel) -> float:
    """Return the slowest decay time ``1 / min(-Re eig F)`` of a dissipative drift."""
    return float(1.0 / jnp.min(-jnp.real(jnp.linalg.eigvals(kernel.feedback))))


_KERNEL_FACTORIES = dict(ALL_KERNELS)
# Kernels whose SDE dissipates and therefore adds process noise, and kernels whose SDE only
# rotates the state and adds none.
DISSIPATIVE_KERNEL_IDS = ["matern12", "matern32", "matern52", "matern72", "quasi_periodic_matern12"]
CONSERVATIVE_KERNEL_IDS = ["cosine", "periodic"]
# Step sizes as multiples of each kernel's own time scale, from far below it to far above it.
_STEP_RATIOS = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 50.0, 300.0, 1e3, 1e4)
# Relative residual of the exact identity Q(2h) = A(h) Q(h) A(h)^T + Q(h) in float32. Measured
# over these kernels for step ratios 1e-4 to 50: at most 2.4e-7 for the cancellation-free
# process noise, and 1.4e-5 to 4.6e-4 for P_inf - A P_inf A^T, whose subtraction cancels at
# small steps.
_DOUBLING_RESIDUAL_TOLERANCE = 5e-6


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


@pytest.mark.parametrize(("name", "factory"), ALL_KERNELS, ids=ALL_KERNEL_IDS)
def test_stationary_covariance_solves_lyapunov(
    name: str, factory: Callable[[], StateSpaceKernel]
) -> None:
    """``F P_inf + P_inf F^T + L Q_c L^T = 0`` for every kernel constructor.

    The residual scales inversely with ``lengthscale**(2 m + 1)`` for Matern-(m+1/2), so
    the test bounds the residual relative to ``|F| |P_inf|``.
    """
    kernel = factory()
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
# Exponentially scaled modified Bessel functions of the first kind.
# ---------------------------------------------------------------------------

# Arguments ``x = lengthscale**-2`` spanning lengthscales from 0.006 to 100, plus a dense
# window where the evaluation changes method.
_BESSEL_ARGUMENTS = np.concatenate([np.logspace(-4.0, 4.5, 48), [2990.0, 3000.0, 3010.0]])
# float32 cannot represent smaller values usefully; compare those entries absolutely.
_REPRESENTABLE_FLOOR = 1e-30


@pytest.mark.parametrize("max_order", [0, 6, 30, 100])
def test_scaled_modified_bessel_i_matches_reference(max_order: int) -> None:
    """``I_n(x) e^{-x}`` for ``n = 0..max_order`` agrees with ``scipy.special.ive``.

    The reference is evaluated at the float32-rounded argument so the comparison measures the
    evaluation rather than the argument's rounding.
    """
    orders = np.arange(max_order + 1)
    for argument in _BESSEL_ARGUMENTS:
        x32 = float(np.float32(argument))
        values = np.asarray(scaled_modified_bessel_i(max_order, jnp.asarray(x32)), dtype=np.float64)
        reference = ive(orders, x32)
        representable = reference > _REPRESENTABLE_FLOOR
        relative = np.abs(values - reference)[representable] / reference[representable]
        assert relative.max() < 1e-5, (max_order, x32, float(relative.max()))
        tiny_error = np.abs(values - reference)[~representable]
        assert np.all(tiny_error < _REPRESENTABLE_FLOOR), (max_order, x32)


@pytest.mark.parametrize("max_order", [6, 100])
def test_scaled_modified_bessel_i_is_nonnegative(max_order: int) -> None:
    """Every value is non-negative, so Bessel-weighted stationary covariances stay PSD."""
    values = jax.vmap(lambda x: scaled_modified_bessel_i(max_order, x))(
        jnp.asarray(_BESSEL_ARGUMENTS, dtype=jnp.float32)
    )
    assert bool(jnp.all(values >= 0.0))


def test_scaled_modified_bessel_i_gradient_matches_derivative_identity() -> None:
    r"""``d/dx [I_n(x) e^{-x}] = (I_{n-1}(x) + I_{n+1}(x)) e^{-x} / 2 - I_n(x) e^{-x}``."""
    max_order = 30
    orders = np.arange(max_order + 1)
    for argument in (1e-3, 0.04, 0.25, 1.0, 4.0, 25.0, 400.0, 2990.0, 3010.0, 2.0e4):
        x32 = float(np.float32(argument))
        jacobian = jax.jacfwd(lambda x: scaled_modified_bessel_i(max_order, x))(jnp.asarray(x32))
        jacobian = np.asarray(jacobian, dtype=np.float64)
        exact = 0.5 * (ive(np.abs(orders - 1), x32) + ive(orders + 1, x32)) - ive(orders, x32)
        scale = np.maximum(np.abs(exact), ive(orders, x32))
        representable = ive(orders, x32) > 1e-24
        relative = (np.abs(jacobian - exact) / scale)[representable]
        assert relative.max() < 1e-5, (x32, float(relative.max()))


def test_i0e_vector_is_deprecated_and_delegates() -> None:
    """``i0e_vector`` warns and returns the corrected values at the requested orders."""
    orders = jnp.asarray([0, 3, 5])
    with pytest.warns(DeprecationWarning, match="scaled_modified_bessel_i"):
        values = i0e_vector(orders, 2.0)
    expected = scaled_modified_bessel_i(5, jnp.asarray(2.0))[orders]
    assert bool(jnp.array_equal(values, expected))


@pytest.mark.parametrize("max_order", [-1, 3000])
def test_scaled_modified_bessel_i_rejects_orders_outside_the_stable_range(max_order: int) -> None:
    """Negative orders are undefined here, and forward recurrence needs every order below ``x``."""
    with pytest.raises(ValueError, match="max_order"):
        scaled_modified_bessel_i(max_order, jnp.asarray(1.0))


def test_scaled_modified_bessel_i_is_jit_compatible() -> None:
    """The evaluation compiles with the order as a static argument."""
    compiled = jax.jit(scaled_modified_bessel_i, static_argnums=0)
    values = compiled(12, jnp.asarray(1.5))
    assert values.shape == (13,)
    assert bool(jnp.all(jnp.isfinite(values)))


def _truncation_tail(lengthscale: float, order: int) -> float:
    """Return ``2 sum_{n > order} I_n(x) e^{-x}`` with ``x = lengthscale**-2``.

    The periodic kernel expands ``exp(x (cos(theta) - 1)) = e^{-x} (I_0(x) + 2 sum_n I_n(x)
    cos(n theta))``, so dropping the harmonics above ``order`` changes the covariance by at most
    this amount.
    """
    x = lengthscale**-2
    return float(2.0 * ive(np.arange(order + 1, order + 400), x).sum())


def _state_space_covariance(kernel: StateSpaceKernel, lags: jax.Array) -> jax.Array:
    """Return ``H A(tau) P_inf H^T`` at each lag."""
    return jax.vmap(
        lambda lag: (
            kernel.measurement
            @ kernel.state_transition(lag)
            @ kernel.stationary_cov
            @ kernel.measurement.T
        ).squeeze()
    )(lags)


@pytest.mark.parametrize("lengthscale", [0.5, 1.0, 2.0, 5.0])
def test_periodic_kernel_covariance_matches_closed_form(lengthscale: float) -> None:
    """``H A(tau) P_inf H^T = sigma^2 exp(-2 sin^2(pi tau / p) / l^2)`` up to dropped harmonics."""
    variance, period, order = 1.3, 1.1, 12
    kernel = periodic_kernel(variance=variance, lengthscale=lengthscale, period=period, order=order)
    lags = jnp.linspace(0.0, 2.0 * period, 23)
    closed_form = variance * jnp.exp(-2.0 * jnp.sin(jnp.pi * lags / period) ** 2 / lengthscale**2)
    bound = variance * _truncation_tail(lengthscale, order) + 1e-5 * variance
    worst = float(jnp.max(jnp.abs(_state_space_covariance(kernel, lags) - closed_form)))
    assert worst <= bound, (lengthscale, worst, bound)


@pytest.mark.parametrize("lengthscale_periodic", [0.8, 2.0])
def test_quasi_periodic_kernel_covariance_matches_closed_form(lengthscale_periodic: float) -> None:
    """``H A(tau) P_inf H^T`` equals the Matern-1/2 times periodic product up to dropped harmonics."""
    variance, period, lengthscale_matern, order = 0.9, 0.8, 1.6, 12
    kernel = quasi_periodic_matern12_kernel(
        variance=variance,
        lengthscale_periodic=lengthscale_periodic,
        period=period,
        lengthscale_matern=lengthscale_matern,
        order=order,
    )
    lags = jnp.linspace(0.0, 3.0, 25)
    periodic_part = jnp.exp(-2.0 * jnp.sin(jnp.pi * lags / period) ** 2 / lengthscale_periodic**2)
    closed_form = variance * jnp.exp(-lags / lengthscale_matern) * periodic_part
    bound = variance * _truncation_tail(lengthscale_periodic, order) + 1e-5 * variance
    worst = float(jnp.max(jnp.abs(_state_space_covariance(kernel, lags) - closed_form)))
    assert worst <= bound, (lengthscale_periodic, worst, bound)


# ---------------------------------------------------------------------------
# StateSpaceKernel dataclass — invariants.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Discretisation: transition increment and process noise.
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("float64")
@pytest.mark.parametrize(("name", "factory"), ALL_KERNELS, ids=ALL_KERNEL_IDS)
def test_transition_increment_matches_matrix_exponential(
    name: str, factory: Callable[[], StateSpaceKernel]
) -> None:
    """``transition_increment(dt)`` equals ``expm(F dt) - I`` at every step size.

    The reference is ``scipy.linalg.expm``. ``jax.scipy.linalg.expm`` in float64 is off by 2.9e-9
    for a rotation by 10 rad, where scipy and extended precision agree to 1e-14.
    """
    kernel = factory()
    identity = np.eye(kernel.state_dim)
    for ratio in _STEP_RATIOS:
        dt = ratio * _time_scale(kernel)
        expected = jnp.asarray(scipy_expm(np.asarray(kernel.feedback) * dt) - identity)
        dt = jnp.asarray(dt)
        error = float(jnp.max(jnp.abs(kernel.transition_increment(dt) - expected)))
        assert error <= 1e-9 * max(float(jnp.max(jnp.abs(expected))), 1e-6), (name, ratio, error)


@pytest.mark.usefixtures("float64")
@pytest.mark.parametrize(("name", "factory"), ALL_KERNELS, ids=ALL_KERNEL_IDS)
def test_discretize_matches_van_loan_in_float64(
    name: str, factory: Callable[[], StateSpaceKernel]
) -> None:
    """``discretize(dt)`` reproduces the Van Loan transition and process noise of ``(F, L, Q_c)``."""
    kernel = factory()
    noise_scale = float(jnp.max(jnp.abs(kernel.stationary_cov)))
    for ratio in (1e-3, 0.1, 1.0):
        dt = jnp.asarray(ratio * _time_scale(kernel))
        transition, process_noise = kernel.discretize(dt)
        reference_transition, reference_noise = discretize_lti_sde(
            drift_matrix=kernel.feedback,
            dispersion_matrix=kernel.noise_effect,
            dt=dt,
            diffusion=kernel.diffusion,
        )
        transition_error = float(jnp.max(jnp.abs(transition - reference_transition)))
        noise_error = float(jnp.max(jnp.abs(process_noise - reference_noise)))
        assert transition_error <= 1e-10, (name, ratio, transition_error)
        assert noise_error <= 1e-9 * noise_scale, (name, ratio, noise_error)


@pytest.mark.parametrize("name", DISSIPATIVE_KERNEL_IDS)
def test_process_noise_satisfies_the_doubling_identity(name: str) -> None:
    """In float32, ``Q(2h) = A(h) Q(h) A(h)^T + Q(h)`` holds from tiny to very large steps."""
    kernel = _KERNEL_FACTORIES[name]()
    for ratio in _STEP_RATIOS:
        step = jnp.asarray(ratio * _time_scale(kernel))
        transition, process_noise = kernel.discretize(step)
        _, doubled_noise = kernel.discretize(2.0 * step)
        composed = transition @ process_noise @ transition.T + process_noise
        residual = float(jnp.linalg.norm(doubled_noise - composed) / jnp.linalg.norm(doubled_noise))
        assert residual <= _DOUBLING_RESIDUAL_TOLERANCE, (name, ratio, residual)


@pytest.mark.parametrize("name", DISSIPATIVE_KERNEL_IDS)
def test_process_noise_is_positive_semidefinite(name: str) -> None:
    """Every eigenvalue of ``Q(dt)`` is non-negative up to float32 rounding (~1.2e-7 |Q|)."""
    kernel = _KERNEL_FACTORIES[name]()
    for ratio in _STEP_RATIOS:
        _, process_noise = kernel.discretize(jnp.asarray(ratio * _time_scale(kernel)))
        smallest = float(jnp.linalg.eigvalsh(process_noise).min())
        assert smallest >= -1e-6 * float(jnp.linalg.norm(process_noise)), (name, ratio, smallest)


@pytest.mark.parametrize("name", DISSIPATIVE_KERNEL_IDS)
def test_coarse_steps_reach_the_stationary_distribution(name: str) -> None:
    """Far beyond the slowest decay time, ``A(dt)`` vanishes and ``Q(dt)`` equals ``P_inf``.

    ``exp(-lambda dt)`` underflows there in float32, so both limits are exact up to rounding;
    measured relative errors are 2e-8 to 6e-8. The decay time, not the spectral radius, sets the
    scale: a quasi-periodic kernel rotates much faster than it decays.
    """
    kernel = _KERNEL_FACTORIES[name]()
    stationary_norm = float(jnp.linalg.norm(kernel.stationary_cov))
    for ratio in (300.0, 1e3, 1e4):
        dt = jnp.asarray(ratio * _decay_time_scale(kernel))
        transition, process_noise = kernel.discretize(dt)
        largest_transition = float(jnp.max(jnp.abs(transition)))
        noise_error = (
            float(jnp.linalg.norm(process_noise - kernel.stationary_cov)) / stationary_norm
        )
        assert largest_transition <= 1e-6, (name, ratio, largest_transition)
        assert noise_error <= 1e-6, (name, ratio, noise_error)


@pytest.mark.parametrize("name", CONSERVATIVE_KERNEL_IDS)
def test_conservative_kernels_add_no_process_noise(name: str) -> None:
    """A rotation SDE keeps its stationary covariance, so ``Q(dt)`` vanishes at every step."""
    kernel = _KERNEL_FACTORIES[name]()
    stationary_scale = float(jnp.max(jnp.abs(kernel.stationary_cov)))
    for ratio in _STEP_RATIOS:
        _, process_noise = kernel.discretize(jnp.asarray(ratio * _time_scale(kernel)))
        largest = float(jnp.max(jnp.abs(process_noise)))
        assert largest <= 1e-6 * stationary_scale, (name, ratio, largest)


@pytest.mark.parametrize(("name", "factory"), ALL_KERNELS, ids=ALL_KERNEL_IDS)
def test_discretize_transition_is_the_state_transition(
    name: str, factory: Callable[[], StateSpaceKernel]
) -> None:
    """``discretize(dt)[0]`` is ``state_transition(dt)``."""
    kernel = factory()
    dt = jnp.asarray(0.3 * _time_scale(kernel))
    assert bool(jnp.array_equal(kernel.discretize(dt)[0], kernel.state_transition(dt))), name


def test_state_space_kernel_built_from_a_state_transition_is_deprecated() -> None:
    """The ``state_transition=`` constructor argument warns and still describes the same SDE."""
    reference = matern32_kernel(variance=1.2, lengthscale=0.9)
    with pytest.warns(DeprecationWarning, match="transition_increment"):
        legacy = StateSpaceKernel(
            feedback=reference.feedback,
            noise_effect=reference.noise_effect,
            diffusion=reference.diffusion,
            measurement=reference.measurement,
            stationary_cov=reference.stationary_cov,
            state_transition=reference.state_transition,
        )
    dt = jnp.asarray(0.3)
    legacy_transition, legacy_noise = legacy.discretize(dt)
    reference_transition, reference_noise = reference.discretize(dt)
    # I + (A - I) rounds within two float32 ULPs of A.
    assert float(jnp.max(jnp.abs(legacy_transition - reference_transition))) <= 1e-6
    noise_scale = float(jnp.max(jnp.abs(reference.stationary_cov)))
    assert float(jnp.max(jnp.abs(legacy_noise - reference_noise))) <= 1e-5 * noise_scale


@pytest.mark.parametrize("supplied", ["both", "neither"])
def test_state_space_kernel_requires_exactly_one_transition_description(supplied: str) -> None:
    """Passing both or neither of ``transition_increment`` and ``state_transition`` raises."""
    reference = matern12_kernel(variance=1.0, lengthscale=1.0)
    transitions = (
        {
            "transition_increment": reference.transition_increment,
            "state_transition": reference.state_transition,
        }
        if supplied == "both"
        else {}
    )
    with pytest.raises(ValueError, match="transition_increment"):
        StateSpaceKernel(
            feedback=reference.feedback,
            noise_effect=reference.noise_effect,
            diffusion=reference.diffusion,
            measurement=reference.measurement,
            stationary_cov=reference.stationary_cov,
            **transitions,
        )


def test_state_space_kernel_is_immutable() -> None:
    """``StateSpaceKernel`` instances are frozen — attributes cannot be reassigned."""
    kernel = matern12_kernel(variance=1.0, lengthscale=1.0)
    with pytest.raises((AttributeError, TypeError)):
        kernel.feedback = jnp.zeros((1, 1))  # type: ignore[misc]

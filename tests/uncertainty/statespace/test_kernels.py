"""Tests for state-space kernels.

State-space kernels expose the continuous-time linear SDE
``(F, L, Q_c, H, P_inf)`` of a temporal GP prior and the closed-form
discrete-time state transition ``A(dt) = exp(F dt)`` of Särkkä & Solin
2019, eq. (6.24).

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
the periodic harmonic weights match ``scipy.special.ive`` in value and in their derivative with
respect to the lengthscale.

Discretisation. In float32 every component of the Matern process noise matches its
extended-precision reference (the Stillfjord-Tronarp Gramian, and the closed form for Matern-1/2),
and the quasi-periodic process noise matches the float64 identity ``P_inf - A P_inf A^T``. In
float64 the process noise is that identity, held to its relative Frobenius error. A kernel is a
pytree, so new hyperparameters reuse a compiled program, and the float32 and float64 methods are
chosen while tracing. A kernel built from its matrices alone discretises from ``F``; the deprecated
``state_transition=`` argument keeps the 0.2.5 discretisation.

References
----------
* Särkkä & Solin 2019 — *Applied Stochastic Differential Equations* §12.3.
* Hartikainen & Särkkä 2010 — *Kalman filtering and smoothing solutions to
  temporal Gaussian process regression models*, MLSP.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import expm
from scipy.linalg import expm as scipy_expm
from scipy.special import ive
from tensorflow_probability.substrates.jax.math import bessel_ive

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
from opifex.uncertainty.statespace.kernels import i0e_vector
from tests.uncertainty.statespace._accuracy import (
    relative_frobenius_error,
    scaled_error,
    transition_error,
)
from tests.uncertainty.statespace._process_noise_references import REFERENCES


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
# Relative residual of the exact identity Q(2h) = A(h) Q(h) A(h)^T + Q(h) in float32. Measured over
# these kernels and step ratios: at most 2.1e-7 for the Matern-1/2 closed form, 4.4e-7 to 6.2e-7 for
# the Stillfjord-Tronarp Gramian of Matern-3/2 to 7/2, and 1.2e-6 for the quasi-periodic kernel.
# P_inf - A P_inf A^T, whose subtraction cancels at small steps, measured 1.4e-5 to 4.6e-4.
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
# Periodic harmonic weights: exponentially scaled modified Bessel functions.
# ---------------------------------------------------------------------------

# float32 cannot represent smaller values usefully; compare those entries absolutely.
_REPRESENTABLE_FLOOR = 1e-30
# The weights come from TensorFlow Probability's ``bessel_ive``, whose float32 accuracy contract is
# rtol 7e-6 for orders and arguments in [1, 10], 1e-6 for [10, 100], and a gradient error below
# 2e-4.
_BESSEL_WEIGHT_TOLERANCE = 1e-5
_BESSEL_GRADIENT_TOLERANCE = 2e-4


def _harmonic_multiplicity(order: int) -> np.ndarray:
    """Return ``[1, 2, ..., 2]``, the cosine-series multiplicity of each harmonic."""
    return np.concatenate([[1.0], np.full(order, 2.0)])


@pytest.mark.parametrize("order", [6, 30])
def test_periodic_harmonic_weights_are_scaled_bessel_functions(order: int) -> None:
    """Stationary variances are ``sigma^2 [1, 2, ..., 2] I_n(x) e^{-x}`` at ``x = lengthscale**-2``."""
    variance = 1.3
    orders = np.arange(order + 1)
    for lengthscale in (0.1, 0.3, 1.0, 3.0, 30.0, 100.0):
        kernel = periodic_kernel(
            variance=variance, lengthscale=lengthscale, period=1.1, order=order
        )
        weights = np.asarray(jnp.diag(kernel.stationary_cov)[::2], dtype=np.float64)
        argument = float(np.float32(1.0 / lengthscale**2))
        reference = variance * _harmonic_multiplicity(order) * ive(orders, argument)
        representable = reference > _REPRESENTABLE_FLOOR
        relative = np.abs(weights - reference)[representable] / reference[representable]
        assert relative.max() <= _BESSEL_WEIGHT_TOLERANCE, (
            order,
            lengthscale,
            float(relative.max()),
        )
        assert np.all(np.abs(weights - reference)[~representable] < _REPRESENTABLE_FLOOR)


def test_periodic_harmonic_weights_are_differentiable_in_the_lengthscale() -> None:
    r"""``d/dl`` of the weights follows ``d/dx [I_n e^{-x}] = (I_{n-1} + I_{n+1}) e^{-x} / 2 - I_n e^{-x}``."""
    variance, order = 1.3, 12
    orders = np.arange(order + 1)
    multiplicity = _harmonic_multiplicity(order)
    for lengthscale in (0.3, 1.0, 3.0):

        def weights(ell: jax.Array) -> jax.Array:
            kernel = periodic_kernel(variance=variance, lengthscale=ell, period=1.1, order=order)
            return jnp.diag(kernel.stationary_cov)[::2]

        jacobian = np.asarray(jax.jacfwd(weights)(jnp.asarray(lengthscale)), dtype=np.float64)
        argument = float(np.float32(1.0 / lengthscale**2))
        value = variance * multiplicity * ive(orders, argument)
        slope = 0.5 * (ive(np.abs(orders - 1), argument) + ive(orders + 1, argument)) - ive(
            orders, argument
        )
        expected = variance * multiplicity * slope * (-2.0 / lengthscale**3)
        representable = value > 1e-24
        scale = np.maximum(np.abs(expected), value)
        relative = (np.abs(jacobian - expected) / scale)[representable]
        assert relative.max() <= _BESSEL_GRADIENT_TOLERANCE, (lengthscale, float(relative.max()))


def test_i0e_vector_is_deprecated_and_delegates() -> None:
    """``i0e_vector`` warns and returns TensorFlow Probability's ``bessel_ive`` values."""
    orders = jnp.asarray([0, 3, 5])
    with pytest.warns(DeprecationWarning, match="bessel_ive"):
        values = i0e_vector(orders, 2.0)
    expected = bessel_ive(jnp.asarray([0.0, 3.0, 5.0]), jnp.asarray(2.0))
    assert bool(jnp.array_equal(values, expected))


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
# Discretisation: transition and process noise.
# ---------------------------------------------------------------------------

# Matern kernels at unit variance and lengthscale, keyed by their extended-precision reference.
_REFERENCE_KERNELS: dict[str, Callable[..., StateSpaceKernel]] = {
    "matern12": matern12_kernel,
    "matern32": matern32_kernel,
    "matern52": matern52_kernel,
    "matern72": matern72_kernel,
}
# Float32 process noise: worst scaled error of the Stillfjord-Tronarp Gramian over the references,
# measured at 1.4e-5 (Matern-7/2 at lambda dt = 1e4); the Matern-1/2 closed form measured 1.2e-7.
_FLOAT32_NOISE_TOLERANCE = 5e-5
# Float64 process noise is P_inf - A P_inf A^T, which cancels per component at small steps and is
# therefore held to its relative Frobenius error, measured at most 1.7e-13 (Matern-3/2).
_FLOAT64_NOISE_TOLERANCE = 1e-12
# Closed-form transitions; max |A - R| / max(1, max |R|).
_TRANSITION_TOLERANCE = {"float32": 1e-4, "float64": 1e-12}
# A vectorised closed form may round differently from the same expression at batch size one:
# measured at most one float32 ULP of the matrix scale (1.2e-7 for Matern-5/2), eager and jitted.
# The process noise matched bitwise, so it is compared exactly.
_BATCH_TRANSITION_TOLERANCE = 4.0 * float(np.finfo(np.float32).eps)
# Far beyond the decay time, float32 ``Q`` reaches ``P_inf`` exactly for the Matern-1/2 closed form
# and up to the round-off of the Stillfjord-Tronarp doublings otherwise: measured relative Frobenius
# errors of 1.2e-7 to 2.4e-6 for Matern-3/2 to 7/2 and 2.3e-6 to 5.1e-6 for the quasi-periodic
# kernel, at 50 to 1e4 decay times. Float64 reaches ``P_inf`` exactly. The earlier closed-form
# increment measured 2e-8 to 6e-8 but lost the small components of ``Q`` at short steps, which is
# why float32 moved to the Gramian.
_COARSE_NOISE_TOLERANCE = 1e-5


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("name", list(_REFERENCE_KERNELS))
def test_matern_discretisation_matches_extended_precision_references(
    request: pytest.FixtureRequest, name: str, precision: str
) -> None:
    """Transition and process noise match the references from 1e-4 to 1e4 lengthscales."""
    if precision == "float64":
        request.getfixturevalue("float64")
    kernel = _REFERENCE_KERNELS[name](variance=1.0, lengthscale=1.0)
    for step in REFERENCES[name].steps:
        transition, process_noise = kernel.discretize(jnp.asarray(step.dt))
        assert str(process_noise.dtype) == precision
        error = transition_error(transition, step.transition)
        assert error <= _TRANSITION_TOLERANCE[precision], (name, step.scale, error)
        if precision == "float32":
            noise_error = scaled_error(process_noise, step.process_noise)
            assert noise_error <= _FLOAT32_NOISE_TOLERANCE, (name, step.scale, noise_error)
        else:
            noise_error = relative_frobenius_error(process_noise, step.process_noise)
            assert noise_error <= _FLOAT64_NOISE_TOLERANCE, (name, step.scale, noise_error)


def test_quasi_periodic_float32_process_noise_matches_the_float64_identity(
    request: pytest.FixtureRequest,
) -> None:
    """Every component matches ``P_inf - A P_inf A^T`` evaluated in float64.

    A Matern-1/2 envelope of rotations does not cancel in that identity: the rotation leaves each
    harmonic block of ``P_inf`` unchanged, so the float64 reference keeps its relative digits.
    """
    parameters = {
        "variance": 0.9,
        "lengthscale_periodic": 1.2,
        "period": 0.8,
        "lengthscale_matern": 1.6,
        "order": 8,
    }
    kernel = quasi_periodic_matern12_kernel(**parameters)
    time_scale = _time_scale(kernel)
    ratios = (1e-4, 1e-2, 1.0, 1e2, 1e4)
    estimates = [kernel.discretize(jnp.asarray(ratio * time_scale))[1] for ratio in ratios]
    request.getfixturevalue("float64")
    reference_kernel = quasi_periodic_matern12_kernel(**parameters)
    feedback = np.asarray(reference_kernel.feedback)
    stationary = np.asarray(reference_kernel.stationary_cov)
    for ratio, estimate in zip(ratios, estimates, strict=True):
        transition = scipy_expm(feedback * ratio * time_scale)
        reference = stationary - transition @ stationary @ transition.T
        error = scaled_error(estimate, reference)
        assert error <= _FLOAT32_NOISE_TOLERANCE, (ratio, error)


@pytest.mark.usefixtures("float64")
@pytest.mark.parametrize(("name", "factory"), ALL_KERNELS, ids=ALL_KERNEL_IDS)
def test_state_transition_matches_matrix_exponential(
    name: str, factory: Callable[[], StateSpaceKernel]
) -> None:
    """``state_transition(dt)`` equals ``expm(F dt)`` at every step size.

    The reference is ``scipy.linalg.expm``. ``jax.scipy.linalg.expm`` in float64 is off by 2.9e-9
    for a rotation by 10 rad, where scipy and extended precision agree to 1e-14.
    """
    kernel = factory()
    for ratio in _STEP_RATIOS:
        dt = ratio * _time_scale(kernel)
        expected = scipy_expm(np.asarray(kernel.feedback) * dt)
        error = transition_error(kernel.state_transition(jnp.asarray(dt)), expected)
        assert error <= 1e-9, (name, ratio, error)


@pytest.mark.usefixtures("float64")
@pytest.mark.parametrize(("name", "factory"), ALL_KERNELS, ids=ALL_KERNEL_IDS)
def test_discretize_matches_discretize_lti_sde_in_float64(
    name: str, factory: Callable[[], StateSpaceKernel]
) -> None:
    """``discretize(dt)`` reproduces ``discretize_lti_sde`` of the kernel's ``(F, L, Q_c)``."""
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
        transition_difference = float(jnp.max(jnp.abs(transition - reference_transition)))
        noise_difference = float(jnp.max(jnp.abs(process_noise - reference_noise)))
        assert transition_difference <= 1e-10, (name, ratio, transition_difference)
        assert noise_difference <= 1e-9 * noise_scale, (name, ratio, noise_difference)


@pytest.mark.parametrize(("name", "factory"), ALL_KERNELS, ids=ALL_KERNEL_IDS)
def test_discretize_steps_matches_each_step(
    name: str, factory: Callable[[], StateSpaceKernel]
) -> None:
    """``discretize_steps`` over a sequence equals ``discretize`` at each of its steps."""
    kernel = factory()
    steps = jnp.asarray([0.0, 1e-4, 1e-2, 1.0, 1e2, 1e4]) * _time_scale(kernel)
    transitions, process_noises = kernel.discretize_steps(steps)
    for index in range(steps.shape[0]):
        transition, process_noise = kernel.discretize(steps[index])
        scale = max(1.0, float(jnp.max(jnp.abs(transition))))
        difference = float(jnp.max(jnp.abs(transitions[index] - transition)))
        assert difference <= _BATCH_TRANSITION_TOLERANCE * scale, (name, index, difference)
        np.testing.assert_array_equal(np.asarray(process_noises[index]), np.asarray(process_noise))


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
    """Far beyond the slowest decay time, ``A(dt)`` vanishes and ``Q(dt)`` approaches ``P_inf``.

    ``exp(-lambda dt)`` underflows there in float32, so ``A`` vanishes; ``Q`` reaches ``P_inf`` up to
    the round-off recorded in ``_COARSE_NOISE_TOLERANCE``. The decay time, not the spectral radius,
    sets the scale: a quasi-periodic kernel rotates much faster than it decays.
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
        assert noise_error <= _COARSE_NOISE_TOLERANCE, (name, ratio, noise_error)


@pytest.mark.parametrize("name", CONSERVATIVE_KERNEL_IDS)
def test_conservative_kernels_add_no_process_noise(name: str) -> None:
    """A rotation SDE keeps its stationary covariance, so ``Q(dt)`` vanishes at every step."""
    kernel = _KERNEL_FACTORIES[name]()
    for ratio in _STEP_RATIOS:
        _, process_noise = kernel.discretize(jnp.asarray(ratio * _time_scale(kernel)))
        assert bool(jnp.all(process_noise == 0.0)), (name, ratio)


@pytest.mark.parametrize(("name", "factory"), ALL_KERNELS, ids=ALL_KERNEL_IDS)
def test_discretize_transition_is_the_state_transition(
    name: str, factory: Callable[[], StateSpaceKernel]
) -> None:
    """``discretize(dt)[0]`` is ``state_transition(dt)``."""
    kernel = factory()
    dt = jnp.asarray(0.3 * _time_scale(kernel))
    assert bool(jnp.array_equal(kernel.discretize(dt)[0], kernel.state_transition(dt))), name


# ---------------------------------------------------------------------------
# StateSpaceKernel — pytree, construction and deprecation.
# ---------------------------------------------------------------------------


def test_kernel_is_a_pytree_of_arrays() -> None:
    """Every leaf of a kernel is an array, and unflattening rebuilds the same discretisation."""
    kernel = matern52_kernel(variance=0.8, lengthscale=1.1)
    leaves, treedef = jax.tree_util.tree_flatten(kernel)
    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    dt = jnp.asarray(0.4)
    for original, copy in zip(kernel.discretize(dt), rebuilt.discretize(dt), strict=True):
        assert bool(jnp.array_equal(original, copy))


def test_jitted_discretisation_compiles_once_across_hyperparameters() -> None:
    """A kernel passed to a jitted function is data: new hyperparameters reuse the program."""
    trace_count = 0

    def discretize(kernel: StateSpaceKernel, dt: jax.Array) -> tuple[jax.Array, jax.Array]:
        nonlocal trace_count
        trace_count += 1
        return kernel.discretize(dt)

    jitted = jax.jit(discretize)
    for lengthscale in (0.7, 1.9, 4.0):
        jitted(matern52_kernel(variance=1.3, lengthscale=lengthscale), jnp.asarray(0.3))
    assert trace_count == 1


def test_discretisation_compiles_once_per_precision(request: pytest.FixtureRequest) -> None:
    """The float32 and float64 methods are chosen while tracing, one compilation per dtype."""
    traced_dtypes: list[str] = []

    def discretize(kernel: StateSpaceKernel, dt: jax.Array) -> tuple[jax.Array, jax.Array]:
        traced_dtypes.append(str(kernel.feedback.dtype))
        return kernel.discretize(dt)

    jitted = jax.jit(discretize)
    for value in (0.1, 5.0):
        jitted(matern72_kernel(variance=1.0, lengthscale=1.0), jnp.asarray(value))
    request.getfixturevalue("float64")
    for value in (0.1, 5.0):
        jitted(matern72_kernel(variance=1.0, lengthscale=1.0), jnp.asarray(value))
    assert traced_dtypes == ["float32", "float64"]


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_kernel_built_from_its_matrices_discretises_from_the_feedback_matrix(
    request: pytest.FixtureRequest, precision: str
) -> None:
    """Given only ``(F, L, Q_c, H, P_inf)``, a kernel uses ``exp(F dt)`` and, in float32, the Gramian."""
    if precision == "float64":
        request.getfixturevalue("float64")
    template = matern52_kernel(variance=1.0, lengthscale=1.0)
    kernel = StateSpaceKernel(
        feedback=template.feedback,
        noise_effect=template.noise_effect,
        diffusion=template.diffusion,
        measurement=template.measurement,
        stationary_cov=template.stationary_cov,
    )
    for step in REFERENCES["matern52"].steps:
        dt = jnp.asarray(step.dt)
        transition, process_noise = kernel.discretize(dt)
        assert bool(jnp.array_equal(transition, kernel.state_transition(dt)))
        error = transition_error(transition, step.transition)
        assert error <= _TRANSITION_TOLERANCE[precision], (step.scale, error)
        if precision == "float32":
            noise_error = scaled_error(process_noise, step.process_noise)
            assert noise_error <= _FLOAT32_NOISE_TOLERANCE, (step.scale, noise_error)
        else:
            noise_error = relative_frobenius_error(process_noise, step.process_noise)
            assert noise_error <= _FLOAT64_NOISE_TOLERANCE, (step.scale, noise_error)


def test_state_space_kernel_built_from_a_state_transition_keeps_the_released_discretisation() -> (
    None
):
    """``state_transition=`` warns and discretises as 0.2.5 did.

    The transition is the supplied callable and the process noise is ``P_inf - A P_inf A^T``;
    rebuilding the kernel from its pytree does not warn again.
    """
    reference = matern32_kernel(variance=1.2, lengthscale=0.9)
    with pytest.warns(DeprecationWarning, match="state_transition"):
        legacy = StateSpaceKernel(
            feedback=reference.feedback,
            noise_effect=reference.noise_effect,
            diffusion=reference.diffusion,
            measurement=reference.measurement,
            stationary_cov=reference.stationary_cov,
            state_transition=reference.state_transition,
        )
    dt = jnp.asarray(0.3)
    transition, process_noise = legacy.discretize(dt)
    expected_transition = reference.state_transition(dt)
    assert bool(jnp.array_equal(transition, expected_transition))
    stationary = reference.stationary_cov
    expected_noise = stationary - expected_transition @ stationary @ expected_transition.T
    noise_scale = float(jnp.max(jnp.abs(stationary)))
    assert float(jnp.max(jnp.abs(process_noise - expected_noise))) <= 1e-6 * noise_scale
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        leaves, treedef = jax.tree_util.tree_flatten(legacy)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert bool(jnp.array_equal(rebuilt.state_transition(dt), expected_transition))


def test_state_space_kernel_is_immutable() -> None:
    """``StateSpaceKernel`` instances are frozen — attributes cannot be reassigned."""
    kernel = matern12_kernel(variance=1.0, lengthscale=1.0)
    with pytest.raises((AttributeError, TypeError)):
        kernel.feedback = jnp.zeros((1, 1))  # type: ignore[misc]

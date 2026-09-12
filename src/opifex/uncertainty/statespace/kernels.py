"""State-space kernel constructors for temporal Gaussian processes.

Each constructor returns a :class:`StateSpaceKernel` carrying the
continuous-time SDE parameters ``(F, L, Q_c, H, P_inf)`` and a closed-form
discrete-time state-transition matrix ``A(dt) = exp(F dt)``. The closed
forms avoid the runtime cost of a per-step ``expm`` and remain
differentiable w.r.t. ``dt`` and the kernel hyperparameters.

Canonical reference (line-by-line port):
* ``../bayesnewton/bayesnewton/kernels.py`` — ``Matern12`` (line 141),
  ``Matern32`` (line 200), ``Matern52`` (line 253), ``Matern72`` (line
  321), ``Cosine`` (line 770), ``Periodic`` (line 802),
  ``QuasiPeriodicMatern12`` (line 882).

References:
----------
* Särkkä & Solin 2019 — *Applied Stochastic Differential Equations* §12.3
  and Table 12.2.
* Hartikainen & Särkkä 2010 — *Kalman filtering and smoothing solutions to
  temporal Gaussian process regression models*, MLSP.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable  # noqa: TC003 — kept eager for consistency
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import i0e, i1e


# Arguments at or above this value take forward recurrence from ``i0e`` and ``i1e``, which is
# stable while every order stays below the argument. Below it, backward recurrence is used.
_BESSEL_FORWARD_RECURRENCE_THRESHOLD = 3000.0
# Orders added above the highest requested order before backward recurrence starts from a zero
# ratio. Measured in float32 against ``scipy.special.ive`` for arguments below the threshold:
# relative error at most 6.4e-7 for 6 orders, 1.3e-6 for 30 and 2.6e-6 for 100.
_BESSEL_BACKWARD_EXTRA_DEPTH = 512


@dataclass(frozen=True, slots=True, kw_only=True)
class StateSpaceKernel:
    """Continuous-time SDE representation of a stationary temporal GP kernel.

    Attributes:
        feedback: Drift matrix ``F`` (shape ``(n, n)``).
        noise_effect: Dispersion matrix ``L`` (shape ``(n, k)``).
        diffusion: Wiener diffusion ``Q_c`` (shape ``(k, k)``).
        measurement: Observation matrix ``H`` (shape ``(1, n)``).
        stationary_cov: Stationary covariance ``P_inf`` (shape ``(n, n)``),
            satisfying ``F P_inf + P_inf F^T + L Q_c L^T = 0``.
        state_transition: Closed-form ``A(dt) = exp(F dt)``.
    """

    feedback: jax.Array
    noise_effect: jax.Array
    diffusion: jax.Array
    measurement: jax.Array
    stationary_cov: jax.Array
    state_transition: Callable[[jax.Array], jax.Array]

    @property
    def state_dim(self) -> int:
        """Dimension of the state-space representation."""
        return int(self.feedback.shape[0])


def matern12_kernel(*, variance: float, lengthscale: float) -> StateSpaceKernel:
    r"""Matern-1/2 (exponential) kernel in SDE form.

    State dimension 1. ``F = [[-1/ell]]``, ``Q_c = 2 sigma^2 / ell``,
    ``A(dt) = exp(-dt/ell)``. Ports bayesnewton ``Matern12`` (line 141).
    """
    feedback = jnp.asarray([[-1.0 / lengthscale]])
    noise_effect = jnp.asarray([[1.0]])
    diffusion = jnp.asarray([[2.0 * variance / lengthscale]])
    measurement = jnp.asarray([[1.0]])
    stationary_cov = jnp.asarray([[variance]])

    def state_transition(dt: jax.Array) -> jax.Array:
        """Return the discrete-time state-transition matrix over a step ``dt``."""
        return jnp.broadcast_to(jnp.exp(-dt / lengthscale)[..., None, None], (1, 1))

    return StateSpaceKernel(
        feedback=feedback,
        noise_effect=noise_effect,
        diffusion=diffusion,
        measurement=measurement,
        stationary_cov=stationary_cov,
        state_transition=state_transition,
    )


def matern32_kernel(*, variance: float, lengthscale: float) -> StateSpaceKernel:
    r"""Matern-3/2 kernel in SDE form. Ports bayesnewton ``Matern32`` (line 200)."""
    lam = jnp.sqrt(3.0) / lengthscale
    feedback = jnp.asarray([[0.0, 1.0], [-(lam**2), -2.0 * lam]])
    noise_effect = jnp.asarray([[0.0], [1.0]])
    diffusion = jnp.asarray([[12.0 * jnp.sqrt(3.0) / lengthscale**3 * variance]])
    measurement = jnp.asarray([[1.0, 0.0]])
    stationary_cov = jnp.asarray([[variance, 0.0], [0.0, 3.0 * variance / lengthscale**2]])

    def state_transition(dt: jax.Array) -> jax.Array:
        """Return the discrete-time state-transition matrix over a step ``dt``."""
        inner = dt * jnp.asarray([[lam, 1.0], [-(lam**2), -lam]]) + jnp.eye(2)
        return jnp.exp(-dt * lam) * inner

    return StateSpaceKernel(
        feedback=feedback,
        noise_effect=noise_effect,
        diffusion=diffusion,
        measurement=measurement,
        stationary_cov=stationary_cov,
        state_transition=state_transition,
    )


def matern52_kernel(*, variance: float, lengthscale: float) -> StateSpaceKernel:
    r"""Matern-5/2 kernel in SDE form. Ports bayesnewton ``Matern52`` (line 253)."""
    lam = jnp.sqrt(5.0) / lengthscale
    feedback = jnp.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-(lam**3), -3.0 * lam**2, -3.0 * lam]]
    )
    noise_effect = jnp.asarray([[0.0], [0.0], [1.0]])
    diffusion = jnp.asarray([[variance * 400.0 * jnp.sqrt(5.0) / 3.0 / lengthscale**5]])
    measurement = jnp.asarray([[1.0, 0.0, 0.0]])
    kappa = 5.0 / 3.0 * variance / lengthscale**2
    stationary_cov = jnp.asarray(
        [
            [variance, 0.0, -kappa],
            [0.0, kappa, 0.0],
            [-kappa, 0.0, 25.0 * variance / lengthscale**4],
        ]
    )

    def state_transition(dt: jax.Array) -> jax.Array:
        """Return the discrete-time state-transition matrix over a step ``dt``."""
        dtlam = dt * lam
        matrix = jnp.asarray(
            [
                [lam * (0.5 * dtlam + 1.0), dtlam + 1.0, 0.5 * dt],
                [-0.5 * dtlam * lam**2, lam * (1.0 - dtlam), 1.0 - 0.5 * dtlam],
                [
                    lam**3 * (0.5 * dtlam - 1.0),
                    lam**2 * (dtlam - 3),
                    lam * (0.5 * dtlam - 2.0),
                ],
            ]
        )
        return jnp.exp(-dtlam) * (dt * matrix + jnp.eye(3))

    return StateSpaceKernel(
        feedback=feedback,
        noise_effect=noise_effect,
        diffusion=diffusion,
        measurement=measurement,
        stationary_cov=stationary_cov,
        state_transition=state_transition,
    )


def matern72_kernel(*, variance: float, lengthscale: float) -> StateSpaceKernel:
    r"""Matern-7/2 kernel in SDE form. Ports bayesnewton ``Matern72`` (line 321)."""
    lam = jnp.sqrt(7.0) / lengthscale
    feedback = jnp.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-(lam**4), -4.0 * lam**3, -6.0 * lam**2, -4.0 * lam],
        ]
    )
    noise_effect = jnp.asarray([[0.0], [0.0], [0.0], [1.0]])
    diffusion = jnp.asarray([[variance * 10976.0 * jnp.sqrt(7.0) / 5.0 / lengthscale**7]])
    measurement = jnp.asarray([[1.0, 0.0, 0.0, 0.0]])
    kappa = 7.0 / 5.0 * variance / lengthscale**2
    kappa2 = 9.8 * variance / lengthscale**4
    stationary_cov = jnp.asarray(
        [
            [variance, 0.0, -kappa, 0.0],
            [0.0, kappa, 0.0, -kappa2],
            [-kappa, 0.0, kappa2, 0.0],
            [0.0, -kappa2, 0.0, 343.0 * variance / lengthscale**6],
        ]
    )

    def state_transition(dt: jax.Array) -> jax.Array:
        """Return the discrete-time state-transition matrix over a step ``dt``."""
        lam2 = lam * lam
        lam3 = lam2 * lam
        dtlam = dt * lam
        dtlam2 = dtlam**2
        matrix = jnp.asarray(
            [
                [
                    lam * (1.0 + 0.5 * dtlam + dtlam2 / 6.0),
                    1.0 + dtlam + 0.5 * dtlam2,
                    0.5 * dt * (1.0 + dtlam),
                    dt**2 / 6,
                ],
                [
                    -dtlam2 * lam**2 / 6.0,
                    lam * (1.0 + 0.5 * dtlam - 0.5 * dtlam2),
                    1.0 + dtlam - 0.5 * dtlam2,
                    dt * (0.5 - dtlam / 6.0),
                ],
                [
                    lam3 * dtlam * (dtlam / 6.0 - 0.5),
                    dtlam * lam2 * (0.5 * dtlam - 2.0),
                    lam * (1.0 - 2.5 * dtlam + 0.5 * dtlam2),
                    1.0 - dtlam + dtlam2 / 6.0,
                ],
                [
                    lam2**2 * (dtlam - 1.0 - dtlam2 / 6.0),
                    lam3 * (3.5 * dtlam - 4.0 - 0.5 * dtlam2),
                    lam2 * (4.0 * dtlam - 6.0 - 0.5 * dtlam2),
                    lam * (1.5 * dtlam - 3.0 - dtlam2 / 6.0),
                ],
            ]
        )
        return jnp.exp(-dtlam) * (dt * matrix + jnp.eye(4))

    return StateSpaceKernel(
        feedback=feedback,
        noise_effect=noise_effect,
        diffusion=diffusion,
        measurement=measurement,
        stationary_cov=stationary_cov,
        state_transition=state_transition,
    )


def cosine_kernel(*, frequency: float) -> StateSpaceKernel:
    r"""Cosine kernel as SDE. Ports bayesnewton ``Cosine`` (line 770).

    State dim 2; transition is the 2-D rotation by angle ``frequency dt``.
    """
    feedback = jnp.asarray([[0.0, -frequency], [frequency, 0.0]])
    noise_effect = jnp.zeros((2, 0))
    diffusion = jnp.zeros((0, 0))
    measurement = jnp.asarray([[1.0, 0.0]])
    stationary_cov = jnp.eye(2)

    def state_transition(dt: jax.Array) -> jax.Array:
        """Return the discrete-time state-transition matrix over a step ``dt``."""
        angle = frequency * dt
        cos = jnp.cos(angle)
        sin = jnp.sin(angle)
        return jnp.asarray([[cos, -sin], [sin, cos]])

    return StateSpaceKernel(
        feedback=feedback,
        noise_effect=noise_effect,
        diffusion=diffusion,
        measurement=measurement,
        stationary_cov=stationary_cov,
        state_transition=state_transition,
    )


def periodic_kernel(
    *, variance: float, lengthscale: float, period: float, order: int = 6
) -> StateSpaceKernel:
    r"""Periodic kernel via Bessel-weighted sum of harmonic rotations.

    Ports bayesnewton ``Periodic`` (line 802). State dim ``2(order + 1)``;
    transition is block-diagonal of harmonic rotation matrices. With
    ``x = lengthscale**-2`` the harmonic variances are ``sigma^2 I_0(x) e^{-x}``
    and ``2 sigma^2 I_n(x) e^{-x}``, the cosine-series coefficients of
    ``sigma^2 exp(x (cos(omega tau) - 1))`` (DLMF 10.35.1). Truncating at
    ``order`` changes the covariance by at most ``2 sigma^2 sum_{n > order}
    I_n(x) e^{-x}``.
    """
    omega = 2.0 * jnp.pi / period
    harmonic_indices = jnp.arange(order + 1)
    angular_frequencies = harmonic_indices * omega

    feedback = jnp.kron(
        jnp.diag(harmonic_indices.astype(jnp.float32)),
        jnp.asarray([[0.0, -omega], [omega, 0.0]]),
    )
    state_size = 2 * (order + 1)
    noise_effect = jnp.eye(state_size)
    diffusion = jnp.zeros((state_size, state_size))
    measurement = jnp.kron(jnp.ones((1, order + 1)), jnp.asarray([[1.0, 0.0]]))

    bessel_factors = scaled_modified_bessel_i(order, 1.0 / lengthscale**2)
    q2 = jnp.concatenate([jnp.asarray([1.0]), 2.0 * jnp.ones(order)]) * variance * bessel_factors
    stationary_cov = jnp.kron(jnp.diag(q2), jnp.eye(2))

    def state_transition(dt: jax.Array) -> jax.Array:
        """Return the discrete-time state-transition matrix over a step ``dt``."""

        def single_rotation(angle: jax.Array) -> jax.Array:
            """Return the 2x2 rotation block for one harmonic at the given angle."""
            cos = jnp.cos(angle)
            sin = jnp.sin(angle)
            return jnp.asarray([[cos, -sin], [sin, cos]])

        blocks = jax.vmap(single_rotation)(angular_frequencies * dt)
        return jax.scipy.linalg.block_diag(*blocks)

    return StateSpaceKernel(
        feedback=feedback,
        noise_effect=noise_effect,
        diffusion=diffusion,
        measurement=measurement,
        stationary_cov=stationary_cov,
        state_transition=state_transition,
    )


def scaled_modified_bessel_i(max_order: int, argument: jax.Array | float) -> jax.Array:
    r"""Return the exponentially scaled modified Bessel functions ``I_n(x) e^{-x}``.

    Orders run from ``0`` to ``max_order``. Both branches rest on
    ``I_{n-1}(x) - I_{n+1}(x) = (2 n / x) I_n(x)`` (DLMF 10.29.1). Below
    ``x = 3000`` the ratios ``r_n = I_n(x) / I_{n-1}(x)`` come from the backward
    recurrence ``r_n = 1 / (2 n / x + r_{n+1})``, started a fixed number of orders
    above ``max_order`` with a zero ratio, and the values are ``i0e(x)`` times the
    running product of ratios. Forward recurrence amplifies rounding by roughly
    ``(2 n / x)^n`` once the order exceeds ``x``, so it runs only at and above
    ``x = 3000``, where every accepted order is below ``x``. Each branch is evaluated on
    an argument valid for it, which keeps values and gradients finite across the switch.

    Args:
        max_order: Highest order. Static under ``jax.jit``.
        argument: Positive scalar ``x``.

    Returns:
        Array of shape ``(max_order + 1,)``.

    Raises:
        ValueError: If ``max_order`` is negative, or not below the forward-recurrence
            threshold.
    """
    if not 0 <= max_order < _BESSEL_FORWARD_RECURRENCE_THRESHOLD:
        upper = int(_BESSEL_FORWARD_RECURRENCE_THRESHOLD)
        raise ValueError(f"max_order must lie in [0, {upper}), got {max_order}")
    x = jnp.asarray(argument)
    use_forward = x >= _BESSEL_FORWARD_RECURRENCE_THRESHOLD
    backward_values = _bessel_backward_recurrence(max_order, jnp.where(use_forward, 1.0, x))
    forward_values = _bessel_forward_recurrence(
        max_order, jnp.where(use_forward, x, _BESSEL_FORWARD_RECURRENCE_THRESHOLD)
    )
    return jnp.where(use_forward, forward_values, backward_values)


def _bessel_backward_recurrence(max_order: int, x: jax.Array) -> jax.Array:
    """Return ``I_n(x) e^{-x}`` for ``n = 0..max_order`` by backward ratio recurrence."""
    depth = max_order + _BESSEL_BACKWARD_EXTRA_DEPTH

    def step(next_ratio: jax.Array, order: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return ``r_n = I_n / I_{n-1}`` from ``r_{n+1}``, as carry and output."""
        ratio = 1.0 / (2.0 * order / x + next_ratio)
        return ratio, ratio

    orders = jnp.arange(depth, 0, -1, dtype=x.dtype)
    _, descending_ratios = jax.lax.scan(step, jnp.zeros_like(x), orders)
    ratios = descending_ratios[::-1][:max_order]
    running_product = jnp.concatenate([jnp.ones((1,), dtype=x.dtype), jnp.cumprod(ratios)])
    return i0e(x) * running_product


def _bessel_forward_recurrence(max_order: int, x: jax.Array) -> jax.Array:
    """Return ``I_n(x) e^{-x}`` for ``n = 0..max_order`` by forward recurrence."""

    def step(
        carry: tuple[jax.Array, jax.Array], order: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        """Return ``I_{n+1} e^{-x}`` from the values at orders ``n - 1`` and ``n``."""
        previous, current = carry
        following = previous - 2.0 * order / x * current
        return (current, following), following

    first, second = i0e(x), i1e(x)
    _, higher = jax.lax.scan(step, (first, second), jnp.arange(1, max_order, dtype=x.dtype))
    return jnp.concatenate([jnp.stack([first, second]), higher])[: max_order + 1]


def i0e_vector(orders: jax.Array, argument: float) -> jax.Array:
    """Return ``I_n(x) e^{-x}`` at the requested orders.

    Deprecated: use :func:`scaled_modified_bessel_i`, which returns every order up to
    ``max_order``. This name now delegates to that evaluation.

    Args:
        orders: Non-negative integer orders.
        argument: Positive scalar ``x``.

    Returns:
        The values at ``orders``.
    """
    warnings.warn(
        "i0e_vector is deprecated; use scaled_modified_bessel_i(max_order, argument).",
        DeprecationWarning,
        stacklevel=2,
    )
    return scaled_modified_bessel_i(int(orders.max()), argument)[orders]


def quasi_periodic_matern12_kernel(
    *,
    variance: float,
    lengthscale_periodic: float,
    period: float,
    lengthscale_matern: float,
    order: int = 6,
) -> StateSpaceKernel:
    r"""Quasi-periodic Matern-1/2 kernel: product of Periodic and Matern-1/2.

    Ports bayesnewton ``QuasiPeriodicMatern12`` (line 882). Constructed as
    the Kronecker product of the Matern-1/2 SDE with the Periodic SDE; the
    diffusion is the Matern-1/2 diffusion scaled by the periodic stationary
    covariance, which balances ``P_inf = P_matern (x) P_periodic``.
    """
    matern = matern12_kernel(variance=variance, lengthscale=lengthscale_matern)
    periodic = periodic_kernel(
        variance=1.0, lengthscale=lengthscale_periodic, period=period, order=order
    )

    feedback = jnp.kron(matern.feedback, jnp.eye(periodic.state_dim)) + jnp.kron(
        jnp.eye(matern.state_dim), periodic.feedback
    )
    state_size = matern.state_dim * periodic.state_dim
    noise_effect = jnp.eye(state_size)
    # The periodic SDE conserves its stationary covariance (F_p P_p + P_p F_p^T = 0), so the
    # Lyapunov residual of the product is the Matern term alone: (L_m Q_m L_m^T) (x) P_p.
    diffusion = jnp.kron(matern.diffusion, periodic.stationary_cov)
    measurement = jnp.kron(matern.measurement, periodic.measurement)
    stationary_cov = jnp.kron(matern.stationary_cov, periodic.stationary_cov)

    def state_transition(dt: jax.Array) -> jax.Array:
        """Return the discrete-time state-transition matrix over a step ``dt``."""
        return jnp.kron(matern.state_transition(dt), periodic.state_transition(dt))

    return StateSpaceKernel(
        feedback=feedback,
        noise_effect=noise_effect,
        diffusion=diffusion,
        measurement=measurement,
        stationary_cov=stationary_cov,
        state_transition=state_transition,
    )

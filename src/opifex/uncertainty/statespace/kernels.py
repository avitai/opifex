r"""State-space kernel constructors for temporal Gaussian processes.

Each constructor returns a :class:`StateSpaceKernel` carrying the continuous-time SDE parameters
``(F, L, Q_c, H, P_inf)``. The transition ``A(dt) = exp(F dt)`` of every constructor has a closed
form, and the process noise ``Q(dt)`` of a step comes from the method that stays accurate for that
kernel in that precision:

* Matern-1/2: ``Q = sigma^2 (-expm1(-2 dt / ell))`` (Särkkä & Solin 2019, eq. 6.30; GPJax
  ``gpjax/state_space/sde.py`` ``Matern12SDE.discretise`` at 4f24c59).
* Cosine and periodic: ``Q = 0``, because the SDE only rotates the state.
* Matern-3/2 to Matern-7/2, quasi-periodic, and a kernel built from its matrices alone: the
  stationary identity ``Q = P_inf - A P_inf A^T`` in float64 (bayesnewton ``ops.py:149-150``),
  and the Stillfjord & Tronarp Gramian of :mod:`opifex.uncertainty.statespace._gramian` in
  float32, where the identity loses the small components of ``Q``.

The precision is read from the arrays while tracing, so float32 and float64 each compile their own
program. A kernel is a pytree whose leaves are its matrices, so new hyperparameter values reuse a
compiled program.

Canonical reference (line-by-line port of the closed-form transitions):
* ``../bayesnewton/bayesnewton/kernels.py`` at f72ae9a — ``Matern12.state_transition`` (line 158),
  ``Matern32`` (216), ``Matern52`` (273), ``Matern72`` (344), ``Cosine`` (788), ``Periodic``
  (858), ``QuasiPeriodicMatern12`` (953).

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
from enum import StrEnum

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.math import bessel_ive

from opifex.uncertainty.statespace._gramian import diffusion_factor, exponential_and_gramian


class _TransitionFamily(StrEnum):
    """How a kernel computes its transition and its process noise."""

    GENERIC = "generic"
    STATE_TRANSITION = "state_transition"
    MATERN12 = "matern12"
    MATERN32 = "matern32"
    MATERN52 = "matern52"
    MATERN72 = "matern72"
    ROTATION = "rotation"
    QUASI_PERIODIC_MATERN12 = "quasi_periodic_matern12"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True, init=False)
class StateSpaceKernel:
    """Continuous-time SDE representation of a stationary temporal GP kernel.

    Attributes:
        feedback: Drift matrix ``F`` (shape ``(n, n)``).
        noise_effect: Dispersion matrix ``L`` (shape ``(n, k)``).
        diffusion: Wiener diffusion ``Q_c`` (shape ``(k, k)``).
        measurement: Observation matrix ``H`` (shape ``(1, n)``).
        stationary_cov: Stationary covariance ``P_inf`` (shape ``(n, n)``),
            satisfying ``F P_inf + P_inf F^T + L Q_c L^T = 0``.
        transition_parameters: Rates and angular frequencies of a closed-form transition.
        transition_family: Static tag selecting the transition and process-noise method.
        legacy_state_transition: The deprecated user-supplied ``A(dt)``, if any.
    """

    feedback: jax.Array
    noise_effect: jax.Array
    diffusion: jax.Array
    measurement: jax.Array
    stationary_cov: jax.Array
    transition_parameters: jax.Array
    transition_family: _TransitionFamily
    legacy_state_transition: Callable[[jax.Array], jax.Array] | None

    def __init__(
        self,
        *,
        feedback: jax.Array,
        noise_effect: jax.Array,
        diffusion: jax.Array,
        measurement: jax.Array,
        stationary_cov: jax.Array,
        state_transition: Callable[[jax.Array], jax.Array] | None = None,
    ) -> None:
        """Build a kernel from its SDE matrices; the transition then follows from ``F``.

        Args:
            feedback: Drift matrix ``F``.
            noise_effect: Dispersion matrix ``L``.
            diffusion: Wiener diffusion ``Q_c``.
            measurement: Observation matrix ``H``.
            stationary_cov: Stationary covariance ``P_inf``.
            state_transition: Deprecated closed-form ``A(dt) = exp(F dt)``. When given, the
                kernel discretises as in opifex 0.2.5: ``A`` from this callable and
                ``Q = P_inf - A P_inf A^T``.
        """
        if state_transition is not None:
            warnings.warn(
                "StateSpaceKernel(state_transition=...) is deprecated; omit it and the "
                "transition is computed from the feedback matrix.",
                DeprecationWarning,
                stacklevel=2,
            )
        _assign_fields(
            self,
            feedback=feedback,
            noise_effect=noise_effect,
            diffusion=diffusion,
            measurement=measurement,
            stationary_cov=stationary_cov,
            transition_parameters=jnp.zeros((0,), dtype=jnp.result_type(feedback)),
            transition_family=(
                _TransitionFamily.GENERIC
                if state_transition is None
                else _TransitionFamily.STATE_TRANSITION
            ),
            legacy_state_transition=state_transition,
        )

    @classmethod
    def _with_closed_form(
        cls,
        *,
        feedback: jax.Array,
        noise_effect: jax.Array,
        diffusion: jax.Array,
        measurement: jax.Array,
        stationary_cov: jax.Array,
        transition_parameters: jax.Array,
        transition_family: _TransitionFamily,
    ) -> StateSpaceKernel:
        """Return a kernel whose transition has the closed form of ``transition_family``."""
        kernel = object.__new__(cls)
        _assign_fields(
            kernel,
            feedback=feedback,
            noise_effect=noise_effect,
            diffusion=diffusion,
            measurement=measurement,
            stationary_cov=stationary_cov,
            transition_parameters=transition_parameters,
            transition_family=transition_family,
            legacy_state_transition=None,
        )
        return kernel

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[jax.Array, ...],
        tuple[_TransitionFamily, Callable[[jax.Array], jax.Array] | None],
    ]:
        """Pytree flatten: the matrices and transition parameters are the leaves."""
        children = (
            self.feedback,
            self.noise_effect,
            self.diffusion,
            self.measurement,
            self.stationary_cov,
            self.transition_parameters,
        )
        return children, (self.transition_family, self.legacy_state_transition)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[_TransitionFamily, Callable[[jax.Array], jax.Array] | None],
        children: tuple[jax.Array, ...],
    ) -> StateSpaceKernel:
        """Rebuild without ``__init__``, so a rebuilt deprecated kernel does not warn again."""
        feedback, noise_effect, diffusion, measurement, stationary_cov, parameters = children
        transition_family, legacy_state_transition = aux_data
        kernel = object.__new__(cls)
        _assign_fields(
            kernel,
            feedback=feedback,
            noise_effect=noise_effect,
            diffusion=diffusion,
            measurement=measurement,
            stationary_cov=stationary_cov,
            transition_parameters=parameters,
            transition_family=transition_family,
            legacy_state_transition=legacy_state_transition,
        )
        return kernel

    def _supplied_state_transition(self) -> Callable[[jax.Array], jax.Array]:
        """Return the deprecated callable of a kernel built with ``state_transition=``."""
        if self.legacy_state_transition is None:
            raise TypeError("This kernel was not built with state_transition=.")
        return self.legacy_state_transition

    @property
    def state_dim(self) -> int:
        """Dimension of the state-space representation."""
        return int(self.feedback.shape[0])

    def state_transition(self, dt: jax.Array) -> jax.Array:
        """Return the discrete-time state-transition matrix ``A(dt) = exp(F dt)``."""
        if self.transition_family is _TransitionFamily.GENERIC:
            return self.discretize(dt)[0]
        return self._closed_form_transitions(jnp.reshape(dt, (1,)))[0]

    def discretize(self, dt: jax.Array) -> tuple[jax.Array, jax.Array]:
        r"""Return the transition ``A(dt)`` and process noise ``Q(dt)`` of one step.

        Args:
            dt: Scalar, non-negative step length.

        Returns:
            ``(A(dt), Q(dt))``, each of shape ``(n, n)``.
        """
        transitions, process_noises = self.discretize_steps(jnp.reshape(dt, (1,)))
        return transitions[0], process_noises[0]

    def discretize_steps(self, steps: jax.Array) -> tuple[jax.Array, jax.Array]:
        r"""Return the transitions and process noises of a sequence of steps.

        Discretising a whole sequence in one call lets the float32 Gramian run its extra doubling
        block only when some step of the sequence needs it.

        Args:
            steps: Non-negative step lengths of shape ``(N,)``.

        Returns:
            ``(transitions, process_noises)``, each of shape ``(N, n, n)``.
        """
        if self.transition_family is _TransitionFamily.GENERIC:
            exponentials, gramians = exponential_and_gramian(
                self.feedback, diffusion_factor(self.noise_effect, self.diffusion), steps
            )
            if _is_float64(self.feedback, steps):
                return exponentials, _stationary_identity(exponentials, self.stationary_cov)
            return exponentials, gramians
        transitions = self._closed_form_transitions(steps)
        return transitions, self._closed_form_process_noises(transitions, steps)

    def _closed_form_process_noises(self, transitions: jax.Array, steps: jax.Array) -> jax.Array:
        """Return the process noise of every step for a kernel with a closed-form transition."""
        family = self.transition_family
        if family is _TransitionFamily.MATERN12:
            decay = -2.0 * self.transition_parameters[0] * steps
            return jnp.reshape(self.stationary_cov[0, 0] * -jnp.expm1(decay), (-1, 1, 1))
        if family is _TransitionFamily.ROTATION:
            return jnp.zeros_like(transitions)
        if family is _TransitionFamily.STATE_TRANSITION or _is_float64(self.feedback, steps):
            return _stationary_identity(transitions, self.stationary_cov)
        _, gramians = exponential_and_gramian(
            self.feedback, diffusion_factor(self.noise_effect, self.diffusion), steps
        )
        return gramians

    def _closed_form_transitions(self, steps: jax.Array) -> jax.Array:
        """Return the closed-form ``exp(F dt)`` of every step for this kernel's family."""
        family = self.transition_family
        parameters = self.transition_parameters
        if family is _TransitionFamily.STATE_TRANSITION:
            return jax.vmap(self._supplied_state_transition())(steps)
        if family is _TransitionFamily.ROTATION:
            return jax.vmap(lambda dt: _rotations(dt, parameters))(steps)
        if family is _TransitionFamily.QUASI_PERIODIC_MATERN12:
            return jax.vmap(
                lambda dt: jnp.exp(-dt * parameters[0]) * _rotations(dt, parameters[1:])
            )(steps)
        closed_form = _MATERN_TRANSITIONS[family]
        return jax.vmap(lambda dt: closed_form(dt, parameters[0]))(steps)


def _is_float64(*arrays: jax.Array) -> bool:
    """Return whether the arrays combine to float64; the dtype is static while tracing."""
    return jnp.result_type(*arrays) == jnp.float64


def _assign_fields(kernel: StateSpaceKernel, **fields: object) -> None:
    """Set the fields of a frozen kernel."""
    for name, value in fields.items():
        object.__setattr__(kernel, name, value)


def _stationary_identity(transitions: jax.Array, stationary_cov: jax.Array) -> jax.Array:
    """Return ``P_inf - A P_inf A^T`` for every transition (bayesnewton ``ops.py:149-150``)."""
    return stationary_cov[None] - jnp.einsum(
        "kij,jl,kml->kim", transitions, stationary_cov, transitions
    )


def _matern12_transition(dt: jax.Array, rate: jax.Array) -> jax.Array:
    """Bayesnewton ``Matern12.state_transition``: ``A = exp(-dt / lengthscale)``."""
    return jnp.reshape(jnp.exp(-dt * rate), (1, 1))


def _matern32_transition(dt: jax.Array, lam: jax.Array) -> jax.Array:
    """Bayesnewton ``Matern32.state_transition``."""
    return jnp.exp(-dt * lam) * (
        dt * jnp.asarray([[lam, 1.0], [-(lam**2.0), -lam]]) + jnp.eye(2, dtype=jnp.result_type(lam))
    )


def _matern52_transition(dt: jax.Array, lam: jax.Array) -> jax.Array:
    """Bayesnewton ``Matern52.state_transition``."""
    dtlam = dt * lam
    return jnp.exp(-dtlam) * (
        dt
        * jnp.asarray(
            [
                [lam * (0.5 * dtlam + 1.0), dtlam + 1.0, 0.5 * dt],
                [-0.5 * dtlam * lam**2, lam * (1.0 - dtlam), 1.0 - 0.5 * dtlam],
                [lam**3 * (0.5 * dtlam - 1.0), lam**2 * (dtlam - 3), lam * (0.5 * dtlam - 2.0)],
            ]
        )
        + jnp.eye(3, dtype=jnp.result_type(lam))
    )


def _matern72_transition(dt: jax.Array, lam: jax.Array) -> jax.Array:
    """Bayesnewton ``Matern72.state_transition``."""
    lam2 = lam * lam
    lam3 = lam2 * lam
    dtlam = dt * lam
    dtlam2 = dtlam**2
    return jnp.exp(-dtlam) * (
        dt
        * jnp.asarray(
            [
                [
                    lam * (1.0 + 0.5 * dtlam + dtlam2 / 6.0),
                    1.0 + dtlam + 0.5 * dtlam2,
                    0.5 * dt * (1.0 + dtlam),
                    dt**2 / 6,
                ],
                [
                    -dtlam2 * lam**2.0 / 6.0,
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
        + jnp.eye(4, dtype=jnp.result_type(lam))
    )


_MATERN_TRANSITIONS: dict[_TransitionFamily, Callable[[jax.Array, jax.Array], jax.Array]] = {
    _TransitionFamily.MATERN12: _matern12_transition,
    _TransitionFamily.MATERN32: _matern32_transition,
    _TransitionFamily.MATERN52: _matern52_transition,
    _TransitionFamily.MATERN72: _matern72_transition,
}


def _rotations(dt: jax.Array, angular_frequencies: jax.Array) -> jax.Array:
    """Return the block-diagonal rotations by ``angular_frequencies * dt`` (bayesnewton)."""
    angles = angular_frequencies * dt
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    blocks = jnp.stack([jnp.stack([cos, -sin], axis=-1), jnp.stack([sin, cos], axis=-1)], axis=-2)
    return jax.scipy.linalg.block_diag(*blocks)


def matern12_kernel(
    *, variance: float | jax.Array, lengthscale: float | jax.Array
) -> StateSpaceKernel:
    r"""Matern-1/2 (exponential) kernel in SDE form.

    State dimension 1. ``F = [[-1/ell]]``, ``Q_c = 2 sigma^2 / ell``,
    ``A(dt) = exp(-dt/ell)``. Ports bayesnewton ``Matern12`` (line 141).
    """
    return StateSpaceKernel._with_closed_form(
        feedback=jnp.asarray([[-1.0 / lengthscale]]),
        noise_effect=jnp.asarray([[1.0]]),
        diffusion=jnp.asarray([[2.0 * variance / lengthscale]]),
        measurement=jnp.asarray([[1.0]]),
        stationary_cov=jnp.asarray([[variance]]),
        transition_parameters=jnp.asarray([1.0 / lengthscale]),
        transition_family=_TransitionFamily.MATERN12,
    )


def matern32_kernel(
    *, variance: float | jax.Array, lengthscale: float | jax.Array
) -> StateSpaceKernel:
    r"""Matern-3/2 kernel in SDE form. Ports bayesnewton ``Matern32`` (line 200)."""
    lam = jnp.sqrt(3.0) / lengthscale
    return StateSpaceKernel._with_closed_form(
        feedback=jnp.asarray([[0.0, 1.0], [-(lam**2), -2.0 * lam]]),
        noise_effect=jnp.asarray([[0.0], [1.0]]),
        diffusion=jnp.asarray([[12.0 * jnp.sqrt(3.0) / lengthscale**3 * variance]]),
        measurement=jnp.asarray([[1.0, 0.0]]),
        stationary_cov=jnp.asarray([[variance, 0.0], [0.0, 3.0 * variance / lengthscale**2]]),
        transition_parameters=jnp.reshape(lam, (1,)),
        transition_family=_TransitionFamily.MATERN32,
    )


def matern52_kernel(
    *, variance: float | jax.Array, lengthscale: float | jax.Array
) -> StateSpaceKernel:
    r"""Matern-5/2 kernel in SDE form. Ports bayesnewton ``Matern52`` (line 253)."""
    lam = jnp.sqrt(5.0) / lengthscale
    kappa = 5.0 / 3.0 * variance / lengthscale**2
    return StateSpaceKernel._with_closed_form(
        feedback=jnp.asarray(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-(lam**3), -3.0 * lam**2, -3.0 * lam]]
        ),
        noise_effect=jnp.asarray([[0.0], [0.0], [1.0]]),
        diffusion=jnp.asarray([[variance * 400.0 * jnp.sqrt(5.0) / 3.0 / lengthscale**5]]),
        measurement=jnp.asarray([[1.0, 0.0, 0.0]]),
        stationary_cov=jnp.asarray(
            [
                [variance, 0.0, -kappa],
                [0.0, kappa, 0.0],
                [-kappa, 0.0, 25.0 * variance / lengthscale**4],
            ]
        ),
        transition_parameters=jnp.reshape(lam, (1,)),
        transition_family=_TransitionFamily.MATERN52,
    )


def matern72_kernel(
    *, variance: float | jax.Array, lengthscale: float | jax.Array
) -> StateSpaceKernel:
    r"""Matern-7/2 kernel in SDE form. Ports bayesnewton ``Matern72`` (line 321)."""
    lam = jnp.sqrt(7.0) / lengthscale
    kappa = 7.0 / 5.0 * variance / lengthscale**2
    kappa2 = 9.8 * variance / lengthscale**4
    return StateSpaceKernel._with_closed_form(
        feedback=jnp.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-(lam**4), -4.0 * lam**3, -6.0 * lam**2, -4.0 * lam],
            ]
        ),
        noise_effect=jnp.asarray([[0.0], [0.0], [0.0], [1.0]]),
        diffusion=jnp.asarray([[variance * 10976.0 * jnp.sqrt(7.0) / 5.0 / lengthscale**7]]),
        measurement=jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
        stationary_cov=jnp.asarray(
            [
                [variance, 0.0, -kappa, 0.0],
                [0.0, kappa, 0.0, -kappa2],
                [-kappa, 0.0, kappa2, 0.0],
                [0.0, -kappa2, 0.0, 343.0 * variance / lengthscale**6],
            ]
        ),
        transition_parameters=jnp.reshape(lam, (1,)),
        transition_family=_TransitionFamily.MATERN72,
    )


def cosine_kernel(*, frequency: float | jax.Array) -> StateSpaceKernel:
    r"""Cosine kernel as SDE. Ports bayesnewton ``Cosine`` (line 770).

    State dim 2; transition is the 2-D rotation by angle ``frequency dt``.
    """
    return StateSpaceKernel._with_closed_form(
        feedback=jnp.asarray([[0.0, -frequency], [frequency, 0.0]]),
        noise_effect=jnp.zeros((2, 0)),
        diffusion=jnp.zeros((0, 0)),
        measurement=jnp.asarray([[1.0, 0.0]]),
        stationary_cov=jnp.eye(2),
        transition_parameters=jnp.reshape(jnp.asarray(frequency), (1,)),
        transition_family=_TransitionFamily.ROTATION,
    )


def periodic_kernel(
    *,
    variance: float | jax.Array,
    lengthscale: float | jax.Array,
    period: float | jax.Array,
    order: int = 6,
) -> StateSpaceKernel:
    r"""Periodic kernel via Bessel-weighted sum of harmonic rotations.

    Ports bayesnewton ``Periodic`` (line 802). State dim ``2(order + 1)``;
    transition is block-diagonal of harmonic rotation matrices. With
    ``x = lengthscale**-2`` the harmonic variances are ``sigma^2 I_0(x) e^{-x}``
    and ``2 sigma^2 I_n(x) e^{-x}``, the cosine-series coefficients of
    ``sigma^2 exp(x (cos(omega tau) - 1))`` (DLMF 10.35.1). Truncating at
    ``order`` changes the covariance by at most ``2 sigma^2 sum_{n > order}
    I_n(x) e^{-x}``. As in bayesnewton's ``Periodic.kernel_to_state_space``,
    ``I_n(x) e^{-x}`` comes from TensorFlow Probability's ``bessel_ive``.
    """
    omega = 2.0 * jnp.pi / period
    harmonic_indices = jnp.arange(order + 1)
    state_size = 2 * (order + 1)
    bessel_factors = bessel_ive(
        jnp.asarray(harmonic_indices, dtype=jnp.result_type(float)),
        jnp.asarray(1.0 / lengthscale**2, dtype=jnp.result_type(float)),
    )
    q2 = jnp.concatenate([jnp.asarray([1.0]), 2.0 * jnp.ones(order)]) * variance * bessel_factors
    return StateSpaceKernel._with_closed_form(
        feedback=jnp.kron(
            jnp.diag(harmonic_indices.astype(jnp.result_type(float))),
            jnp.asarray([[0.0, -omega], [omega, 0.0]]),
        ),
        noise_effect=jnp.eye(state_size),
        diffusion=jnp.zeros((state_size, state_size)),
        measurement=jnp.kron(jnp.ones((1, order + 1)), jnp.asarray([[1.0, 0.0]])),
        stationary_cov=jnp.kron(jnp.diag(q2), jnp.eye(2)),
        transition_parameters=harmonic_indices * omega,
        transition_family=_TransitionFamily.ROTATION,
    )


def i0e_vector(orders: jax.Array, argument: float) -> jax.Array:
    """Return ``I_n(x) e^{-x}`` at the requested orders.

    Deprecated: use ``tensorflow_probability.substrates.jax.math.bessel_ive``, which this
    name now delegates to.

    Args:
        orders: Non-negative integer orders.
        argument: Positive scalar ``x``.

    Returns:
        The values at ``orders``.
    """
    warnings.warn(
        "i0e_vector is deprecated; use "
        "tensorflow_probability.substrates.jax.math.bessel_ive(orders, argument).",
        DeprecationWarning,
        stacklevel=2,
    )
    dtype = jnp.result_type(float)
    return bessel_ive(jnp.asarray(orders, dtype=dtype), jnp.asarray(argument, dtype=dtype))


def quasi_periodic_matern12_kernel(
    *,
    variance: float | jax.Array,
    lengthscale_periodic: float | jax.Array,
    period: float | jax.Array,
    lengthscale_matern: float | jax.Array,
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
    state_size = matern.state_dim * periodic.state_dim
    return StateSpaceKernel._with_closed_form(
        feedback=jnp.kron(matern.feedback, jnp.eye(periodic.state_dim))
        + jnp.kron(jnp.eye(matern.state_dim), periodic.feedback),
        noise_effect=jnp.eye(state_size),
        # bayesnewton QuasiPeriodicMatern12 builds the same product diffusion,
        # ``Qc = np.kron(Qc_m, Pinf_p)`` (kernels.py:935 at f72ae9a); its Matern-3/2 sibling cites
        # Solin & Sarkka (2014), eq. (32).
        diffusion=jnp.kron(matern.diffusion, periodic.stationary_cov),
        measurement=jnp.kron(matern.measurement, periodic.measurement),
        stationary_cov=jnp.kron(matern.stationary_cov, periodic.stationary_cov),
        transition_parameters=jnp.concatenate(
            [matern.transition_parameters, periodic.transition_parameters]
        ),
        transition_family=_TransitionFamily.QUASI_PERIODIC_MATERN12,
    )

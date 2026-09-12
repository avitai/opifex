r"""Continuous-time linear time-invariant SDE → discrete-time transition.

Given the continuous-time linear SDE :math:`dx = F x\, dt + L\, dW` with
stationary diffusion :math:`Q_c` for the Wiener process :math:`W`, the
discrete-time transition over an interval :math:`\Delta t` is
:math:`x(t+\Delta t)\,|\, x(t) \sim \mathcal{N}(A x(t),\, Q)`, where
:math:`A = \exp(F\, \Delta t)` and

.. math::

    Q = \int_0^{\Delta t} e^{F\tau} L Q_c L^\top e^{F^\top\tau}\, d\tau.

Van Loan (1978) Theorem 1 computes both quantities in one matrix
exponential by exploiting the block structure of

.. math::

    \Phi = \begin{bmatrix} F & L Q_c L^\top \\ 0 & -F^\top \end{bmatrix}
    \Delta t.

Exponentiating the block at a large step fails: its ``exp(-F^T dt)`` part grows
while the process noise saturates, and ``jax.scipy.linalg.expm`` returns NaN once
it needs more than 16 squarings. Following Van Loan (1978, section III), the step
is split into ``dt / 2^j`` so that the block's 1-norm is at most 3.5, below the
float32 Pade-7 bound of ``expm`` (3.9257), and the result is doubled back with his
eq. (3.5): ``Q(2t) = Q(t) + A(t) Q(t) A(t)^T`` and ``A(2t) = A(t)^2``. ``Q`` is
linear in ``Q_c``, so ``L Q_c L^T`` is normalised by its largest entry first; a
large diffusion then adds no doublings.

Canonical reference (line-by-line port):
* ``../probnum/src/probnum/randprocs/markov/continuous/_mfd.py``
  ``matrix_fraction_decomposition``.

References:
----------
* Van Loan, C. F. 1978 — *Computing integrals involving the matrix
  exponential*, IEEE TAC 23(3).
* Särkkä & Solin 2019 — *Applied Stochastic Differential Equations*
  §6.2 eqn 6.18.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm


# Largest 1-norm of the step block that is exponentiated directly. It stays below the float32
# Pade-7 bound of ``jax.scipy.linalg.expm`` (3.9257), so expm neither squares nor truncates.
# Measured in float32 over Matern SDEs from 1e-4 to 1e3 lengthscales: process noise within 3.6e-6
# and transition within 2.0e-6 of the closed forms, against 7.8e-6 and 3.9e-5 when the block is
# kept below 0.5 as Van Loan (1978) chooses for double precision.
_BLOCK_EXPONENTIAL_NORM = 3.5
# Doublings available to one call. Steps whose block needs more return NaN, as ``expm`` does past
# its own squaring limit; 32 doublings cover a block 1-norm of about 1.5e10.
_MAX_DOUBLINGS = 32


def discretize_lti_sde(
    *,
    drift_matrix: jax.Array,
    dispersion_matrix: jax.Array,
    dt: jax.Array,
    diffusion: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""Discretize a continuous-time linear time-invariant SDE.

    Computes the discrete-time state transition matrix
    :math:`A = \exp(F\, \Delta t)` and the integrated process noise
    covariance via Van Loan's matrix-fraction decomposition, evaluated on
    ``dt / 2^j`` and doubled back (Van Loan 1978, section III, eq. 3.5) so
    that coarse steps stay finite. A step that would need more than 32
    doublings returns NaN.

    Args:
        drift_matrix: Continuous-time drift :math:`F` of shape ``(n, n)``.
        dispersion_matrix: Dispersion :math:`L` of shape ``(n, k)``.
        dt: Time step :math:`\Delta t` (scalar).
        diffusion: Wiener-process diffusion :math:`Q_c` of shape
            ``(k, k)``. Defaults to the identity matrix if ``None``.

    Returns:
        ``(transition, process_noise)`` of shapes ``(n, n)`` and
        ``(n, n)``.
    """
    state_dim = drift_matrix.shape[0]
    diffusion_array = (
        diffusion
        if diffusion is not None
        else jnp.eye(dispersion_matrix.shape[1], dtype=drift_matrix.dtype)
    )
    process_diffusion = dispersion_matrix @ diffusion_array @ dispersion_matrix.T
    noise_scale = jnp.max(jnp.abs(process_diffusion), initial=0.0)
    noise_scale = jnp.where(noise_scale > 0.0, noise_scale, 1.0)

    upper_block = jnp.concatenate([drift_matrix, process_diffusion / noise_scale], axis=1)
    lower_block = jnp.concatenate([jnp.zeros_like(drift_matrix), -drift_matrix.T], axis=1)
    block = jnp.concatenate([upper_block, lower_block], axis=0)
    block_norm = jnp.max(jnp.sum(jnp.abs(block), axis=0)) * jnp.abs(dt)
    doublings = jnp.ceil(jnp.log2(jnp.maximum(block_norm / _BLOCK_EXPONENTIAL_NORM, 1.0)))
    exponential = expm(block * (dt / 2.0**doublings))
    transition = exponential[:state_dim, :state_dim]
    process_noise = exponential[:state_dim, state_dim:] @ transition.T

    def double(
        carry: tuple[jax.Array, jax.Array], index: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        """Apply one doubling ``(A, Q) -> (A^2, Q + A Q A^T)`` while doublings remain."""
        step_transition, step_noise = carry
        is_active = index < doublings
        doubled_transition = jnp.where(
            is_active, step_transition @ step_transition, step_transition
        )
        doubled_noise = jnp.where(
            is_active, step_noise + step_transition @ step_noise @ step_transition.T, step_noise
        )
        return (doubled_transition, doubled_noise), None

    (transition, process_noise), _ = jax.lax.scan(
        double, (transition, process_noise), jnp.arange(_MAX_DOUBLINGS, dtype=doublings.dtype)
    )
    process_noise = 0.5 * (process_noise + process_noise.T) * noise_scale
    is_beyond_limit = doublings > _MAX_DOUBLINGS
    return (
        jnp.where(is_beyond_limit, jnp.nan, transition),
        jnp.where(is_beyond_limit, jnp.nan, process_noise),
    )


def state_transition_matrix(
    *,
    drift_matrix: jax.Array,
    dispersion_matrix: jax.Array,
    dt: jax.Array,
    diffusion: jax.Array | None = None,
) -> jax.Array:
    r"""Return only ``A = exp(F dt)`` from the LTI-SDE discretisation.

    Thin convenience wrapper around :func:`discretize_lti_sde` for the
    common case where only the transition matrix is needed (e.g.
    independent discretisation of a kernel's continuous-time SDE).
    """
    transition, _ = discretize_lti_sde(
        drift_matrix=drift_matrix,
        dispersion_matrix=dispersion_matrix,
        dt=dt,
        diffusion=diffusion,
    )
    return transition


def process_noise_covariance(
    *,
    drift_matrix: jax.Array,
    dispersion_matrix: jax.Array,
    dt: jax.Array,
    diffusion: jax.Array | None = None,
) -> jax.Array:
    r"""Return only ``Q`` (Van Loan process-noise) from the LTI-SDE discretisation.

    Thin convenience wrapper around :func:`discretize_lti_sde` for the
    common case where only the process-noise covariance is needed.
    """
    _, process_noise = discretize_lti_sde(
        drift_matrix=drift_matrix,
        dispersion_matrix=dispersion_matrix,
        dt=dt,
        diffusion=diffusion,
    )
    return process_noise

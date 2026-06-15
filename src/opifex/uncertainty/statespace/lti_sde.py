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

Canonical reference (line-by-line port):
* ``../probnum/src/probnum/randprocs/markov/continuous/_mfd.py``
  ``matrix_fraction_decomposition``.

References
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
    covariance via Van Loan's matrix-fraction decomposition.

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

    upper_block = jnp.concatenate([drift_matrix, process_diffusion], axis=1)
    lower_block = jnp.concatenate([jnp.zeros_like(drift_matrix), -drift_matrix.T], axis=1)
    phi = jnp.concatenate([upper_block, lower_block], axis=0) * dt
    exponential = expm(phi)

    transition = exponential[:state_dim, :state_dim]
    cross_block = exponential[:state_dim, state_dim:]
    process_noise = cross_block @ transition.T
    return transition, process_noise

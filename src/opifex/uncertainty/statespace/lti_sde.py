r"""Continuous-time linear time-invariant SDE → discrete-time transition.

Given the continuous-time linear SDE :math:`dx = F x\, dt + L\, dW` with
stationary diffusion :math:`Q_c` for the Wiener process :math:`W`, the
discrete-time transition over an interval :math:`\Delta t` is
:math:`x(t+\Delta t)\,|\, x(t) \sim \mathcal{N}(A x(t),\, Q)`, where
:math:`A = \exp(F\, \Delta t)` and

.. math::

    Q = \int_0^{\Delta t} e^{F\tau} L Q_c L^\top e^{F^\top\tau}\, d\tau.

Both come from the exponential-and-Gramian doubling of Stillfjord & Tronarp (2023,
arXiv:2310.13462) in :mod:`opifex.uncertainty.statespace._gramian`, ported from probdiffeq. It
keeps every component of ``Q`` accurate from steps far below to far above the SDE's time scale,
in float32 and float64, and it is differentiable in reverse mode. Steps needing more than 32
doublings return NaN.

The Gramian needs a factor ``B`` with ``B B^T = L Q_c L^T``. A positive-definite ``Q_c`` is
factored by Cholesky; a singular positive semi-definite ``Q_c``, which the Cholesky decomposition
rejects, by its symmetric square root.

References:
----------
* Stillfjord, T. & Tronarp, F. 2023 — *Computing the matrix exponential and the Cholesky factor of
  a related finite horizon Gramian*, arXiv:2310.13462.
* Särkkä & Solin 2019 — *Applied Stochastic Differential Equations*
  §6.2 eqn 6.18.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace._gramian import diffusion_factor, exponential_and_gramian


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
    covariance with the exponential-and-Gramian doubling of Stillfjord & Tronarp (2023).
    A step that would need more than 32 doublings returns NaN.

    Args:
        drift_matrix: Continuous-time drift :math:`F` of shape ``(n, n)``.
        dispersion_matrix: Dispersion :math:`L` of shape ``(n, k)``.
        dt: Time step :math:`\Delta t` (scalar, non-negative).
        diffusion: Wiener-process diffusion :math:`Q_c` of shape
            ``(k, k)``, positive semi-definite. Defaults to the identity matrix if ``None``.

    Returns:
        ``(transition, process_noise)`` of shapes ``(n, n)`` and
        ``(n, n)``.
    """
    dtype = jnp.result_type(drift_matrix, dispersion_matrix, dt)
    factor = diffusion_factor(
        jnp.asarray(dispersion_matrix, dtype=dtype),
        None if diffusion is None else jnp.asarray(diffusion, dtype=dtype),
    )
    transitions, process_noises = exponential_and_gramian(
        jnp.asarray(drift_matrix, dtype=dtype),
        factor,
        jnp.reshape(jnp.asarray(dt, dtype=dtype), (1,)),
    )
    return transitions[0], process_noises[0]


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
    r"""Return only ``Q`` (the integrated process noise) from the LTI-SDE discretisation.

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

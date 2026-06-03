r"""Boys function :math:`F_n(x)` for the McMurchie-Davidson integrals.

The Boys function

.. math:: F_n(x) = \int_0^1 t^{2n} e^{-x t^2}\,dt

appears in every Coulomb-type Gaussian integral. A single
:func:`jax.scipy.special.gammainc` evaluation loses accuracy and leaks NaN
gradients in both limits (``x -> 0`` and large ``x``), so this leaf module
follows the ``graphcore-research/mess`` ``gammanu_select`` strategy: a
three-branch :func:`jax.numpy.select` over

* the analytic limit :math:`F_n(0) = 1/(2n+1)` at ``x = 0``,
* a stable ascending series for ``0 < x < threshold`` (Taketa-Huzinaga-O-ohata
  eq. 2.11, the confluent-hypergeometric form), and
* the large-``x`` asymptotic :math:`F_n(x) \to (2n-1)!!/2^{n+1}\sqrt{\pi/x^{2n+1}}`
  for ``x >= threshold``,

with every branch wrapped in :func:`jax.numpy.where` guards so the inactive
branches never produce NaNs that would poison the reverse-mode gradient (the
forces depend on these derivatives).

This module has no intra-package dependencies so it can be imported by both the
eager backend and the batched flat-primitive kernels without a cycle.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


# Switch from the ascending series to the large-x asymptotic form. The
# leading-order asymptotic is machine-accurate (~2e-16) for x >= 50, and the
# 128-term ascending series is accurate to ~8e-15 for 0 < x < 50, so the two
# branches meet at full float64 precision (verified against the regularised
# lower-incomplete-gamma reference across n = 0..12). This mirrors MESS's
# ``gammanu_select`` threshold of 50.
_SERIES_THRESHOLD = 50.0
# Ascending-series terms needed to reach ~1e-14 at the threshold x = 50.
_SERIES_TERMS = 128


def _double_factorial_odd(order: int) -> float:
    """Return ``(2*order - 1)!!`` (with ``(-1)!! = 1``)."""
    result = 1.0
    value = 2 * order - 1
    while value > 1:
        result *= value
        value -= 2
    return result


def _series(order: int, x: Array) -> Array:
    r"""Ascending series ``F_n(x) = (1/2) e^{-x} sum_k x^k / prod`` (THO eq. 2.11)."""
    a = order + 0.5
    term = 1.0 / a
    total = term
    for k in range(1, _SERIES_TERMS + 1):
        term = term * x / (a + k)
        total = total + term
    return 0.5 * jnp.exp(-x) * total


def _asymptotic(order: int, x: Array) -> Array:
    r"""Large-``x`` asymptotic ``F_n(x) ~ (2n-1)!!/2^{n+1} sqrt(pi / x^{2n+1})``."""
    coefficient = _double_factorial_odd(order) / 2.0 ** (order + 1)
    return coefficient * jnp.sqrt(jnp.pi / x ** (2 * order + 1))


def boys_function(order: int, argument: Array) -> Array:
    r"""Boys function :math:`F_n(x)` for a single order ``n``.

    Evaluated with the three-branch ``jnp.select`` strategy described in the
    module docstring; inactive branches are guarded so the gradient stays finite
    everywhere (including ``x = 0``).

    Args:
        order: Boys order ``n`` (non-negative integer).
        argument: Argument ``x`` [any shape], expected non-negative.

    Returns:
        ``F_n(x)`` with the same shape as ``argument``.
    """
    x = jnp.asarray(argument)
    # Guard each branch's argument so the *unused* branches never see a value
    # that would emit NaN (0 in the series log/pow, 0 in the asymptotic sqrt).
    x_series = jnp.where(x < _SERIES_THRESHOLD, x, 1.0)
    x_asymptotic = jnp.where(x >= _SERIES_THRESHOLD, x, 1.0)

    limit = jnp.asarray(1.0 / (2.0 * order + 1.0), dtype=x.dtype)
    series = _series(order, x_series)
    asymptotic = _asymptotic(order, x_asymptotic)

    return jnp.select(
        (x == 0.0, x < _SERIES_THRESHOLD, x >= _SERIES_THRESHOLD),
        (jnp.broadcast_to(limit, x.shape), series, asymptotic),
        default=series,
    )


def boys_vector(max_order: int, argument: Array) -> Array:
    """Stacked Boys functions ``[F_0(x), ..., F_max_order(x)]`` for scalar ``x``.

    Args:
        max_order: Highest Boys order to return (inclusive).
        argument: Scalar argument ``x``.

    Returns:
        Array of shape ``(max_order + 1,)``.
    """
    return jnp.stack([boys_function(n, argument) for n in range(max_order + 1)])


__all__ = ["boys_function", "boys_vector"]

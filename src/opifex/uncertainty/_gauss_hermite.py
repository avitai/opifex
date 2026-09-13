r"""Gauss-Hermite quadrature for expectations under a Gaussian measure.

:func:`gauss_hermite_rule` returns nodes ``x_k`` and weights ``w_k`` with

.. math::

    \sum_k w_k\, g(x_k) \approx \mathbb{E}_{x \sim \mathcal{N}(0, 1)}[g(x)],

exact for polynomials of degree at most ``2n - 1`` (Golub & Welsch 1969, *Calculation of Gauss
Quadrature Rules*, Math. Comp. 23, 221-230). An expectation under ``N(m, v)`` evaluates ``g`` at
``m + sqrt(v) x_k``. The rule is NumPy's physicists' Gauss-Hermite rule
(``numpy.polynomial.hermite.hermgauss``, for the weight ``exp(-t^2)``) with the nodes scaled by
``sqrt(2)`` and the weights by ``1 / sqrt(pi)``. It is built on the host, so the point count is a
static Python integer under :func:`jax.jit` and the nodes and weights enter a compiled program as
constants.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from numpy.polynomial.hermite import hermgauss


def gauss_hermite_rule(num_points: int) -> tuple[jax.Array, jax.Array]:
    """Return Gauss-Hermite nodes and weights for the standard normal measure.

    Args:
        num_points: Number of quadrature points ``n``. The rule is exact for polynomials of
            degree at most ``2n - 1``.

    Returns:
        ``(nodes, weights)``, each of shape ``(num_points,)`` in JAX's default float precision,
        with the weights summing to one.

    Raises:
        ValueError: If ``num_points`` is smaller than one.
    """
    if num_points < 1:
        raise ValueError(f"num_points must be at least 1; got {num_points}.")
    nodes, weights = hermgauss(num_points)
    return jnp.asarray(math.sqrt(2.0) * nodes), jnp.asarray(weights / math.sqrt(math.pi))


__all__ = ["gauss_hermite_rule"]

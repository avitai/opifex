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

:func:`predictive_moments` uses the rule for the mean and variance of a response whose conditional
moments given a Gaussian latent are known, as a non-conjugate GP predictive needs.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from numpy.polynomial.hermite import hermgauss


ConditionalMomentsFn = Callable[[jax.Array], tuple[jax.Array, jax.Array]]
"""``f -> (E[y | f], Var[y | f])``: the conditional moments of a response, elementwise in ``f``."""

_PREDICTIVE_NUM_POINTS = 20


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


def predictive_moments(
    conditional_moments_fn: ConditionalMomentsFn,
    mean: jax.Array,
    variance: jax.Array,
    *,
    num_points: int = _PREDICTIVE_NUM_POINTS,
) -> tuple[jax.Array, jax.Array]:
    r"""Return the mean and variance of a response under a Gaussian latent.

    For ``f ~ N(mean, variance)`` and a response ``y`` with conditional moments ``E[y | f]`` and
    ``Var[y | f]``, the law of total variance gives

    .. math::

        \mathbb{E}[y] = \mathbb{E}\big[\mathbb{E}[y \mid f]\big], \qquad
        \operatorname{Var}[y] = \mathbb{E}\big[\operatorname{Var}[y \mid f]\big]
            + \mathbb{E}\big[(\mathbb{E}[y \mid f] - \mathbb{E}[y])^2\big].

    Each expectation uses :func:`gauss_hermite_rule`. The spread of the conditional mean is taken
    about ``E[y]`` rather than as ``E[E[y | f]^2] - E[y]^2``, which in float32 subtracts two nearly
    equal numbers when the response is concentrated. A variance that is not positive puts every node
    at ``mean``; the square root is then taken of a safe input, so gradients stay finite.

    Args:
        conditional_moments_fn: ``f -> (E[y | f], Var[y | f])``, elementwise in ``f``.
        mean: Latent means.
        variance: Latent variances, broadcastable against ``mean``.
        num_points: Gauss-Hermite points per expectation. Static under :func:`jax.jit`. Defaults
            to 20.

    Returns:
        ``(response_mean, response_variance)``, each with the broadcast shape of ``mean`` and
        ``variance``.
    """
    nodes, weights = gauss_hermite_rule(num_points)
    mean, variance = jnp.broadcast_arrays(mean, variance)
    has_spread = variance > 0.0
    deviation = jnp.where(has_spread, jnp.sqrt(jnp.where(has_spread, variance, 1.0)), 0.0)
    latent = mean[..., None] + deviation[..., None] * nodes
    conditional_mean, conditional_variance = conditional_moments_fn(latent)
    response_mean = jnp.sum(weights * conditional_mean, axis=-1)
    spread = conditional_mean - response_mean[..., None]
    response_variance = jnp.sum(weights * (conditional_variance + spread * spread), axis=-1)
    return response_mean, response_variance


__all__ = ["ConditionalMomentsFn", "gauss_hermite_rule", "predictive_moments"]

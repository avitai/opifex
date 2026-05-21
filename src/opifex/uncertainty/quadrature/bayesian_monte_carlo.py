r"""Bayesian Monte Carlo integration baseline.

Given samples ``x_1, ..., x_N`` drawn from a measure ``π`` and an
integrand ``f``, the Bayesian Monte Carlo estimate of ``∫ f dπ`` is the
sample mean

.. math::

    \\hat I_N = \\frac{1}{N} \\sum_{n=1}^N f(x_n),

with standard-error variance

.. math::

    \\operatorname{Var}(\\hat I_N) = \\frac{1}{N(N-1)} \\sum_{n=1}^N (f(x_n) - \\hat I_N)^2.

This baseline is the simplest no-GP integrator against which Bayesian
quadrature methods (WSABI-L, vanilla BQ, SOBER, FFBQ) are benchmarked.

Canonical reference:
* ``../emukit/emukit/quadrature/loop/bayesian_monte_carlo_loop.py``
  ``BayesianMonteCarlo`` (line 18).

References
----------
* Rasmussen, C. E. & Ghahramani, Z. 2003 — *Bayesian Monte Carlo*,
  NeurIPS 16.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass(slots=True, kw_only=True)
class IntegralEstimate:
    """Point estimate + standard-error variance of an integral.

    Registered as a JAX pytree (``flax.struct.dataclass``) so functions
    returning this type can be jit-compiled.

    Attributes:
        mean: Scalar estimate of ``∫ f dπ``.
        variance: Standard-error variance of ``mean`` (i.e.,
            ``Var(f(X)) / N`` for ``N`` samples).
        num_samples: Number of samples that produced the estimate
            (carried as static pytree aux-data, not as a leaf).
    """

    mean: jax.Array = struct.field()
    variance: jax.Array = struct.field()
    num_samples: int = struct.field(pytree_node=False)


def bayesian_monte_carlo(
    *,
    integrand: Callable[[jax.Array], jax.Array],
    samples: jax.Array,
) -> IntegralEstimate:
    """Integrate ``integrand`` against the measure that produced ``samples``.

    Args:
        integrand: Maps a single sample ``x`` (shape ``(d,)``) to a
            scalar ``f(x)``.
        samples: Pre-drawn samples of shape ``(num_samples, d)`` from
            the target measure ``π``. ``num_samples`` must be at least
            ``2`` to yield a non-trivial standard-error estimate.

    Returns:
        :class:`IntegralEstimate` with the sample mean and the
        standard-error variance ``Var(f(X)) / N``.

    Raises:
        ValueError: If fewer than two samples are provided.
    """
    num_samples = samples.shape[0]
    if num_samples < 2:
        raise ValueError(f"bayesian_monte_carlo requires at least 2 samples; got {num_samples}.")
    integrand_values = jax.vmap(integrand)(samples)
    mean = jnp.mean(integrand_values)
    variance = jnp.var(integrand_values, ddof=1) / num_samples
    return IntegralEstimate(mean=mean, variance=variance, num_samples=num_samples)

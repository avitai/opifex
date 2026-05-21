r"""Integration measures for Bayesian quadrature.

The two canonical measures are the (diagonal-covariance) Gaussian and
the uniform Lebesgue measure over a hyperrectangular domain. Both
expose a ``sample(num_samples, key)`` method so callers can plug them
into :func:`bayesian_monte_carlo`.

Canonical reference:
* ``../emukit/emukit/quadrature/measures/gaussian_measure.py``
  ``GaussianMeasure``.
* ``../emukit/emukit/quadrature/measures/lebesgue_measure.py``
  ``LebesgueMeasure``.

References
----------
* Briol, F.-X. et al. 2019 — *Probabilistic Integration*,
  Statistical Science 34(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True, kw_only=True)
class GaussianMeasure:
    r"""Diagonal-covariance Gaussian integration measure.

    Density:

    .. math::

        p(x) = (2\pi)^{-d/2} \prod_i \sigma_i^{-1}
               \exp\Bigl(-\tfrac12 \sum_i (x_i - \mu_i)^2 / \sigma_i^2\Bigr).

    Attributes:
        mean: Per-dimension mean ``μ`` of shape ``(d,)``.
        variance: Per-dimension diagonal variance ``σ²`` of shape
            ``(d,)``. Must be entry-wise strictly positive.
    """

    mean: jax.Array
    variance: jax.Array

    def __post_init__(self) -> None:
        """Validate shape and positivity."""
        if self.mean.shape != self.variance.shape:
            raise ValueError(
                "GaussianMeasure: mean and variance must have the same shape; "
                f"got {self.mean.shape!r} and {self.variance.shape!r}."
            )
        if not bool(jnp.all(self.variance > 0.0)):
            raise ValueError(
                "GaussianMeasure: diagonal variance must be positive entry-wise; "
                f"got {self.variance!r}."
            )

    @property
    def input_dim(self) -> int:
        """Dimensionality of the support."""
        return int(self.mean.shape[0])

    def sample(self, *, num_samples: int, key: jax.Array) -> jax.Array:
        r"""Draw ``num_samples`` IID samples from ``N(μ, diag(σ²))``."""
        standard = jax.random.normal(key, (num_samples, self.input_dim))
        return self.mean + standard * jnp.sqrt(self.variance)


@dataclass(frozen=True, slots=True, kw_only=True)
class LebesgueMeasure:
    r"""Uniform Lebesgue measure over a hyperrectangle.

    Attributes:
        lower: Per-dimension lower bound of shape ``(d,)``.
        upper: Per-dimension upper bound of shape ``(d,)``. Must be
            entry-wise strictly greater than ``lower``.
    """

    lower: jax.Array
    upper: jax.Array

    def __post_init__(self) -> None:
        """Validate shape and ordering of bounds."""
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                "LebesgueMeasure: lower and upper must have the same shape; "
                f"got {self.lower.shape!r} and {self.upper.shape!r}."
            )
        if not bool(jnp.all(self.upper > self.lower)):
            raise ValueError(
                "LebesgueMeasure: lower bound must be strictly less than upper "
                f"bound entry-wise; got lower={self.lower!r}, upper={self.upper!r}."
            )

    @property
    def input_dim(self) -> int:
        """Dimensionality of the support."""
        return int(self.lower.shape[0])

    @property
    def volume(self) -> jax.Array:
        r"""Volume ``∏_i (upper_i - lower_i)`` of the hyperrectangle."""
        return jnp.prod(self.upper - self.lower)

    def sample(self, *, num_samples: int, key: jax.Array) -> jax.Array:
        r"""Draw ``num_samples`` IID uniform samples on ``[lower, upper]``."""
        uniform_unit = jax.random.uniform(key, (num_samples, self.input_dim))
        return self.lower + uniform_unit * (self.upper - self.lower)

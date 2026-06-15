"""Diagonal-Gaussian prior log-density.

Same ``(prior_mean, prior_std)`` parameterization as
:func:`opifex.uncertainty.kernels.bayesian.diagonal_gaussian_kl` so the same
posterior parameters can be plugged into either helper without translation.

:class:`PriorSpec` is a frozen+slotted hashable capability declaration.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp


def diagonal_gaussian_log_prior(
    theta: jax.Array,
    *,
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
) -> jax.Array:
    """Log-density of ``theta`` under ``N(prior_mean, prior_std^2)`` summed over features.

    Returns a scalar (the total log-prior over every feature in ``theta``).

    Parameterization is shared with
    :func:`opifex.uncertainty.kernels.bayesian.diagonal_gaussian_kl`: both
    accept the prior as ``(prior_mean, prior_std)`` so the same posterior
    parameters plug into either helper without translation.

    Numerical formulation follows the standard closed-form Gaussian
    log-density (see Bishop 2006, "Pattern Recognition and Machine
    Learning", Eq. 2.43). Artifex's distribution package exposes a
    ``Distribution.log_prob`` method on its ``Normal`` class but no
    standalone pure-JAX ``gaussian_log_density`` function; constructing a
    full ``nnx.Module``-backed ``Normal`` per call is too heavy for a
    pure-JAX kernel, so the formula lives here. If Artifex adds a
    standalone helper later, this implementation collapses to a
    thin-wrapper delegation.
    """
    if prior_std <= 0.0:
        raise ValueError(f"prior_std must be > 0; got {prior_std!r}.")
    var = prior_std * prior_std
    log_density_per_element = -0.5 * (
        jnp.log(2 * jnp.pi) + jnp.log(var) + (theta - prior_mean) ** 2 / var
    )
    return jnp.sum(log_density_per_element)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PriorSpec:
    """Capability declaration for a prior family.

    Frozen, slotted, hashable; sequence fields are tuples.
    """

    name: str
    family: str
    parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate ``parameter_names`` shape and a non-empty ``name``."""
        if not isinstance(self.parameter_names, tuple):
            raise TypeError("parameter_names must be a tuple.")
        if not self.name:
            raise ValueError("PriorSpec.name must be non-empty.")


__all__ = ["PriorSpec", "diagonal_gaussian_log_prior"]

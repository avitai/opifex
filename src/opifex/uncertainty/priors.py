"""Phase 1 Task 1.6 — diagonal-Gaussian prior log-density.

Same ``(prior_mean, prior_std)`` parameterization as
:func:`opifex.uncertainty.kernels.bayesian.diagonal_gaussian_kl` so the same
posterior parameters can be plugged into either helper without translation.

Container pattern: :class:`PriorSpec` is pattern (A) per GUIDE_ALIGNMENT §5a.
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
    """Pattern (A) capability declaration for a prior family.

    Sequence fields are tuples (GUIDE_ALIGNMENT item 22a).
    """

    name: str
    family: str
    parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_names, tuple):
            raise TypeError("parameter_names must be a tuple (GUIDE_ALIGNMENT item 22a).")
        if not self.name:
            raise ValueError("PriorSpec.name must be non-empty.")


__all__ = ["PriorSpec", "diagonal_gaussian_log_prior"]

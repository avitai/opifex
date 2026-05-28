"""Sobol global-sensitivity indices via the Saltelli (2002) pick-freeze scheme.

The estimator returns first-order indices ``S_i`` and total-order
indices ``T_i`` for a scalar-valued model ``f: R^d -> R`` evaluated
under independent uniform inputs over the box ``[lower, upper]``.

Reference: Saltelli, A. (2002), "Making best use of model evaluations
to compute sensitivity indices", Computer Physics Communications
145, pp. 280–297. The two-matrix design with ``AB_i`` (column ``i``
of ``A`` replaced by column ``i`` of ``B``) matches the
``saltelli.sample`` layout used in SALib's reference NumPy
implementation (not imported here — Task 6.4 forbids it as a
dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True, kw_only=True)
class SobolResult:
    """Container for first-order + total-order Sobol indices.

    Attributes:
        first_order: ``S_i = V[E[Y|X_i]] / V[Y]``, shape ``(d,)``.
        total_order: ``T_i = E[V[Y|X_{~i}]] / V[Y]``, shape ``(d,)``.
        variance: ``V[Y]`` estimate over the combined sample.
        num_samples: Base sample count ``N`` (total ``f`` evaluations =
            ``N * (d + 2)``).
    """

    first_order: jax.Array
    total_order: jax.Array
    variance: jax.Array
    num_samples: int


def sobol_indices(
    model: Callable[[jax.Array], jax.Array],
    *,
    num_samples: int,
    lower: jax.Array,
    upper: jax.Array,
    rng_key: jax.Array,
) -> SobolResult:
    """Estimate first- + total-order Sobol indices via Saltelli (2002).

    Args:
        model: Scalar-valued model ``f(x)`` accepting input arrays of
            shape ``(..., d)`` and returning shape ``(...,)``. The
            function must be JAX-traceable.
        num_samples: Base sample count ``N`` (total evaluations
            ``= N * (d + 2)``).
        lower: Lower box bounds, shape ``(d,)``.
        upper: Upper box bounds, shape ``(d,)``.
        rng_key: Caller-owned JAX PRNG key. Two independent ``N x d``
            sample matrices are drawn from ``rng_key``.

    Returns:
        A :class:`SobolResult` with first-order, total-order, and
        variance estimates.

    Raises:
        ValueError: If ``lower`` and ``upper`` have different shapes,
            or ``num_samples <= 0``.
    """
    if lower.shape != upper.shape:
        raise ValueError(f"lower and upper must share shape; got {lower.shape} vs {upper.shape}.")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples}.")

    dim = lower.shape[0]
    key_a, key_b = jax.random.split(rng_key)

    # Uniform [lower, upper] sample matrices.
    scale = upper - lower
    a = jax.random.uniform(key_a, (num_samples, dim)) * scale + lower
    b = jax.random.uniform(key_b, (num_samples, dim)) * scale + lower

    y_a = model(a)
    y_b = model(b)

    # Build the AB_i evaluations by replacing column i of A with B's.
    # ``one_hot`` produces a (d, d) selector matrix so we can mask each
    # column without indexing into a traced integer — this keeps the
    # estimator clean under ``jax.jit``.
    selectors = jnp.eye(dim, dtype=a.dtype)  # (d, d)

    def _eval_swapped(selector: jax.Array) -> jax.Array:
        ab = a * (1.0 - selector) + b * selector
        return model(ab)

    y_ab = jax.vmap(_eval_swapped)(selectors)  # shape (d, N)

    variance = jnp.var(jnp.concatenate([y_a, y_b]))

    # Jansen (1999) / Saltelli (2010) low-bias estimators.
    # S_i  = (1/N) Σ y_b * (y_ab_i - y_a) / V[Y]
    # T_i  = (1/(2N)) Σ (y_a - y_ab_i)^2  / V[Y]
    safe_variance = jnp.where(variance > 0, variance, 1.0)
    first_order = jnp.mean(y_b[None, :] * (y_ab - y_a[None, :]), axis=1) / safe_variance
    total_order = jnp.mean((y_a[None, :] - y_ab) ** 2, axis=1) / (2.0 * safe_variance)
    # Clip total-order to non-negative (Monte Carlo can drift slightly below 0).
    total_order = jnp.clip(total_order, 0.0)

    return SobolResult(
        first_order=first_order,
        total_order=total_order,
        variance=variance,
        num_samples=num_samples,
    )


__all__ = ["SobolResult", "sobol_indices"]

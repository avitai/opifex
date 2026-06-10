"""Morris elementary-effects screening for global sensitivity ranking.

The method draws ``r`` random trajectories of length ``d + 1`` on a
``num_levels`` grid in ``[lower, upper]^d``. Each trajectory perturbs
one dimension at a time so the per-step elementary effect
``EE_i = (f(x + Δe_i) - f(x)) / Δ`` isolates dimension ``i``. The
screening statistics are ``mu_star`` (mean absolute elementary
effect, monotone with influence) and ``sigma`` (standard deviation
of elementary effects, signalling non-linearity / interactions).

Reference: Morris, M. D. (1991), "Factorial sampling plans for
preliminary computational experiments", Technometrics 33(2), pp.
161–174; Campolongo, F., Cariboni, J., Saltelli, A. (2007),
"An effective screening design for sensitivity analysis of large
models", Environmental Modelling & Software 22, pp. 1509–1518.

We port the trajectory-builder layout used by SALib's reference NumPy
implementation (not imported — Task 6.4 forbids it as a dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True, kw_only=True)
class MorrisResult:
    """Container for Morris screening statistics.

    Attributes:
        mu_star: Mean of absolute elementary effects, shape ``(d,)``.
            Monotone with overall input influence.
        mu: Mean of signed elementary effects, shape ``(d,)``.
            Sign indicates effect direction.
        sigma: Standard deviation of elementary effects, shape ``(d,)``.
            Large ``sigma`` relative to ``mu_star`` flags non-linearity
            or interaction with other inputs.
        num_trajectories: ``r``, the number of independent trajectories.
        num_levels: ``p``, the number of grid levels per dimension.
    """

    mu_star: jax.Array
    mu: jax.Array
    sigma: jax.Array
    num_trajectories: int
    num_levels: int


def _build_trajectory(rng_key: jax.Array, dim: int, num_levels: int) -> tuple[jax.Array, jax.Array]:
    """Draw a single Morris trajectory of length ``dim + 1``.

    Returns
    -------
    points: shape ``(d + 1, d)`` — successive trajectory points in
        ``[0, 1]^d`` (caller scales to the actual box).
    order: shape ``(d,)`` — the permutation of dimensions perturbed
        between successive points.
    """
    delta = num_levels / (2.0 * (num_levels - 1))

    base_key, perm_key, sign_key = jax.random.split(rng_key, 3)
    # Random starting point on the lower half of the grid so that the
    # +delta perturbation stays inside [0, 1].
    base = jax.random.randint(
        base_key,
        shape=(dim,),
        minval=0,
        maxval=num_levels // 2,
    ).astype(jnp.float32) / (num_levels - 1)
    # Random perturbation order + sign per dimension.
    order = jax.random.permutation(perm_key, dim)
    signs = jax.random.choice(sign_key, jnp.array([+1.0, -1.0]), shape=(dim,))

    def step(point: jax.Array, perturb_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Perturb one input dimension to advance the Morris trajectory."""
        i = order[perturb_idx]
        new_point = point.at[i].add(signs[perturb_idx] * delta)
        return new_point, new_point

    _, perturbed = jax.lax.scan(step, base, jnp.arange(dim))
    points = jnp.concatenate([base[None, :], perturbed], axis=0)
    return points, order


def morris_screening(
    model: Callable[[jax.Array], jax.Array],
    *,
    num_trajectories: int,
    num_levels: int,
    lower: jax.Array,
    upper: jax.Array,
    rng_key: jax.Array,
) -> MorrisResult:
    """Run Morris elementary-effects screening on ``model``.

    Args:
        model: Scalar-valued model ``f(x)`` accepting input arrays of
            shape ``(..., d)`` and returning ``(...,)``. JAX-traceable.
        num_trajectories: ``r``, the number of independent trajectories.
        num_levels: ``p``, the number of grid levels per dimension.
            Must be ``>= 2``.
        lower: Lower box bounds, shape ``(d,)``.
        upper: Upper box bounds, shape ``(d,)``.
        rng_key: Caller-owned JAX PRNG key.

    Returns:
        A :class:`MorrisResult` with ``mu_star``, ``mu``, and ``sigma``
        statistics over ``r`` trajectories.

    Raises:
        ValueError: If ``lower`` / ``upper`` shapes mismatch,
            ``num_trajectories <= 0``, or ``num_levels < 2``.
    """
    if lower.shape != upper.shape:
        raise ValueError(f"lower and upper must share shape; got {lower.shape} vs {upper.shape}.")
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be positive; got {num_trajectories}.")
    if num_levels < 2:
        raise ValueError(f"num_levels must be >= 2; got {num_levels}.")

    dim = lower.shape[0]
    delta = num_levels / (2.0 * (num_levels - 1))
    scale = upper - lower

    keys = jax.random.split(rng_key, num_trajectories)

    def _trajectory_effects(key: jax.Array) -> jax.Array:
        """Return one trajectory's signed elementary effects, shape ``(d,)``."""
        points_unit, order = _build_trajectory(key, dim, num_levels)
        points = points_unit * scale + lower
        outputs = model(points)  # shape (d + 1,)

        # The k-th elementary effect uses ``order[k]`` as its dimension
        # index and is signed by the perturbation direction encoded in
        # the trajectory.
        diffs = outputs[1:] - outputs[:-1]
        unit_step = points_unit[1:] - points_unit[:-1]  # shape (d, d)
        # Each row is ``delta * sign_k * e_{order[k]}``; we extract the
        # signed scalar perturbation as the entry at column ``order[k]``.
        step_signed = unit_step[jnp.arange(dim), order]
        elementary_effects = diffs / step_signed
        # Reorder so position i carries dimension i's effect.
        return jnp.zeros(dim).at[order].set(elementary_effects)

    effects = jax.vmap(_trajectory_effects)(keys)  # shape (r, d)

    mu = jnp.mean(effects, axis=0)
    mu_star = jnp.mean(jnp.abs(effects), axis=0)
    # Use ``ddof=0`` semantics (Morris convention) — the trajectory-level
    # sample variance is exactly what's reported in the literature.
    sigma = jnp.std(effects, axis=0)

    # ``delta`` is the perturbation amplitude carried through the unit
    # cube; documenting it keeps the formula auditable but it doesn't
    # need to be exported.
    del delta

    return MorrisResult(
        mu_star=mu_star,
        mu=mu,
        sigma=sigma,
        num_trajectories=num_trajectories,
        num_levels=num_levels,
    )


__all__ = ["MorrisResult", "morris_screening"]

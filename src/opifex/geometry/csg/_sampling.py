"""Rejection sampling shared by the CSG shapes, as DeepXDE's samplers do it.

DeepXDE's ``random_points`` and ``random_boundary_points`` (``deepxde/geometry/csg.py``) draw a
round of candidates, keep the ones the shape accepts, and repeat until ``n`` points are held.
The rounds here are drawn from successive splits of one key, eagerly, because the number of
points a round keeps depends on the draw.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float


def draw_until(
    n: int, key: jax.Array, draw_round: Callable[[jax.Array], Float[jax.Array, "m d"]]
) -> Float[jax.Array, "n d"]:
    """Collect the points ``draw_round`` keeps from successive keys until there are ``n``.

    Args:
        n: Number of points to return.
        key: Key the rounds' keys are split from.
        draw_round: Draws one round of candidates from a key and returns those it keeps.

    Returns:
        The first ``n`` points kept, in the order they were drawn.

    Raises:
        ValueError: If ``n`` is below one.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    kept: list[jax.Array] = []
    count = 0
    while count < n:
        key, round_key = jax.random.split(key)
        accepted = draw_round(round_key)
        kept.append(accepted)
        count += int(accepted.shape[0])
    return jnp.concatenate(kept)[:n]

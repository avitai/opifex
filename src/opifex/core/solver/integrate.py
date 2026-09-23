"""The shared time-stepping driver.

Every time-stepping method takes a fixed number of equal steps and saves the state on a
static schedule. The count is static so the trajectory is a ``lax.scan``, which
differentiates in reverse mode; a ``lax.while_loop`` stepping to a condition has no
transpose rule. Gradients are therefore discretise-then-optimise: they differentiate the
steps actually taken, which Gholami, Keutzer and Biros (2019, IJCAI, 730) show the
continuous adjoint need not match.

Saving is a nested scan -- an outer scan over the save intervals, an inner one over the
steps inside each -- so only the saved states are stacked, never the full trajectory.
Each step is rematerialised by default: the backward pass recomputes a step's internals
instead of storing them, so memory grows with the number of steps and not with the work
done inside each (Griewank and Walther 2008, *Evaluating Derivatives*, ch. 12).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True, kw_only=True)
class Snapshots:
    """Which states a trajectory saves.

    Attributes:
        count: Number of evenly spaced saves; the last is the final state.
        include_initial: Whether the initial state is prepended to the saves.
    """

    count: int = 1
    include_initial: bool = False

    def __post_init__(self) -> None:
        """Refuse a schedule that saves nothing."""
        if self.count < 1:
            msg = f"Snapshots.count must be at least one, got {self.count}"
            raise ValueError(msg)


FINAL_STATE: Final = Snapshots()
"""Save only the state after the last step."""


def integrate[S](
    step: Callable[[S, jax.Array], S],
    initial: S,
    *,
    num_steps: int,
    save: Snapshots = FINAL_STATE,
    checkpoint: bool = True,
) -> S:
    """Apply ``step`` ``num_steps`` times and return the saved states.

    Args:
        step: Maps a state and the global step index, an ``int32`` scalar counting from
            zero, to the next state. Time-dependent schemes read their time from the index.
        initial: The initial state; any pytree.
        num_steps: Total number of steps. Static.
        save: The save schedule; its count must divide ``num_steps``.
        checkpoint: Rematerialise each step in the backward pass.

    Returns:
        The saved states, each leaf stacked along a new leading axis of length
        ``save.count``, plus one when the initial state is included.

    Raises:
        ValueError: If ``num_steps`` is below one or not divisible by ``save.count``.
    """
    if num_steps < 1:
        msg = f"num_steps must be at least one, got {num_steps}"
        raise ValueError(msg)
    if num_steps % save.count:
        msg = f"num_steps ({num_steps}) must be divisible by the save count ({save.count})"
        raise ValueError(msg)
    steps_per_save = num_steps // save.count
    advance = jax.checkpoint(step) if checkpoint else step

    def run(state: S, first_index: jax.Array | int) -> S:
        def one_step(inner: S, offset: jax.Array) -> tuple[S, None]:
            return advance(inner, first_index + offset), None

        end, _ = jax.lax.scan(one_step, state, jnp.arange(steps_per_save, dtype=jnp.int32))
        return end

    if save.count == 1:
        # A single save needs no outer loop; nesting one of length one costs about 2% on
        # an FNO rollout, measured on an RTX 4090.
        saved = jax.tree.map(lambda leaf: jnp.asarray(leaf)[None], run(initial, 0))
    else:

        def interval(state: S, first_index: jax.Array) -> tuple[S, S]:
            end = run(state, first_index)
            return end, end

        first_indices = jnp.arange(save.count, dtype=jnp.int32) * steps_per_save
        _, saved = jax.lax.scan(interval, initial, first_indices)
    if not save.include_initial:
        return saved
    return jax.tree.map(
        lambda start, rest: jnp.concatenate([jnp.asarray(start)[None], rest]), initial, saved
    )


__all__ = ["FINAL_STATE", "Snapshots", "integrate"]

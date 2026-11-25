r"""Checkpointed bounded while-loop primitive for batched adaptive JAX solvers.

Provides :func:`unvmap_any` and :func:`bounded_while_loop` so adaptive solvers can exit
early as soon as all active batch lanes terminate while remaining differentiable in both
forward (:func:`jax.jvp`, :func:`jax.jacfwd`) and reverse (:func:`jax.grad`,
:func:`jax.jacrev`) modes with logarithmic :math:`O(b \log_b N)` activation memory.

References:
    * Griewank, A., & Walther, A. (2000). "Algorithm 799: Revolve: An implementation of
      checkpointing for the reverse or adjoint mode of computational differentiation."
      *ACM Trans. Math. Softw.* 26(1), 19-45.
    * Griewank, A., & Walther, A. (2008). *Evaluating Derivatives: Principles and
      Techniques of Algorithmic Differentiation* (2nd ed.). SIAM, §12.2 (base-:math:`b`
      hierarchical checkpointing).
"""

from __future__ import annotations

import math
from collections.abc import Callable  # noqa: TC003
from typing import TypeVar

import jax
import jax.numpy as jnp


CarryT = TypeVar("CarryT")


@jax.custom_batching.custom_vmap
def unvmap_any(pred: jax.Array) -> jax.Array:
    """OR-reduce a boolean predicate across all enclosing :func:`jax.vmap` batch axes.

    Args:
        pred: Boolean array (scalar or arbitrarily shaped).

    Returns:
        Unbatched scalar boolean ``jax.Array`` equal to ``True`` when any lane is ``True``.
    """
    return jnp.any(pred)


def _unvmap_any_vmap(
    axis_size: int,
    in_batched: tuple[bool],
    pred: jax.Array,
) -> tuple[jax.Array, bool]:
    """Recursively OR-reduce ``pred`` across nested ``vmap`` batch axes."""
    del axis_size, in_batched
    return unvmap_any(jnp.any(pred)), False


unvmap_any.def_vmap(_unvmap_any_vmap)


def _run_hierarchical_bounded_loop(
    active_pred: Callable[[tuple[CarryT, jax.Array]], jax.Array],
    counted_body: Callable[[tuple[CarryT, jax.Array]], tuple[CarryT, jax.Array]],
    state: tuple[CarryT, jax.Array],
    steps: int,
    base: int,
) -> tuple[CarryT, jax.Array]:
    """Execute a ``steps``-bounded recursive ``scan`` + ``cond`` + ``checkpoint`` tree."""
    if steps == 1:
        return counted_body(state)

    def call(s: tuple[CarryT, jax.Array]) -> tuple[CarryT, jax.Array]:
        return _run_hierarchical_bounded_loop(active_pred, counted_body, s, steps // base, base)

    def scan_step(s: tuple[CarryT, jax.Array], _: None) -> tuple[tuple[CarryT, jax.Array], None]:
        return jax.lax.cond(active_pred(s), call, lambda x: x, s), None

    if steps != base:
        scan_step = jax.checkpoint(scan_step, prevent_cse=False)
    return jax.lax.scan(scan_step, state, xs=None, length=base)[0]


def bounded_while_loop(
    cond_fn: Callable[[CarryT], jax.Array],
    body_fn: Callable[[CarryT], CarryT],
    init_val: CarryT,
    *,
    max_steps: int,
    base: int = 16,
) -> CarryT:
    r"""Run ``body_fn`` while ``cond_fn`` holds, up to ``max_steps``, with checkpointed AD.

    Organises ``max_steps`` into a base-``base`` tree of :func:`jax.lax.scan` blocks
    guarded by :func:`jax.lax.cond` over :func:`unvmap_any`, applying
    :func:`jax.checkpoint` above the leaf level so reverse-mode differentiation stores at
    most :math:`O(\text{base} \cdot \lceil \log_{\text{base}}(\text{max\_steps}) \rceil)`
    carries while forward execution skips remaining blocks once every lane has finished.

    Args:
        cond_fn: Predicate ``val -> bool_array``; under ``vmap``, the loop continues while
            any batch lane reports ``True`` and fewer than ``max_steps`` steps have run.
        body_fn: Single-step transition ``val -> next_val`` preserving pytree structure
            and dtypes; when ``vmap`` mixes active and finished lanes, ``body_fn`` must
            leave finished lanes unchanged.
        init_val: Initial loop carry pytree.
        max_steps: Static upper bound ``>= 0`` on the number of ``body_fn`` evaluations.
        base: Static radix ``>= 2`` of the hierarchical ``scan`` tree.

    Returns:
        Final loop carry pytree matching ``init_val``.

    Raises:
        ValueError: When ``max_steps < 0`` or ``base < 2``.
    """
    if max_steps < 0:
        raise ValueError(f"max_steps must be non-negative; got {max_steps}.")
    if base < 2:
        raise ValueError(f"base must be at least 2; got {base}.")
    if max_steps == 0:
        return init_val
    if max_steps == 1:
        return jax.lax.cond(unvmap_any(cond_fn(init_val)), body_fn, lambda x: x, init_val)

    rounded = base ** math.ceil(math.log(max_steps, base))

    def active_pred(state: tuple[CarryT, jax.Array]) -> jax.Array:
        val, step_idx = state
        return (step_idx < max_steps) & unvmap_any(cond_fn(val))

    def counted_body(state: tuple[CarryT, jax.Array]) -> tuple[CarryT, jax.Array]:
        val, step_idx = state
        return body_fn(val), step_idx + jnp.int32(1)

    final_val, _ = _run_hierarchical_bounded_loop(
        active_pred, counted_body, (init_val, jnp.int32(0)), rounded, base
    )
    return final_val


__all__ = ["bounded_while_loop", "unvmap_any"]

"""The optimiser interface shared by hand-designed and learned optimisers.

Mirrors ``learned_optimization/optimizers/base.py``: an :class:`Optimizer` threads an opaque
state through ``init`` / ``update`` / ``get_params``. Both an optax-wrapped hand-designed
optimiser (:class:`OptaxOptimizer`) and a meta-learned optimiser fit this interface, so they
are interchangeable at every call site. ``init`` accepts ``num_steps`` so horizon-aware
optimisers (e.g. learned optimisers conditioning on training fraction) know the unroll length;
``update`` accepts the current ``loss`` because learned optimisers may consume it.
"""

from __future__ import annotations

import abc
from typing import Generic, TYPE_CHECKING, TypeVar

import jax
import jax.numpy as jnp
import optax
from flax import struct


if TYPE_CHECKING:
    from jaxtyping import PyTree


class OptaxOptState(struct.PyTreeNode):
    """Pytree state for :class:`OptaxOptimizer` (params + optax state + step count)."""

    params: PyTree
    opt_state: optax.OptState
    iteration: jax.Array


# Optimiser-state type, specialised per concrete optimiser (an ``flax.struct`` pytree node).
StateT = TypeVar("StateT", bound=struct.PyTreeNode)


class Optimizer(abc.ABC, Generic[StateT]):
    """Stateful optimiser interface (object-aware, not an ``optax.GradientTransformation``).

    The state is opaque to callers; use :meth:`get_params` to read the current optimisee
    parameters. ``num_steps`` (at :meth:`init`) is the planned unroll length, surfaced for
    horizon-aware optimisers; ``loss`` (at :meth:`update`) is the current objective value,
    consumed by learned optimisers and ignored by hand-designed ones.
    """

    @abc.abstractmethod
    def init(
        self,
        params: PyTree,
        *,
        num_steps: int | None = None,
        key: jax.Array | None = None,
    ) -> StateT:
        """Initialise optimiser state for the given starting ``params``."""

    @abc.abstractmethod
    def update(
        self,
        state: StateT,
        grad: PyTree,
        *,
        loss: jax.Array | None = None,
    ) -> StateT:
        """Apply one optimiser step, returning the new state."""

    @abc.abstractmethod
    def get_params(self, state: StateT) -> PyTree:
        """Read the current optimisee parameters from ``state``."""


class OptaxOptimizer(Optimizer[OptaxOptState]):
    """Wraps any ``optax.GradientTransformation`` as an :class:`Optimizer`.

    This is the hand-designed baseline family (SGD/Adam/...). ``num_steps``/``key`` are
    accepted for interface uniformity and ignored; ``loss`` is ignored (optax updates depend
    only on the gradient and optimiser state).
    """

    def __init__(self, transformation: optax.GradientTransformation) -> None:
        """Store the wrapped optax transformation."""
        self.transformation = transformation

    def init(
        self,
        params: PyTree,
        *,
        num_steps: int | None = None,  # noqa: ARG002 - uniform Optimizer interface
        key: jax.Array | None = None,  # noqa: ARG002 - uniform Optimizer interface
    ) -> OptaxOptState:
        """Initialise optax state around ``params``."""
        return OptaxOptState(
            params=params,
            opt_state=self.transformation.init(params),
            iteration=jnp.zeros((), dtype=jnp.int32),
        )

    def update(
        self,
        state: OptaxOptState,
        grad: PyTree,
        *,
        loss: jax.Array | None = None,  # noqa: ARG002 - optax ignores loss
    ) -> OptaxOptState:
        """Apply the optax update and advance the iteration counter."""
        updates, new_opt_state = self.transformation.update(grad, state.opt_state, state.params)
        return OptaxOptState(
            params=optax.apply_updates(state.params, updates),
            opt_state=new_opt_state,
            iteration=state.iteration + 1,
        )

    def get_params(self, state: OptaxOptState) -> PyTree:
        """Return the current parameters."""
        return state.params


__all__ = ["OptaxOptState", "OptaxOptimizer", "Optimizer"]

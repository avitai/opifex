r"""Structural typing contracts for the VMC stack.

These ``Protocol`` definitions decouple the training loop from concrete
implementations so that, for example, a PsiFormer attention ansatz, a MALA
sampler, or a K-FAC optimiser can swap in without touching the driver (the
``../deepqmc`` ``Protocol`` style). They document the minimal surface each
component must expose; concrete classes need not subclass them.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float  # noqa: TC002


@runtime_checkable
class Wavefunction(Protocol):
    """A neural wavefunction evaluated one walker at a time.

    Calling the wavefunction with a single walker's positions of shape
    ``(nelectron, ndim)`` must return a ``(sign, log|psi|)`` tuple of scalars,
    evaluated in the log domain for numerical stability.
    """

    def __call__(self, positions: Float[Array, "nelectron ndim"]) -> tuple[Array, Array]:
        """Return ``(sign, log|psi|)`` for a single walker."""
        ...


@runtime_checkable
class Sampler(Protocol):
    """A Metropolis sampler of the Born density ``|psi|^2``."""

    @property
    def steps(self) -> int:
        """Number of MCMC sweeps performed per :meth:`sample` call."""
        ...

    def sample(
        self,
        log_abs: Callable[[Array], Array],
        walkers: Float[Array, "batch nelectron ndim"],
        key: Array,
    ) -> tuple[Array, Array]:
        """Advance ``walkers`` and return ``(new_walkers, acceptance_fraction)``."""
        ...


__all__ = ["Sampler", "Wavefunction"]

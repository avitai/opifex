r"""Energy property head: per-atom scalar energies summed to a total energy.

The :class:`EnergyHead` reads the backbone's per-atom scalar (``l = 0``)
embeddings, maps each to a scalar atomic energy with a small invariant MLP, and
sums them into the total potential energy

.. math:: E = \sum_i \varepsilon_i .

Because the summand is a per-atom invariant and summation is permutation
invariant, the total energy is an E(3)- and permutation-invariant scalar -- the
defining contract of an interatomic potential (Schütt et al. 2018, SchNet;
Batzner et al. 2022, NequIP). Per-element reference energies (``E0``) and the
output normaliser are supplied by an optional
:class:`~opifex.neural.atomistic.scale_shift.AtomicScaleShift` (the MACE
``ScaleShiftBlock`` affine readout); when omitted the head is the bare
sum-of-atomic-energies readout that the conservative force/stress heads
differentiate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array  # noqa: TC002

# Eager (not TYPE_CHECKING): the ``nnx.Data[AtomicScaleShift | None]`` class
# annotation below is resolved by Flax NNX at runtime, so the name must exist.
from opifex.neural.atomistic.scale_shift import AtomicScaleShift  # noqa: TC001


if TYPE_CHECKING:
    from opifex.core.quantum.molecular_system import MolecularSystem


class EnergyHead(nnx.Module):
    """Sum-of-atomic-energies readout producing a scalar total energy.

    Args:
        feature_dim: Width of the backbone's ``"node_features"`` embedding.
        hidden_dim: Hidden width of the per-atom MLP. Defaults to ``feature_dim``.
        scale_shift: Optional per-atom energy scale-shift (``E0`` + normaliser)
            applied to the summed energy. ``None`` (default) leaves the raw
            sum-of-atomic-energies unchanged.
        rngs: Random number generators (keyword-only) seeding the MLP weights.
    """

    # The scale-shift carries (frozen) data arrays, so it is an nnx data leaf
    # (it travels with the model state under split/merge), not a static attr.
    scale_shift: nnx.Data[AtomicScaleShift | None]

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int | None = None,
        scale_shift: AtomicScaleShift | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the per-atom energy MLP and store the optional scale-shift."""
        super().__init__()
        width = hidden_dim if hidden_dim is not None else feature_dim
        self.hidden = nnx.Linear(feature_dim, width, rngs=rngs)
        self.readout = nnx.Linear(width, 1, rngs=rngs)
        self.scale_shift = scale_shift

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """This head emits the total ``"energy"``."""
        return ("energy",)

    def __call__(
        self,
        system: MolecularSystem,
        graph: tuple[Array, Array],
        embeddings: dict[str, Array],
    ) -> dict[str, Array]:
        """Map per-atom embeddings to atomic energies and sum to the total.

        Args:
            system: The molecular system (used only for its atom count contract).
            graph: The ``(senders, receivers)`` edge index (unused by this head).
            embeddings: Must contain ``"node_features"`` of shape
                ``(n_atoms, feature_dim)``.

        Returns:
            ``{"energy": scalar}`` -- the summed total potential energy, with the
            optional scale-shift (``E0`` + normaliser) applied.
        """
        del graph
        node_features = embeddings["node_features"]
        atomic_energies = self.readout(nnx.silu(self.hidden(node_features)))
        energy = jnp.sum(atomic_energies)
        if self.scale_shift is not None:
            energy = self.scale_shift.apply(energy, system.n_atoms)
        return {"energy": energy}


__all__ = ["EnergyHead"]

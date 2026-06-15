r"""Conservative force head: forces as the negative position-gradient of energy.

The :class:`ForcesHead` implements the *conservative* force strategy

.. math:: \mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i},

computed by reverse-mode autodiff of the total energy with respect to atomic
positions (Schütt et al. 2018, SchNet; the ``../mace`` ``compute_forces`` pattern,
``mace/modules/utils.py``). Conservative forces are energy-consistent by
construction (they integrate to a conserved Hamiltonian in MD) -- this is the
default strategy.

A *direct* force head (an equivariant ``l = 1`` readout predicting forces
without differentiating the energy, as in Orb / fairchem) is a future variant
that plugs in through the same :class:`opifex.core.quantum.protocols.PropertyHead`
protocol.

To differentiate the energy, the assembled model injects a position-to-energy
closure into ``embeddings`` under the reserved key :data:`ENERGY_FN_KEY`; the head
is otherwise stateless.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Callable


ENERGY_FN_KEY = "_energy_fn"
"""Reserved ``embeddings`` key under which the model injects ``positions -> energy``."""


class ForcesHead:
    """Forces as ``-grad(energy)`` w.r.t. positions (the conservative strategy).

    Stateless: it differentiates the position-to-energy closure injected by the
    assembled model, so it owns no parameters.
    """

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """This head emits ``"forces"``."""
        return ("forces",)

    def __call__(
        self,
        system: MolecularSystem,
        graph: tuple[Array, Array],
        embeddings: dict[str, Array],
    ) -> dict[str, Array]:
        """Return ``{"forces": -dE/dR}`` of shape ``(n_atoms, 3)``.

        Args:
            system: The molecular system providing the differentiation point
                (its positions).
            graph: The ``(senders, receivers)`` edge index (unused by this head).
            embeddings: Must contain :data:`ENERGY_FN_KEY` -- a callable mapping
                positions to the scalar total energy.

        Returns:
            ``{"forces": Array}`` with the conservative forces.

        Raises:
            KeyError: If the energy closure was not injected by the model.
        """
        del graph
        if ENERGY_FN_KEY not in embeddings:
            raise KeyError(
                f"ForcesHead requires the position->energy closure under "
                f"{ENERGY_FN_KEY!r}; the assembled AtomisticModel must inject it."
            )
        energy_fn: Callable[[Array], Array] = embeddings[ENERGY_FN_KEY]  # type: ignore[assignment]
        forces = -jax.grad(energy_fn)(system.positions)
        return {"forces": forces}


__all__ = ["ENERGY_FN_KEY", "ForcesHead"]

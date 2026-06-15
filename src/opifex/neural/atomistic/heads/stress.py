r"""Stress head: virial via strain-displacement autodiff.

The :class:`StressHead` implements the *conservative* stress strategy used by
``../mace`` (``compute_forces_virials`` / ``get_symmetric_displacement`` in
``mace/modules/utils.py``, after the NequIP recipe). A symmetric infinitesimal
strain :math:`\varepsilon` is applied to positions and the cell,

.. math::
   \mathbf{r}_i \to \mathbf{r}_i + \varepsilon\,\mathbf{r}_i, \qquad
   \mathbf{h} \to \mathbf{h} + \mathbf{h}\,\varepsilon ,

and the virial is the energy gradient with respect to that strain evaluated at
:math:`\varepsilon = 0`:

.. math:: W = \left.\frac{\partial E}{\partial \varepsilon}\right|_{\varepsilon=0},
   \qquad \sigma = \frac{W}{\Omega},

with :math:`\Omega = |\det \mathbf{h}|` the cell volume. Because the strain is
symmetrised before use, the resulting virial / stress is a symmetric ``3x3``
tensor. This is the default conservative strategy; a direct stress head is a
future variant plugging into the same ``PropertyHead`` protocol.

The assembled model injects a strain-to-energy closure into ``embeddings`` under
the reserved key :data:`STRAIN_ENERGY_FN_KEY`; the head is otherwise stateless.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Callable


STRAIN_ENERGY_FN_KEY = "_strain_energy_fn"
"""Reserved ``embeddings`` key under which the model injects ``strain -> energy``."""


class StressHead:
    """Stress as the strain-derivative of energy divided by cell volume.

    Stateless: it differentiates the symmetric-strain-to-energy closure injected
    by the assembled model, so it owns no parameters.
    """

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """This head emits ``"stress"``."""
        return ("stress",)

    def __call__(
        self,
        system: MolecularSystem,
        graph: tuple[Array, Array],
        embeddings: dict[str, Array],
    ) -> dict[str, Array]:
        """Return ``{"stress": W / volume}`` -- the symmetric ``3x3`` stress.

        Args:
            system: The periodic molecular system; its ``cell`` sets the volume.
            graph: The ``(senders, receivers)`` edge index (unused by this head).
            embeddings: Must contain :data:`STRAIN_ENERGY_FN_KEY` -- a callable
                mapping a symmetric ``3x3`` strain to the scalar total energy.

        Returns:
            ``{"stress": Array}`` of shape ``(3, 3)``.

        Raises:
            KeyError: If the strain closure was not injected by the model.
            ValueError: If the system has no periodic cell to define a volume.
        """
        del graph
        if STRAIN_ENERGY_FN_KEY not in embeddings:
            raise KeyError(
                f"StressHead requires the strain->energy closure under "
                f"{STRAIN_ENERGY_FN_KEY!r}; the assembled AtomisticModel must inject it."
            )
        if system.cell is None:
            raise ValueError("StressHead requires a periodic cell to define the volume.")
        strain_energy_fn: Callable[[Array], Array] = embeddings[STRAIN_ENERGY_FN_KEY]  # type: ignore[assignment]
        zero_strain = jnp.zeros((3, 3), dtype=system.positions.dtype)
        virial = jax.grad(strain_energy_fn)(zero_strain)
        symmetric_virial = 0.5 * (virial + virial.T)
        volume = jnp.abs(jnp.linalg.det(system.cell))
        return {"stress": symmetric_virial / volume}


__all__ = ["STRAIN_ENERGY_FN_KEY", "StressHead"]

r"""ASE Calculator binding for any opifex :class:`AtomisticModel`.

:class:`OpifexCalculator` is the single external-infrastructure adapter that lets
an opifex interatomic potential drive Atomic Simulation Environment (ASE)
workflows -- molecular dynamics, geometry relaxation, phonons -- by exposing the
model through the ``ase.calculators.calculator.Calculator`` contract.

It follows the structure of the reference ``../mace`` calculator
(``mace/calculators/mace.py``, :class:`mace.calculators.MACECalculator`): subclass
:class:`ase.calculators.calculator.Calculator`, derive
:attr:`implemented_properties` from the wrapped model, and fill ``self.results``
inside :meth:`calculate` with host (NumPy) arrays in ASE's unit convention --
energies in eV, forces in eV/Ang, stress as a 6-component Voigt vector in
eV/Ang^3 (ASE calculator API; cf. :func:`ase.stress.full_3x3_to_voigt_6_stress`).

Unit bridge (the boundary's single responsibility). An opifex
:class:`~opifex.core.quantum.molecular_system.MolecularSystem` lives in atomic
units -- positions in Bohr, energies in Hartree -- whereas ``ase.Atoms`` lives in
Angstrom/eV. The calculator is the only place these two conventions meet, so it
converts ASE -> atomic units on the way in (positions/cell scaled by
:data:`ase.units.Bohr`) and atomic units -> ASE on the way out (energy * Hartree,
forces * Hartree / Bohr, stress * Hartree / Bohr^3).

The graph is built by the *model's* injected neighbour list
(:class:`~opifex.core.quantum.protocols.RadiusNeighborList`, which delegates to
:func:`opifex.neural.equivariant.radius_graph`) inside the jitted forward -- the
calculator never hand-rolls a neighbour list. The whole model evaluation runs
through a single ``jax.jit``-compiled function (built once at construction via the
Flax NNX ``split``/``merge`` pattern), so repeated MD steps with the same atom
count reuse one compiled executable.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms  # noqa: TC002
from ase.calculators.calculator import all_changes, Calculator, PropertyNotImplementedError
from ase.stress import full_3x3_to_voigt_6_stress
from ase.units import Bohr, Hartree
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.atomistic import AtomisticModel  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


_FORCES_PER_LENGTH = Hartree / Bohr
"""Hartree/Bohr -> eV/Ang force conversion factor."""

_STRESS_PER_VOLUME = Hartree / Bohr**3
"""Hartree/Bohr^3 -> eV/Ang^3 stress conversion factor."""

_STRESS_PROPERTY = "stress"
"""ASE property name whose opifex 3x3 tensor is returned as a Voigt 6-vector."""

_INFO_CHARGE_KEY = "charge"
"""Key under which a total molecular charge may be supplied via ``atoms.info``."""


class OpifexCalculator(Calculator):
    r"""ASE :class:`~ase.calculators.calculator.Calculator` wrapping an opifex model.

    Converts an ``ase.Atoms`` object into an opifex
    :class:`~opifex.core.quantum.molecular_system.MolecularSystem`, runs the
    wrapped :class:`~opifex.neural.atomistic.AtomisticModel` on the jitted path,
    and writes ``self.results`` as host NumPy arrays in ASE units (eV, eV/Ang,
    eV/Ang^3 Voigt stress).

    Args:
        model: The assembled atomistic model to wrap. Its
            :attr:`~opifex.neural.atomistic.AtomisticModel.implemented_properties`
            (mapped to ASE names) become this calculator's
            :attr:`implemented_properties`. The model owns its own neighbour list,
            which builds the radius graph inside the jitted forward.
        cutoff: Neighbour cutoff radius in Bohr, recorded for provenance and made
            available as :attr:`cutoff`. The graph itself is built by the model's
            injected neighbour list; this value should match the radius that
            neighbour list was configured with.
        max_edges: Static upper bound on the number of edges, recorded as
            :attr:`max_edges` for provenance. Must match the model's ``max_edges``.
        label: Optional ASE calculator label (forwarded to the base class).

    Raises:
        ValueError: If ``cutoff`` or ``max_edges`` is not positive.
    """

    def __init__(
        self,
        model: AtomisticModel,
        *,
        cutoff: float,
        max_edges: int,
        label: str | None = None,
    ) -> None:
        """Wrap ``model`` and build the jitted forward function (once)."""
        super().__init__(label=label)
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if max_edges <= 0:
            raise ValueError(f"max_edges must be positive, got {max_edges}")
        self._model = model
        self.cutoff = cutoff
        self.max_edges = max_edges
        self._model_properties = tuple(model.implemented_properties)
        self.implemented_properties = list(self._ase_properties(self._model_properties))
        self.jitted_forward = self._build_jitted_forward(model)
        logger.debug(
            "OpifexCalculator ready: properties=%s cutoff=%.3f Bohr max_edges=%d",
            self.implemented_properties,
            cutoff,
            max_edges,
        )

    @staticmethod
    def _ase_properties(model_properties: Sequence[str]) -> tuple[str, ...]:
        """Map opifex head property names to ASE property names.

        The opifex stress head emits a 3x3 tensor under ``"stress"``; ASE's
        ``"stress"`` is the same physical quantity, returned as a Voigt 6-vector
        by :meth:`calculate`. All other names pass through unchanged.
        """
        return tuple(dict.fromkeys(model_properties))

    @staticmethod
    def _build_jitted_forward(
        model: AtomisticModel,
    ) -> jax.stages.Wrapped:
        """Build a ``jax.jit`` closure evaluating the model from raw arrays.

        Uses the Flax NNX ``split``/``merge`` pattern (the canonical opifex jitted
        path): the static ``graphdef`` is captured in the closure and the dynamic
        ``state`` is threaded as an argument, so the function is pure and
        ``jit``-traceable. ``cell`` is passed as ``None`` for non-periodic systems
        (a static argument), so periodic and free evaluations compile separately.
        """
        graphdef, _ = nnx.split(model)

        @partial(jax.jit, static_argnames=("charge",))
        def forward(
            state: nnx.State,
            atomic_numbers: Array,
            positions: Array,
            cell: Array | None,
            charge: int,
        ) -> dict[str, Array]:
            rebuilt = nnx.merge(graphdef, state)
            system = MolecularSystem(
                atomic_numbers=atomic_numbers,
                positions=positions,
                cell=cell,
                charge=charge,
            )
            return rebuilt(system)

        return forward

    def _require_supported(self, properties: Sequence[str]) -> None:
        """Raise :class:`PropertyNotImplementedError` for any unsupported request.

        Fails fast at the system boundary -- there is no silent fallback to a
        zeroed property.
        """
        for requested in properties:
            if requested not in self.implemented_properties:
                raise PropertyNotImplementedError(
                    f"{type(self).__name__} cannot compute {requested!r}; the wrapped "
                    f"model implements {self.implemented_properties}."
                )

    def _system_inputs(self, atoms: Atoms) -> tuple[Array, Array, Array | None, int]:
        """Convert ``ase.Atoms`` (Angstrom) to jitted-forward inputs (Bohr).

        Returns the atomic numbers, positions in Bohr, the cell in Bohr (or
        ``None`` when the atoms are non-periodic) and the static total charge.

        The charge is read from ``atoms.info["charge"]`` if present, otherwise it
        is the rounded sum of the ASE initial charges (zero when none are set).
        """
        atomic_numbers = jnp.asarray(atoms.get_atomic_numbers())
        positions = jnp.asarray(atoms.get_positions()) / Bohr
        cell: Array | None = None
        if atoms.cell.rank == 3 and bool(np.any(atoms.pbc)):
            cell = jnp.asarray(atoms.cell.array) / Bohr
        charge = self._total_charge(atoms)
        return atomic_numbers, positions, cell, charge

    @staticmethod
    def _total_charge(atoms: Atoms) -> int:
        """Return the system's static total charge from ``atoms``.

        Prefers an explicit ``atoms.info["charge"]``; otherwise falls back to the
        rounded sum of the ASE initial charges (``0`` when unset).
        """
        if _INFO_CHARGE_KEY in atoms.info:
            return int(atoms.info[_INFO_CHARGE_KEY])
        return round(float(np.sum(atoms.get_initial_charges())))

    def _to_ase_results(self, outputs: dict[str, Array]) -> dict[str, object]:
        """Convert atomic-unit model outputs to host ASE-unit results.

        Energy -> eV (Python float), forces -> eV/Ang ``(N, 3)`` array, stress ->
        eV/Ang^3 Voigt 6-vector. Only properties the model actually emits are
        written.
        """
        results: dict[str, object] = {}
        if "energy" in outputs:
            results["energy"] = float(np.asarray(outputs["energy"])) * Hartree
        if "forces" in outputs:
            results["forces"] = np.asarray(outputs["forces"]) * _FORCES_PER_LENGTH
        if _STRESS_PROPERTY in outputs:
            stress_3x3 = np.asarray(outputs[_STRESS_PROPERTY]) * _STRESS_PER_VOLUME
            results[_STRESS_PROPERTY] = full_3x3_to_voigt_6_stress(stress_3x3)
        return results

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Sequence[str] = all_changes,
    ) -> None:
        """Run the wrapped model on ``atoms`` and fill ``self.results``.

        Args:
            atoms: The :class:`ase.Atoms` to evaluate. Defaults to the calculator's
                stored ``self.atoms``.
            properties: ASE property names to compute (defaults to ``["energy"]``).
                Every requested property must be in :attr:`implemented_properties`.
            system_changes: ASE's list of changed inputs since the last call
                (forwarded to the base class; the model is re-evaluated regardless).

        Raises:
            PropertyNotImplementedError: If any requested property is unsupported.
        """
        requested = list(properties) if properties is not None else ["energy"]
        self._require_supported(requested)
        super().calculate(atoms, requested, system_changes)
        if self.atoms is None:  # pragma: no cover - guarded by ASE base class.
            raise ValueError("OpifexCalculator.calculate requires an Atoms object.")

        atomic_numbers, positions, cell, charge = self._system_inputs(self.atoms)
        _, state = nnx.split(self._model)
        outputs = self.jitted_forward(state, atomic_numbers, positions, cell, charge=charge)
        self.results = self._to_ase_results(outputs)


__all__ = ["OpifexCalculator"]

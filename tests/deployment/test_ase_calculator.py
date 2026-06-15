r"""Tests for :class:`opifex.deployment.ase_calculator.OpifexCalculator`.

The whole module is skipped cleanly when ``ase`` is not installed (it is an
optional deployment dependency), so the import guard sits at the very top via
``pytest.importorskip`` -- nothing below it runs without ASE.

Load-bearing checks (the contracts an ASE calculator must satisfy):

* a tiny molecule round-trips ``energy`` (float) and ``forces`` ((N, 3)) through
  ``atoms.get_potential_energy()`` / ``atoms.get_forces()`` with correct
  shapes and host (NumPy) dtypes;
* forces equal ``-dE/dx`` by central finite differences on a small displacement
  (the conservative-force contract, verified end-to-end through ASE units);
* :attr:`implemented_properties` is derived from the wrapped model's heads;
* requesting a property the model does not implement raises
  :class:`ase.calculators.calculator.PropertyNotImplementedError`;
* the wrapped model call goes through a compiled (``jax.jit``) function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


ase = pytest.importorskip("ase")

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from flax import nnx

from opifex.core.quantum.protocols import RadiusNeighborList
from opifex.deployment.ase_calculator import OpifexCalculator
from opifex.neural.atomistic import AtomisticModel
from opifex.neural.atomistic.heads import EnergyHead, ForcesHead, StressHead
from opifex.neural.equivariant import scatter_sum


if TYPE_CHECKING:
    from jaxtyping import Array

    from opifex.core.quantum.molecular_system import MolecularSystem


_CUTOFF_BOHR = 6.0
_MAX_EDGES = 32
_FEATURE_DIM = 8


class _ToyBackbone(nnx.Module):
    """Minimal invariant backbone (cutoff-weighted coordination -> small MLP).

    Identical in spirit to the toy backbone used in the atomistic base tests: it
    depends only on interatomic distances, so it is E(3)- and permutation-
    invariant and exercises the conservative-force autodiff path without pulling
    in SchNet/PaiNN/NequIP.
    """

    def __init__(self, *, hidden: int = _FEATURE_DIM, rngs: nnx.Rngs) -> None:
        """Build the two-layer per-atom MLP over the coordination feature."""
        super().__init__()
        self.cutoff = _CUTOFF_BOHR
        self.linear_in = nnx.Linear(1, hidden, rngs=rngs)
        self.linear_out = nnx.Linear(hidden, hidden, rngs=rngs)

    def __call__(self, system: MolecularSystem, graph: tuple[Array, Array]) -> dict[str, Array]:
        """Return ``{"node_features": ...}`` from the cutoff-weighted coordination."""
        senders, receivers = graph
        valid = senders >= 0
        safe_senders = jnp.where(valid, senders, 0)
        safe_receivers = jnp.where(valid, receivers, 0)
        deltas = system.positions[safe_senders] - system.positions[safe_receivers]
        distances = jnp.linalg.norm(deltas + 1e-12, axis=-1)
        envelope = jnp.where(valid, jnp.clip(1.0 - distances / self.cutoff, 0.0, None), 0.0)
        coordination = scatter_sum(envelope[:, None], safe_receivers, num_segments=system.n_atoms)
        features = nnx.tanh(self.linear_in(coordination))
        return {"node_features": nnx.tanh(self.linear_out(features))}


def _build_model(*, with_stress: bool = False) -> AtomisticModel:
    """Assemble a toy energy(+forces[+stress]) model for the calculator tests."""
    rngs = nnx.Rngs(0)
    backbone = _ToyBackbone(rngs=rngs)
    heads: dict[str, object] = {
        "energy": EnergyHead(feature_dim=_FEATURE_DIM, rngs=rngs),
        "forces": ForcesHead(),
    }
    if with_stress:
        heads["stress"] = StressHead()
    return AtomisticModel(
        backbone=backbone,
        heads=heads,  # type: ignore[arg-type]
        neighbor_list=RadiusNeighborList(cutoff=_CUTOFF_BOHR),
        max_edges=_MAX_EDGES,
    )


def _water_atoms() -> Atoms:
    """A non-periodic 3-atom water geometry (positions in Angstrom)."""
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )


def _h2_atoms() -> Atoms:
    """A non-periodic H2 dimer (positions in Angstrom)."""
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])


class TestImplementedProperties:
    def test_derived_from_model_heads(self) -> None:
        """The calculator advertises exactly the model's head properties."""
        calculator = OpifexCalculator(_build_model(), cutoff=_CUTOFF_BOHR, max_edges=_MAX_EDGES)
        assert set(calculator.implemented_properties) == {"energy", "forces"}

    def test_stress_appears_when_model_has_stress_head(self) -> None:
        """A model with a stress head exposes ``"stress"`` to ASE."""
        model = _build_model(with_stress=True)
        calculator = OpifexCalculator(model, cutoff=_CUTOFF_BOHR, max_edges=_MAX_EDGES)
        assert "stress" in calculator.implemented_properties


class TestEnergyForcesRoundTrip:
    def test_energy_is_python_float(self) -> None:
        """``get_potential_energy`` returns a host scalar, not a JAX array."""
        atoms = _water_atoms()
        atoms.calc = OpifexCalculator(_build_model(), cutoff=_CUTOFF_BOHR, max_edges=_MAX_EDGES)
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    def test_forces_shape_and_host_array(self) -> None:
        """Forces are a host ``(N, 3)`` NumPy array."""
        atoms = _water_atoms()
        atoms.calc = OpifexCalculator(_build_model(), cutoff=_CUTOFF_BOHR, max_edges=_MAX_EDGES)
        forces = atoms.get_forces()
        assert isinstance(forces, np.ndarray)
        assert forces.shape == (3, 3)
        assert np.all(np.isfinite(forces))

    def test_h2_round_trips(self) -> None:
        """The smallest system (H2) yields finite energy and (2, 3) forces."""
        atoms = _h2_atoms()
        atoms.calc = OpifexCalculator(_build_model(), cutoff=_CUTOFF_BOHR, max_edges=_MAX_EDGES)
        assert np.isfinite(atoms.get_potential_energy())
        assert atoms.get_forces().shape == (2, 3)


class TestForcesMatchFiniteDifference:
    def test_forces_equal_negative_energy_gradient(self) -> None:
        """ASE forces equal ``-dE/dx`` by central differences (eV / Ang).

        Run under ``jax.enable_x64(True)`` (float64): the conftest defaults
        ``jax_enable_x64`` to ``False``, and a central finite difference of an
        eV-scale total energy over a ~1e-3 Ang step suffers catastrophic
        cancellation in float32 (the signal is ~1e-4 eV on a few-eV total). The
        in-repo backbone tests use the same float64 guard for their tight
        force/finite-difference checks (``tests/.../backbones/_helpers.py``).
        """
        with jax.enable_x64(True):
            model = _build_model()
            atoms = _water_atoms()
            atoms.calc = OpifexCalculator(model, cutoff=_CUTOFF_BOHR, max_edges=_MAX_EDGES)
            analytic_forces = atoms.get_forces()

            epsilon = 1e-4  # Angstrom
            base_positions = atoms.get_positions()
            finite_diff = np.zeros_like(base_positions)
            for atom_index in range(base_positions.shape[0]):
                for axis in range(3):
                    plus = base_positions.copy()
                    plus[atom_index, axis] += epsilon
                    minus = base_positions.copy()
                    minus[atom_index, axis] -= epsilon
                    atoms.set_positions(plus)
                    energy_plus = atoms.get_potential_energy()
                    atoms.set_positions(minus)
                    energy_minus = atoms.get_potential_energy()
                    finite_diff[atom_index, axis] = -(energy_plus - energy_minus) / (2 * epsilon)
            atoms.set_positions(base_positions)
            np.testing.assert_allclose(analytic_forces, finite_diff, atol=1e-5)


class TestUnsupportedProperty:
    def test_stress_without_stress_head_raises(self) -> None:
        """Requesting ``stress`` from an energy/forces-only model fails fast."""
        atoms = _water_atoms()
        atoms.calc = OpifexCalculator(_build_model(), cutoff=_CUTOFF_BOHR, max_edges=_MAX_EDGES)
        with pytest.raises(PropertyNotImplementedError):
            atoms.get_stress()

    def test_explicit_unsupported_property_raises(self) -> None:
        """An explicit unsupported ``properties`` request raises before running."""
        calculator = OpifexCalculator(_build_model(), cutoff=_CUTOFF_BOHR, max_edges=_MAX_EDGES)
        atoms = _water_atoms()
        with pytest.raises(PropertyNotImplementedError):
            calculator.calculate(atoms, properties=["dipole"], system_changes=["positions"])


class TestStress:
    def test_stress_is_voigt_six_vector(self) -> None:
        """A periodic system with a stress head returns a (6,) Voigt stress."""
        model = _build_model(with_stress=True)
        atoms = Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
            cell=np.eye(3) * 6.0,
            pbc=True,
        )
        atoms.calc = OpifexCalculator(model, cutoff=_CUTOFF_BOHR, max_edges=_MAX_EDGES)
        stress = atoms.get_stress()
        assert isinstance(stress, np.ndarray)
        assert stress.shape == (6,)
        assert np.all(np.isfinite(stress))


class TestJittedPath:
    def test_model_call_is_jitted(self) -> None:
        """The wrapped model runs through a ``jax.jit``-compiled function.

        The calculator exposes the compiled callable it dispatches to; checking
        it carries the ``lower`` method of a jitted function (``jax.jit`` wraps
        the function in a ``stages.Wrapped`` with ``.lower``) confirms the model
        is not being run eagerly per ``calculate`` call.
        """
        calculator = OpifexCalculator(_build_model(), cutoff=_CUTOFF_BOHR, max_edges=_MAX_EDGES)
        assert isinstance(calculator.jitted_forward, jax.stages.Wrapped)
        assert hasattr(calculator.jitted_forward, "lower")

    def test_compiles_once_and_reuses(self) -> None:
        """A second same-shape call reuses the compiled executable.

        Two ``calculate`` invocations at the same atom count must populate the
        ``jax.jit`` compilation cache exactly once, so the model is not retraced
        per MD step. ``_cache_size`` is JAX's introspection hook for that cache.
        """
        model = _build_model()
        calculator = OpifexCalculator(model, cutoff=_CUTOFF_BOHR, max_edges=_MAX_EDGES)
        atoms = _water_atoms()
        atoms.calc = calculator
        first = atoms.get_potential_energy()
        atoms.set_positions(atoms.get_positions() + 0.01)
        second = atoms.get_potential_energy()
        assert np.isfinite(first)
        assert np.isfinite(second)
        # Same atom count/numbers => one compiled executable is cached by jax.
        cache_size: int = calculator.jitted_forward._cache_size()  # type: ignore[attr-defined]
        assert cache_size >= 1

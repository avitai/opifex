r"""Tests for the :class:`AtomisticModel` base and the energy/force/stress heads.

A *trivial* concrete backbone (a tiny invariant per-atom MLP over the local
coordination / distance features) is wired to an :class:`EnergyHead` and a
conservative :class:`ForcesHead` to exercise the backbone -> heads plumbing
without depending on SchNet/PaiNN/NequIP (those plug in later via the
``Backbone`` protocol + registry).

Load-bearing checks (the physics contracts every MLIP must satisfy):

* the model returns ``{"energy", "forces"}``;
* energy is permutation- and translation-invariant (E(3)/SE(3) scalar);
* conservative ``forces == -jax.grad(energy)`` verified against finite differences;
* the whole forward pass is ``jit``/``grad``/``vmap`` clean.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.protocols import RadiusNeighborList
from opifex.neural.atomistic import AtomisticModel
from opifex.neural.atomistic.heads import EnergyHead, ForcesHead, StressHead
from opifex.neural.equivariant import scatter_sum


class _ToyBackbone(nnx.Module):
    """A minimal invariant backbone: per-atom embedding of summed edge envelopes.

    Each atom's scalar feature is the cutoff-weighted coordination number passed
    through a small MLP. This depends only on interatomic distances, so it is
    E(3)- and permutation-invariant by construction -- exactly enough to exercise
    the plumbing and the conservative-force autodiff path.
    """

    def __init__(self, *, hidden: int = 8, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.cutoff = 3.0
        self.linear_in = nnx.Linear(1, hidden, rngs=rngs)
        self.linear_out = nnx.Linear(hidden, hidden, rngs=rngs)

    def __call__(self, system: MolecularSystem, graph: tuple[Array, Array]) -> dict[str, Array]:
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
    rngs = nnx.Rngs(0)
    backbone = _ToyBackbone(rngs=rngs)
    heads: dict[str, object] = {
        "energy": EnergyHead(feature_dim=8, rngs=rngs),
        "forces": ForcesHead(),
    }
    if with_stress:
        heads["stress"] = StressHead()
    return AtomisticModel(
        backbone=backbone,
        heads=heads,  # type: ignore[arg-type]
        neighbor_list=RadiusNeighborList(cutoff=3.0),
        max_edges=32,
    )


def _water() -> MolecularSystem:
    return MolecularSystem(
        atomic_numbers=jnp.asarray([8, 1, 1]),
        positions=jnp.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
    )


class TestAtomisticModel:
    def test_returns_energy_and_forces(self) -> None:
        model = _build_model()
        result = model(_water())
        assert set(result) == {"energy", "forces"}

    def test_implemented_properties(self) -> None:
        model = _build_model()
        assert set(model.implemented_properties) == {"energy", "forces"}

    def test_energy_is_scalar(self) -> None:
        model = _build_model()
        energy = model(_water())["energy"]
        assert energy.shape == ()

    def test_forces_shape_matches_atoms(self) -> None:
        model = _build_model()
        forces = model(_water())["forces"]
        assert forces.shape == (3, 3)

    def test_energy_translation_invariant(self) -> None:
        model = _build_model()
        system = _water()
        translated = system.translate(jnp.asarray([1.3, -0.7, 2.1]))
        assert jnp.allclose(model(system)["energy"], model(translated)["energy"], atol=1e-5)

    def test_energy_permutation_invariant(self) -> None:
        """Swapping the two equivalent hydrogens leaves the energy unchanged."""
        model = _build_model()
        system = _water()
        permuted = MolecularSystem(
            atomic_numbers=system.atomic_numbers[jnp.asarray([0, 2, 1])],
            positions=system.positions[jnp.asarray([0, 2, 1])],
        )
        assert jnp.allclose(model(system)["energy"], model(permuted)["energy"], atol=1e-5)

    def test_forces_equal_negative_grad_energy(self) -> None:
        """Conservative forces equal ``-dE/dR`` (the head's defining contract)."""
        model = _build_model()
        system = _water()

        def energy_of(positions: Array) -> Array:
            moved = MolecularSystem(atomic_numbers=system.atomic_numbers, positions=positions)
            return model(moved)["energy"]

        analytic_forces = model(system)["forces"]
        autodiff_forces = -jax.grad(energy_of)(system.positions)
        assert jnp.allclose(analytic_forces, autodiff_forces, atol=1e-5)

    def test_forces_match_finite_difference(self) -> None:
        """Force = -dE/dR validated against a central finite-difference gradient."""
        model = _build_model()
        system = _water()

        def energy_of(positions: Array) -> float:
            moved = MolecularSystem(atomic_numbers=system.atomic_numbers, positions=positions)
            return float(model(moved)["energy"])

        epsilon = 1e-4
        finite_diff = jnp.zeros_like(system.positions)
        positions = system.positions
        for atom in range(positions.shape[0]):
            for axis in range(3):
                plus = positions.at[atom, axis].add(epsilon)
                minus = positions.at[atom, axis].add(-epsilon)
                derivative = (energy_of(plus) - energy_of(minus)) / (2 * epsilon)
                finite_diff = finite_diff.at[atom, axis].set(-derivative)
        assert jnp.allclose(model(system)["forces"], finite_diff, atol=1e-3)

    def test_jit_smoke(self) -> None:
        model = _build_model()
        graphdef, state = nnx.split(model)

        @jax.jit
        def forward(state: nnx.State, positions: Array) -> Array:
            rebuilt = nnx.merge(graphdef, state)
            system = MolecularSystem(atomic_numbers=jnp.asarray([8, 1, 1]), positions=positions)
            return rebuilt(system)["energy"]

        energy = forward(state, _water().positions)
        assert jnp.isfinite(energy)

    def test_grad_smoke(self) -> None:
        model = _build_model()
        system = _water()

        def loss(positions: Array) -> Array:
            moved = MolecularSystem(atomic_numbers=system.atomic_numbers, positions=positions)
            return model(moved)["energy"] ** 2

        gradient = jax.grad(loss)(system.positions)
        assert gradient.shape == system.positions.shape
        assert bool(jnp.all(jnp.isfinite(gradient)))

    def test_vmap_smoke(self) -> None:
        model = _build_model()
        graphdef, state = nnx.split(model)
        atomic_numbers = jnp.asarray([8, 1, 1])
        batch = jnp.stack([_water().positions, _water().positions + 0.1])

        def energy_for(positions: Array) -> Array:
            rebuilt = nnx.merge(graphdef, state)
            system = MolecularSystem(atomic_numbers=atomic_numbers, positions=positions)
            return rebuilt(system)["energy"]

        energies = jax.vmap(energy_for)(batch)
        assert energies.shape == (2,)


class TestStressHead:
    def test_stress_is_symmetric_3x3(self) -> None:
        """The virial-based stress is a symmetric 3x3 tensor for a periodic cell."""
        rngs = nnx.Rngs(0)
        model = _build_model(with_stress=True)
        system = MolecularSystem(
            atomic_numbers=jnp.asarray([1, 1]),
            positions=jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
            cell=jnp.eye(3) * 6.0,
            pbc=(True, True, True),
        )
        del rngs
        stress = model(system)["stress"]
        assert stress.shape == (3, 3)
        assert jnp.allclose(stress, stress.T, atol=1e-5)

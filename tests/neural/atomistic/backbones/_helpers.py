r"""Shared fixtures and physics-contract assertions for backbone tests (DRY).

Every interatomic-potential backbone must satisfy the same physics contracts, so
the rotation/translation/permutation invariance, force equivariance,
force = -grad(E), toy-potential learning and jit/grad/vmap checks are factored
here and parametrised over the concrete backbone in each test module.

Tight numerical assertions run under ``jax.enable_x64(True)`` (float64) because
the conftest defaults ``jax_enable_x64`` to ``False``; random rotations use
:meth:`opifex.geometry.algebra.SO3Group.random_element`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.protocols import Backbone, RadiusNeighborList
from opifex.geometry.algebra import SO3Group
from opifex.neural.atomistic import AtomisticModel
from opifex.neural.atomistic.heads import EnergyHead, ForcesHead


_CUTOFF = 5.0
_MAX_EDGES = 32
_ROTATION_KEY = jax.random.PRNGKey(7)

BackboneFactory = Callable[[nnx.Rngs], Backbone]
"""Factory taking ``rngs`` and returning a concrete backbone module."""


def build_model(
    make_backbone: BackboneFactory, *, feature_dim: int, seed: int = 0
) -> AtomisticModel:
    """Assemble a backbone with energy + conservative-force heads.

    Args:
        make_backbone: Factory taking ``rngs`` and returning the backbone module.
        feature_dim: Width of the backbone's ``"node_features"`` (energy-head input).
        seed: Seed for the shared ``nnx.Rngs``.

    Returns:
        The assembled :class:`AtomisticModel`.
    """
    rngs = nnx.Rngs(seed)
    backbone = make_backbone(rngs)
    heads: dict[str, object] = {
        "energy": EnergyHead(feature_dim=feature_dim, rngs=rngs),
        "forces": ForcesHead(),
    }
    return AtomisticModel(
        backbone=backbone,
        heads=heads,  # type: ignore[arg-type]
        neighbor_list=RadiusNeighborList(cutoff=_CUTOFF),
        max_edges=_MAX_EDGES,
    )


def water(positions: Array | None = None) -> MolecularSystem:
    """Return a water molecule (optionally at custom positions, in Bohr-like units)."""
    if positions is None:
        positions = jnp.asarray(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=jnp.float64
        )
    return MolecularSystem(atomic_numbers=jnp.asarray([8, 1, 1]), positions=positions)


def random_rotation(dtype: jnp.dtype = jnp.float64) -> Array:
    """Return a uniformly random SO(3) rotation matrix of the given dtype."""
    return SO3Group().random_element(_ROTATION_KEY).astype(dtype)


def assert_energy_invariant(make_backbone: BackboneFactory, *, feature_dim: int) -> None:
    """Energy is invariant under rotation, translation and atom permutation."""
    with jax.enable_x64(True):
        model = build_model(make_backbone, feature_dim=feature_dim)
        system = water()
        rotation = random_rotation()
        translation = jnp.asarray([1.3, -0.7, 2.1], dtype=jnp.float64)
        moved = water(system.positions @ rotation.T + translation)
        permutation = jnp.asarray([0, 2, 1])
        permuted = MolecularSystem(
            atomic_numbers=system.atomic_numbers[permutation],
            positions=system.positions[permutation],
        )
        energy = model(system)["energy"]
        assert jnp.allclose(energy, model(moved)["energy"], atol=1e-9)
        assert jnp.allclose(energy, model(permuted)["energy"], atol=1e-9)


def assert_force_equivariant(make_backbone: BackboneFactory, *, feature_dim: int) -> None:
    """Forces rotate with the system: ``F(R . sys) = R . F(sys)``."""
    with jax.enable_x64(True):
        model = build_model(make_backbone, feature_dim=feature_dim)
        system = water()
        rotation = random_rotation()
        forces = model(system)["forces"]
        rotated_forces = model(water(system.positions @ rotation.T))["forces"]
        assert jnp.allclose(rotated_forces, forces @ rotation.T, atol=1e-8)


def assert_forces_match_finite_difference(
    make_backbone: BackboneFactory, *, feature_dim: int
) -> None:
    """Conservative forces equal ``-dE/dR``, checked by central finite differences."""
    with jax.enable_x64(True):
        model = build_model(make_backbone, feature_dim=feature_dim)
        system = water()

        def energy_of(positions: Array) -> float:
            moved = MolecularSystem(atomic_numbers=system.atomic_numbers, positions=positions)
            return float(model(moved)["energy"])

        epsilon = 1e-5
        positions = system.positions
        finite_diff = jnp.zeros_like(positions)
        for atom in range(positions.shape[0]):
            for axis in range(3):
                plus = positions.at[atom, axis].add(epsilon)
                minus = positions.at[atom, axis].add(-epsilon)
                derivative = (energy_of(plus) - energy_of(minus)) / (2 * epsilon)
                finite_diff = finite_diff.at[atom, axis].set(-derivative)
        assert jnp.allclose(model(system)["forces"], finite_diff, atol=1e-5)


def assert_learns_toy_potential(make_backbone: BackboneFactory, *, feature_dim: int) -> None:
    """A few gradient steps reduce the loss on a synthetic pairwise-energy target.

    The target is a sum of pairwise Morse-like well energies -- a smooth function
    of interatomic distances that the invariant backbone can fit -- evaluated on a
    handful of randomly perturbed water geometries.
    """
    model = build_model(make_backbone, feature_dim=feature_dim)
    base = water().positions

    def pairwise_target(positions: Array) -> Array:
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = jnp.sqrt(jnp.sum(deltas**2, axis=-1) + 1e-9)
        upper = jnp.triu(jnp.ones_like(distances), k=1)
        well = (1.0 - jnp.exp(-(distances - 1.0))) ** 2
        return jnp.sum(well * upper)

    key = jax.random.PRNGKey(0)
    perturbations = 0.1 * jax.random.normal(key, (6, *base.shape))
    geometries = base[None] + perturbations
    targets = jax.vmap(pairwise_target)(geometries)

    graphdef, state = nnx.split(model)

    def loss_fn(state: nnx.State) -> Array:
        rebuilt = nnx.merge(graphdef, state)

        def predict(positions: Array) -> Array:
            system = MolecularSystem(atomic_numbers=jnp.asarray([8, 1, 1]), positions=positions)
            return rebuilt(system)["energy"]

        predictions = jax.vmap(predict)(geometries)
        return jnp.mean((predictions - targets) ** 2)

    initial_loss = loss_fn(state)
    learning_rate = 1e-3
    for _ in range(40):
        gradients = jax.grad(loss_fn)(state)
        state = jax.tree.map(lambda p, g: p - learning_rate * g, state, gradients)
    final_loss = loss_fn(state)
    assert float(final_loss) < float(initial_loss)


def assert_jit_grad_vmap_smoke(make_backbone: BackboneFactory, *, feature_dim: int) -> None:
    """The full forward pass is ``jit``/``grad``/``vmap`` clean and finite."""
    model = build_model(make_backbone, feature_dim=feature_dim)
    graphdef, state = nnx.split(model)
    atomic_numbers = jnp.asarray([8, 1, 1])

    def energy_for(positions: Array) -> Array:
        rebuilt = nnx.merge(graphdef, state)
        system = MolecularSystem(atomic_numbers=atomic_numbers, positions=positions)
        return rebuilt(system)["energy"]

    jitted = jax.jit(energy_for)
    energy = jitted(water().positions)
    assert jnp.isfinite(energy)

    gradient = jax.grad(lambda p: energy_for(p) ** 2)(water().positions)
    assert gradient.shape == water().positions.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))

    batch = jnp.stack([water().positions, water().positions + 0.1])
    energies = jax.vmap(energy_for)(batch)
    assert energies.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(energies)))

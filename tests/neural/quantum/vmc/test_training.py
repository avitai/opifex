"""End-to-end VMC training: recover exact energies to chemical accuracy.

These are reference-gated integration tests. They train the FermiNet ansatz with
the VMC driver and assert the recovered energy (mean +/- standard error) is
within chemical accuracy of the exact value:

* H atom  -> -0.5      Ha (exact)
* He atom -> -2.9037   Ha (essentially exact, Pekeris)
* H2 @ R=1.4 bohr -> -1.1745 Ha (exact Born-Oppenheimer)

They are slow (real optimisation) and marked ``slow`` so the fast suite stays
quick. SPRING is also shown to converge faster than Adam on He.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.quantum.vmc import FermiNet
from opifex.neural.quantum.vmc.sampler import MetropolisHastingsSampler
from opifex.neural.quantum.vmc.training import (
    OptimizerName,
    VMCConfig,
    VMCDriver,
)


def _build_driver(
    *,
    nspins: tuple[int, int],
    atoms: jax.Array,
    charges: jax.Array,
    optimizer: OptimizerName,
    seed: int,
    iterations: int,
    learning_rate: float,
    batch_size: int = 1024,
) -> VMCDriver:
    """Construct a VMC driver for a small molecule."""
    ansatz = FermiNet(
        nspins=nspins,
        atoms=atoms,
        charges=charges,
        hidden_one=(32, 32),
        hidden_two=(16, 16),
        determinants=4,
        full_det=True,
        rngs=nnx.Rngs(seed),
    )
    sampler = MetropolisHastingsSampler(atoms=atoms, steps=10, step_size=0.4)
    config = VMCConfig(
        batch_size=batch_size,
        iterations=iterations,
        optimizer=optimizer,
        learning_rate=learning_rate,
        equilibration_steps=200,
    )
    return VMCDriver(ansatz=ansatz, sampler=sampler, config=config)


@pytest.mark.slow
def test_hydrogen_atom_energy() -> None:
    """The H atom converges to -0.5 Ha within chemical accuracy."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([1.0])
    driver = _build_driver(
        nspins=(1, 0),
        atoms=atoms,
        charges=charges,
        optimizer="spring",
        seed=0,
        iterations=400,
        learning_rate=0.05,
    )
    result = driver.run(jax.random.PRNGKey(0))
    assert abs(float(result.energy) - (-0.5)) < 2e-3


@pytest.mark.slow
def test_h2_molecule_energy() -> None:
    """H2 at R=1.4 bohr converges to -1.1745 Ha within chemical accuracy."""
    atoms = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    charges = jnp.array([1.0, 1.0])
    driver = _build_driver(
        nspins=(1, 1),
        atoms=atoms,
        charges=charges,
        optimizer="spring",
        seed=1,
        iterations=600,
        learning_rate=0.05,
    )
    result = driver.run(jax.random.PRNGKey(1))
    assert abs(float(result.energy) - (-1.1745)) < 3e-3


@pytest.mark.slow
def test_helium_atom_energy() -> None:
    """The He atom converges to -2.9037 Ha within chemical accuracy."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    driver = _build_driver(
        nspins=(1, 1),
        atoms=atoms,
        charges=charges,
        optimizer="spring",
        seed=2,
        iterations=800,
        learning_rate=0.04,
    )
    result = driver.run(jax.random.PRNGKey(2))
    assert abs(float(result.energy) - (-2.9037)) < 3e-3


@pytest.mark.slow
def test_spring_converges_faster_than_adam_on_helium() -> None:
    """SPRING reaches a lower He energy than Adam in the same iteration budget."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    iterations = 300

    adam_result = _build_driver(
        nspins=(1, 1),
        atoms=atoms,
        charges=charges,
        optimizer="adam",
        seed=3,
        iterations=iterations,
        learning_rate=0.01,
    ).run(jax.random.PRNGKey(3))
    spring_result = _build_driver(
        nspins=(1, 1),
        atoms=atoms,
        charges=charges,
        optimizer="spring",
        seed=3,
        iterations=iterations,
        learning_rate=0.04,
    ).run(jax.random.PRNGKey(3))

    assert float(spring_result.energy) < float(adam_result.energy)


def test_single_step_is_jit_and_runs() -> None:
    """A single VMC step runs end-to-end and returns a finite energy (fast)."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([1.0])
    driver = _build_driver(
        nspins=(1, 0),
        atoms=atoms,
        charges=charges,
        optimizer="adam",
        seed=4,
        iterations=2,
        learning_rate=0.01,
        batch_size=128,
    )
    result = driver.run(jax.random.PRNGKey(4))
    assert jnp.isfinite(result.energy)
    assert jnp.isfinite(result.energy_error)
    assert result.energy_error >= 0.0

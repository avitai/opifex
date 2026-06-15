"""Tests for the molecular local energy.

The local energy ``E_loc = -1/2 nabla^2 psi / psi + V`` must reproduce exact
analytic values for hydrogenic systems, where the local energy is *constant*
(the eigenstate property) and equals the exact total energy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from opifex.neural.quantum.vmc.hamiltonian import (
    local_energy,
    potential_energy,
)


def test_hydrogen_atom_local_energy_is_exactly_minus_half() -> None:
    r"""For the exact H 1s state ``log|psi| = -r``, ``E_loc == -0.5`` everywhere.

    The 1s eigenstate makes the local energy constant and equal to the ground-
    state energy ``-0.5 Ha`` at *every* electron position -- the cleanest
    possible reference for the kinetic + electron-nucleus terms.
    """
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([1.0])

    def log_abs(positions: jax.Array) -> jax.Array:
        return -jnp.linalg.norm(positions[0])

    energy_fn = local_energy(log_abs, atoms=atoms, charges=charges)
    for seed in range(5):
        positions = 2.0 * jax.random.normal(jax.random.PRNGKey(seed), (1, 3), dtype=jnp.float64)
        np.testing.assert_allclose(energy_fn(positions), -0.5, atol=1e-9)


def test_he_plus_like_local_energy_for_hydrogenic_z() -> None:
    r"""For a Z-charge hydrogenic 1s state ``log|psi| = -Z r``, ``E_loc = -Z^2/2``."""
    z = 2.0
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([z])

    def log_abs(positions: jax.Array) -> jax.Array:
        return -z * jnp.linalg.norm(positions[0])

    energy_fn = local_energy(log_abs, atoms=atoms, charges=charges)
    positions = jnp.array([[0.3, -0.4, 0.5]])
    np.testing.assert_allclose(energy_fn(positions), -(z**2) / 2.0, atol=1e-9)


def test_potential_energy_matches_explicit_coulomb_sum() -> None:
    """The molecular Coulomb potential equals the explicit pairwise sum."""
    atoms = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    charges = jnp.array([1.0, 1.0])
    positions = jnp.array([[0.0, 0.0, 0.2], [0.0, 0.0, 1.2]])

    got = potential_energy(positions, atoms=atoms, charges=charges)

    # Explicit: V_ee + V_eN + V_NN.
    v_ee = 1.0 / jnp.linalg.norm(positions[0] - positions[1])
    v_en = -(
        1.0 / jnp.linalg.norm(positions[0] - atoms[0])
        + 1.0 / jnp.linalg.norm(positions[0] - atoms[1])
        + 1.0 / jnp.linalg.norm(positions[1] - atoms[0])
        + 1.0 / jnp.linalg.norm(positions[1] - atoms[1])
    )
    v_nn = 1.0 / 1.4
    np.testing.assert_allclose(got, v_ee + v_en + v_nn, rtol=1e-10)


def test_local_energy_forward_and_oracle_agree() -> None:
    """Forward-Laplacian and oracle kinetic methods give the same local energy."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([1.0])

    def log_abs(positions: jax.Array) -> jax.Array:
        return -1.3 * jnp.linalg.norm(positions[0]) ** 1.0

    e_fwd = local_energy(log_abs, atoms=atoms, charges=charges, method="forward")
    e_oracle = local_energy(log_abs, atoms=atoms, charges=charges, method="oracle")
    positions = jnp.array([[0.5, 0.1, -0.2]])
    np.testing.assert_allclose(e_fwd(positions), e_oracle(positions), atol=1e-7)


def test_local_energy_is_jit_and_vmap_clean() -> None:
    """The local energy jits and vmaps over a batch of walkers."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([1.0])

    def log_abs(positions: jax.Array) -> jax.Array:
        return -jnp.linalg.norm(positions[0])

    energy_fn = local_energy(log_abs, atoms=atoms, charges=charges)
    walkers = jax.random.normal(jax.random.PRNGKey(9), (16, 1, 3), dtype=jnp.float64)
    energies = jax.jit(jax.vmap(energy_fn))(walkers)
    assert energies.shape == (16,)
    np.testing.assert_allclose(energies, -0.5 * jnp.ones(16), atol=1e-9)

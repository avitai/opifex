"""Tests for the harmonic-mean Metropolis-Hastings sampler.

The sampler must produce walkers distributed as ``|psi|^2`` (validated against
the exact H 1s density), report an acceptance fraction in a sane band, and run
under ``jit`` via an internal ``lax.scan`` over MCMC steps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from opifex.neural.quantum.vmc.sampler import MetropolisHastingsSampler


def _h_atom_log_abs(positions: jax.Array) -> jax.Array:
    """Exact hydrogen 1s log-amplitude ``log|psi| = -r``."""
    return -jnp.linalg.norm(positions[0])


def test_sampler_returns_walkers_and_acceptance() -> None:
    """One sampling call returns updated walkers and an acceptance fraction."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    sampler = MetropolisHastingsSampler(atoms=atoms, steps=20, step_size=0.1)
    walkers = jax.random.normal(jax.random.PRNGKey(0), (256, 1, 3), dtype=jnp.float64)
    new_walkers, acceptance = sampler.sample(_h_atom_log_abs, walkers, jax.random.PRNGKey(1))
    assert new_walkers.shape == walkers.shape
    assert 0.0 <= float(acceptance) <= 1.0


def test_acceptance_in_sane_band() -> None:
    """A reasonable step size yields an acceptance fraction in ``[0.3, 0.95]``."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    sampler = MetropolisHastingsSampler(atoms=atoms, steps=40, step_size=0.3)
    walkers = jax.random.normal(jax.random.PRNGKey(2), (512, 1, 3), dtype=jnp.float64)
    _, acceptance = sampler.sample(_h_atom_log_abs, walkers, jax.random.PRNGKey(3))
    assert 0.3 <= float(acceptance) <= 0.95


def test_samples_match_hydrogen_1s_density() -> None:
    r"""Equilibrated walkers reproduce ``<r> = 1.5 a0`` for the H 1s density.

    The radial expectation of ``|psi_{1s}|^2 \propto e^{-2r}`` is exactly
    ``<r> = 3/2`` Bohr -- a sharp, distribution-level check on the sampler.
    """
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    sampler = MetropolisHastingsSampler(atoms=atoms, steps=50, step_size=0.4)
    walkers = jax.random.normal(jax.random.PRNGKey(4), (4096, 1, 3), dtype=jnp.float64)
    key = jax.random.PRNGKey(5)
    # Burn-in then collect.
    for _ in range(20):
        key, subkey = jax.random.split(key)
        walkers, _ = sampler.sample(_h_atom_log_abs, walkers, subkey)
    radii = jnp.linalg.norm(walkers[:, 0, :], axis=-1)
    np.testing.assert_allclose(float(jnp.mean(radii)), 1.5, atol=0.1)


def test_sample_is_jit_clean() -> None:
    """The sampling step jits without retracing errors."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    sampler = MetropolisHastingsSampler(atoms=atoms, steps=10, step_size=0.2)
    walkers = jax.random.normal(jax.random.PRNGKey(6), (128, 1, 3), dtype=jnp.float64)

    jitted = jax.jit(lambda w, k: sampler.sample(_h_atom_log_abs, w, k))
    new_walkers, acceptance = jitted(walkers, jax.random.PRNGKey(7))
    assert new_walkers.shape == walkers.shape
    assert jnp.isfinite(acceptance)

r"""Tests for the long-range electrostatics add-on (Latent Ewald Summation).

Load-bearing physics contracts (Cheng 2025, "Latent Ewald Summation",
arXiv:2408.15165; standard Ewald summation; ``../jax-md`` ``_energy``
``electrostatics.py``):

* **Free systems** -- the long-range energy is the bare pairwise Coulomb sum
  :math:`\tfrac12 \sum_{i\neq j} q_i q_j / r_{ij}`; the helper reproduces a
  hand-computed value for a small toy.
* **Invariance** -- the energy is translation and rotation invariant (it depends
  only on interatomic distances).
* **Ewald correctness** -- for a periodic system the full Ewald energy
  (real + reciprocal + self + net-charge background) is **independent of the
  splitting parameter** :math:`\eta` within tolerance (the standard Ewald
  convergence check).
* **Charge conservation** -- :class:`LatentEwaldHead`'s latent charges sum to the
  system net charge (reusing ``conserve_total_charge``).
* **Transform safety** -- the forward pass is ``jit`` / ``grad`` / ``vmap`` clean.

Tight numerical assertions run under ``jax.enable_x64(True)`` (the conftest
defaults ``jax_enable_x64`` to ``False``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.atomistic.long_range import (
    latent_ewald_energy,
    LatentEwaldHead,
)


_FEATURE_DIM = 8
_GRAPH: tuple[Array, Array] = (jnp.asarray([0, 1, 2]), jnp.asarray([1, 2, 0]))


def _rotation_z(angle: float) -> Array:
    """Return a proper rotation about the z-axis (float64)."""
    cos = jnp.cos(jnp.asarray(angle, dtype=jnp.float64))
    sin = jnp.sin(jnp.asarray(angle, dtype=jnp.float64))
    return jnp.asarray([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64)


def _embeddings(seed: int = 0, n_atoms: int = 3) -> dict[str, Array]:
    """Return per-atom invariant ``node_features``."""
    key = jax.random.PRNGKey(seed)
    return {"node_features": jax.random.normal(key, (n_atoms, _FEATURE_DIM), dtype=jnp.float64)}


def _system(charge: int = 0, positions: Array | None = None) -> MolecularSystem:
    """Return a neutral 3-atom free system."""
    default = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [-0.3, 1.1, 0.0]], dtype=jnp.float64)
    return MolecularSystem(
        atomic_numbers=jnp.asarray([8, 1, 1]),
        positions=default if positions is None else positions,
        charge=charge,
    )


class TestFreeEnergy:
    def test_matches_hand_computed_pair_sum(self) -> None:
        with jax.enable_x64(True):
            positions = jnp.asarray(
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=jnp.float64
            )
            charges = jnp.asarray([1.0, -1.0, 0.5], dtype=jnp.float64)
            # 1/2 sum_{i!=j} q_i q_j / r_ij  (r_01=3, r_02=4, r_12=5).
            expected = (
                charges[0] * charges[1] / 3.0
                + charges[0] * charges[2] / 4.0
                + charges[1] * charges[2] / 5.0
            )
            energy = latent_ewald_energy(charges, positions, cell=None)
            assert jnp.allclose(energy, expected, atol=1e-10)

    def test_two_charge_system(self) -> None:
        with jax.enable_x64(True):
            positions = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
            charges = jnp.asarray([1.5, -2.0], dtype=jnp.float64)
            energy = latent_ewald_energy(charges, positions, cell=None)
            assert jnp.allclose(energy, 1.5 * -2.0 / 2.0, atol=1e-10)

    def test_translation_invariance(self) -> None:
        with jax.enable_x64(True):
            positions = jnp.asarray(
                [[0.0, 0.0, 0.0], [1.3, 0.2, 0.0], [0.1, 1.1, 0.4]], dtype=jnp.float64
            )
            charges = jnp.asarray([0.7, -0.4, -0.3], dtype=jnp.float64)
            shift = jnp.asarray([5.0, -2.0, 3.0], dtype=jnp.float64)
            base = latent_ewald_energy(charges, positions, cell=None)
            moved = latent_ewald_energy(charges, positions + shift, cell=None)
            assert jnp.allclose(base, moved, atol=1e-10)

    def test_rotation_invariance(self) -> None:
        with jax.enable_x64(True):
            positions = jnp.asarray(
                [[0.0, 0.0, 0.0], [1.3, 0.2, 0.0], [0.1, 1.1, 0.4]], dtype=jnp.float64
            )
            charges = jnp.asarray([0.7, -0.4, -0.3], dtype=jnp.float64)
            rotation = _rotation_z(0.7)
            base = latent_ewald_energy(charges, positions, cell=None)
            rotated = latent_ewald_energy(charges, positions @ rotation.T, cell=None)
            assert jnp.allclose(base, rotated, atol=1e-10)


class TestPeriodicEwald:
    def test_independent_of_eta(self) -> None:
        with jax.enable_x64(True):
            cell = 6.0 * jnp.eye(3, dtype=jnp.float64)
            positions = jnp.asarray(
                [[0.5, 0.5, 0.5], [3.0, 3.2, 2.8], [1.5, 4.0, 5.0]], dtype=jnp.float64
            )
            charges = jnp.asarray([1.0, -0.6, -0.4], dtype=jnp.float64)
            energy_low = latent_ewald_energy(
                charges, positions, cell=cell, eta=0.30, reciprocal_cutoff=12
            )
            energy_high = latent_ewald_energy(
                charges, positions, cell=cell, eta=0.55, reciprocal_cutoff=12
            )
            residual = float(jnp.abs(energy_low - energy_high))
            assert residual < 1e-4, f"eta-dependence residual too large: {residual}"

    def test_periodic_translation_invariance(self) -> None:
        with jax.enable_x64(True):
            cell = 6.0 * jnp.eye(3, dtype=jnp.float64)
            positions = jnp.asarray(
                [[0.5, 0.5, 0.5], [3.0, 3.2, 2.8], [1.5, 4.0, 5.0]], dtype=jnp.float64
            )
            charges = jnp.asarray([1.0, -0.6, -0.4], dtype=jnp.float64)
            shift = jnp.asarray([1.0, 2.0, -0.5], dtype=jnp.float64)
            base = latent_ewald_energy(charges, positions, cell=cell, eta=0.4)
            moved = latent_ewald_energy(charges, positions + shift, cell=cell, eta=0.4)
            assert jnp.allclose(base, moved, atol=1e-6)


class TestTransformSafety:
    def test_free_jit(self) -> None:
        positions = jnp.asarray(
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=jnp.float32
        )
        charges = jnp.asarray([0.5, -0.5, 0.0], dtype=jnp.float32)
        jitted = jax.jit(lambda q, r: latent_ewald_energy(q, r, cell=None))
        assert jnp.isfinite(jitted(charges, positions))

    def test_free_grad(self) -> None:
        positions = jnp.asarray(
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=jnp.float32
        )
        charges = jnp.asarray([0.5, -0.5, 0.0], dtype=jnp.float32)
        grad = jax.grad(lambda r: latent_ewald_energy(charges, r, cell=None))(positions)
        assert grad.shape == positions.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_free_vmap(self) -> None:
        positions = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 3))
        charges = jnp.asarray([0.5, -0.5, 0.0], dtype=jnp.float32)
        energies = jax.vmap(lambda r: latent_ewald_energy(charges, r, cell=None))(positions)
        assert energies.shape == (4,)
        assert jnp.all(jnp.isfinite(energies))

    def test_periodic_jit_grad(self) -> None:
        cell = 6.0 * jnp.eye(3, dtype=jnp.float32)
        positions = jnp.asarray(
            [[0.5, 0.5, 0.5], [3.0, 3.2, 2.8], [1.5, 4.0, 5.0]], dtype=jnp.float32
        )
        charges = jnp.asarray([1.0, -0.6, -0.4], dtype=jnp.float32)
        jitted = jax.jit(lambda r: latent_ewald_energy(charges, r, cell=cell, eta=0.4))
        grad = jax.grad(jitted)(positions)
        assert jnp.isfinite(jitted(positions))
        assert jnp.all(jnp.isfinite(grad))


class TestLatentEwaldHead:
    def test_implemented_properties(self) -> None:
        head = LatentEwaldHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        assert head.implemented_properties == ("long_range_energy",)

    def test_emits_long_range_energy_scalar(self) -> None:
        head = LatentEwaldHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        out = head(_system(), _GRAPH, _embeddings())
        assert "long_range_energy" in out
        assert out["long_range_energy"].shape == ()

    def test_latent_charges_conserve_total_charge(self) -> None:
        with jax.enable_x64(True):
            head = LatentEwaldHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
            for net_charge in (0, 1, -2):
                charges = head.latent_charges(_system(charge=net_charge), _embeddings())
                assert jnp.allclose(jnp.sum(charges), float(net_charge), atol=1e-9)

    def test_head_jit_grad_vmap(self) -> None:
        head = LatentEwaldHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        graphdef, state = nnx.split(head)
        system = _system()

        def energy(state_arg: nnx.State, features: Array) -> Array:
            module = nnx.merge(graphdef, state_arg)
            return module(system, _GRAPH, {"node_features": features})["long_range_energy"]

        features = _embeddings()["node_features"].astype(jnp.float32)
        assert jnp.isfinite(jax.jit(energy)(state, features))
        grad = jax.grad(lambda f: energy(state, f))(features)
        assert grad.shape == features.shape
        batched = jax.vmap(lambda f: energy(state, f))(features[None])
        assert batched.shape == (1,)

"""Tests for the PsiFormer self-attention wavefunction ansatz.

The PsiFormer (von Glehn, Spencer & Pfau, *A Self-Attention Ansatz for
Ab-initio Quantum Chemistry*, arXiv:2211.13672; reference implementation
``../ferminet`` ``psiformer.py``) replaces FermiNet's pooled equivariant
backbone with a stack of multi-head self-attention blocks over the electrons.
It must satisfy the same contract as :class:`FermiNet`: a single-walker
``(sign, log|psi|)`` evaluation that is antisymmetric under same-spin electron
exchange, permutation-equivariant in its attention backbone, and fully ``jit``
/ ``grad`` / ``vmap`` clean. A sanity check confirms the He local energy is
finite through the existing Hamiltonian/Laplacian stack.

``float64`` and the ``high`` matmul precision required for the ~1e-9
antisymmetry tolerances are supplied by the package conftest autouse fixture
(``tests/neural/quantum/vmc/conftest.py::vmc_x64_and_precision``), which is
inherited by this subdirectory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from opifex.neural.quantum.vmc.hamiltonian import local_energy
from opifex.neural.quantum.vmc.wavefunctions.psiformer import PsiFormer


def _make_helium_ansatz() -> tuple[PsiFormer, jax.Array]:
    """Return a tiny PsiFormer for the He atom and a single-walker config."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    ansatz = PsiFormer(
        nspins=(1, 1),
        atoms=atoms,
        charges=charges,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        mlp_hidden=(8,),
        determinants=2,
        use_layer_norm=True,
        rngs=nnx.Rngs(0),
    )
    positions = jax.random.normal(jax.random.PRNGKey(5), (2, 3), dtype=jnp.float64)
    return ansatz, positions


def _make_lithium_same_spin_ansatz() -> tuple[PsiFormer, jax.Array]:
    """Return a PsiFormer with three same-spin electrons (Li-like, all spin-up)."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([3.0])
    ansatz = PsiFormer(
        nspins=(3, 0),
        atoms=atoms,
        charges=charges,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        mlp_hidden=(8,),
        determinants=2,
        use_layer_norm=False,
        rngs=nnx.Rngs(1),
    )
    positions = jax.random.normal(jax.random.PRNGKey(6), (3, 3), dtype=jnp.float64)
    return ansatz, positions


def test_output_is_finite_scalar_sign_and_logabs() -> None:
    """A single-walker evaluation returns a finite scalar ``(sign, log|psi|)``."""
    ansatz, positions = _make_helium_ansatz()
    sign, log_abs = ansatz(positions)
    assert sign.shape == ()
    assert log_abs.shape == ()
    assert jnp.isfinite(log_abs)
    assert jnp.abs(jnp.abs(sign) - 1.0) < 1e-10


def test_antisymmetry_under_same_spin_exchange() -> None:
    """Swapping two same-spin electrons flips the sign, preserves ``log|psi|``."""
    ansatz, positions = _make_lithium_same_spin_ansatz()
    sign_a, log_a = ansatz(positions)
    # Exchange electrons 0 and 2 (both spin-up).
    swapped = positions[jnp.array([2, 1, 0])]
    sign_b, log_b = ansatz(swapped)
    np.testing.assert_allclose(log_a, log_b, atol=1e-9)
    np.testing.assert_allclose(sign_a, -sign_b, atol=1e-9)


def test_attention_backbone_is_permutation_equivariant() -> None:
    """The attention backbone commutes with a same-spin electron permutation.

    Permuting same-spin electrons must permute the per-electron backbone
    features by the same permutation (equivariance), which is what guarantees
    the determinant antisymmetry downstream.
    """
    ansatz, positions = _make_lithium_same_spin_ansatz()
    permutation = jnp.array([2, 0, 1])
    features = ansatz.backbone_features(positions)
    permuted_features = ansatz.backbone_features(positions[permutation])
    np.testing.assert_allclose(permuted_features, features[permutation], atol=1e-9)


def test_logabs_is_jit_and_grad_clean() -> None:
    """The log-magnitude is differentiable under ``jit``."""
    ansatz, positions = _make_helium_ansatz()
    graphdef, state = nnx.split(ansatz)

    def log_abs(pos: jax.Array) -> jax.Array:
        return nnx.merge(graphdef, state)(pos)[1]

    grad = jax.jit(jax.grad(log_abs))(positions)
    assert grad.shape == positions.shape
    assert jnp.all(jnp.isfinite(grad))


def test_logabs_vmaps_over_walkers() -> None:
    """The single-walker ansatz vmaps cleanly over a batch of walkers."""
    ansatz, _ = _make_helium_ansatz()
    graphdef, state = nnx.split(ansatz)
    walkers = jax.random.normal(jax.random.PRNGKey(7), (16, 2, 3), dtype=jnp.float64)

    def log_abs(pos: jax.Array) -> jax.Array:
        return nnx.merge(graphdef, state)(pos)[1]

    out = jax.jit(jax.vmap(log_abs))(walkers)
    assert out.shape == (16,)
    assert jnp.all(jnp.isfinite(out))


def test_helium_local_energy_is_finite() -> None:
    """The He local energy is finite through the Hamiltonian/Laplacian stack."""
    ansatz, positions = _make_helium_ansatz()
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    graphdef, state = nnx.split(ansatz)

    def log_abs(pos: jax.Array) -> jax.Array:
        return nnx.merge(graphdef, state)(pos)[1]

    energy_fn = local_energy(log_abs, atoms=atoms, charges=charges, method="forward")
    energy = jax.jit(energy_fn)(positions)
    assert energy.shape == ()
    assert jnp.isfinite(energy)


def test_layer_norm_toggle_runs() -> None:
    """Both LayerNorm-on and LayerNorm-off configurations produce finite output."""
    ansatz, positions = _make_lithium_same_spin_ansatz()  # use_layer_norm=False
    _, log_abs = ansatz(positions)
    assert jnp.isfinite(log_abs)


def test_invalid_attention_config_raises() -> None:
    """A non-positive head count or empty system is rejected at construction."""
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([2.0])
    with pytest.raises(ValueError, match="num_heads"):
        PsiFormer(
            nspins=(1, 1),
            atoms=atoms,
            charges=charges,
            num_heads=0,
            head_dim=4,
            rngs=nnx.Rngs(0),
        )
    with pytest.raises(ValueError, match="No electrons"):
        PsiFormer(
            nspins=(0, 0),
            atoms=atoms,
            charges=charges,
            rngs=nnx.Rngs(0),
        )

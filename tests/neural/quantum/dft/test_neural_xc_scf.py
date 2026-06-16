r"""Tests for the learned exchange-correlation functional wired into the real SCF.

The :class:`~opifex.neural.quantum.neural_xc.NeuralXCFunctional` is used as the
exchange-correlation functional of the molecular restricted Kohn-Sham solver,
exactly like the LDA/PBE paths. Validation covers:

* the neural-XC SCF converges on H2 (finite energy, idempotent density);
* the autodiff XC potential -- both the ``rho`` and the ``sigma`` (density
  gradient) channels -- matches a finite difference of the neural energy density
  (the previously dead gradient channel is live);
* the learned XC is trainable through the implicit-diff SCF: a short fit to PBE
  total energies on H2 reduces the loss monotonically (exact ``dE/dtheta`` via
  the implicit function theorem);
* the neural-XC SCF energy and its parameter gradient compile under ``jax.jit``.

References
----------
* J. Kirkpatrick et al., *Science* **374**, 1385 (2021), arXiv:2102.06179 (DM21)
  -- a machine-learned enhancement-factor functional.
* X. Zhang, G. K.-L. Chan, *J. Chem. Phys.* **157**, 204801 (2022),
  arXiv:2207.13836 -- implicit differentiation of the SCF fixed point.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from opifex.core.quantum.backend import JaxGaussianBackend
from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.quantum.dft.scf import SCFSolver
from opifex.neural.quantum.neural_xc import NeuralXCFunctional


_BOHR_PER_ANGSTROM = 1.0 / 0.52917721067


def _h2_system(bond_angstrom: float = 0.74) -> MolecularSystem:
    """H2 on the z-axis at the given bond length (Angstrom)."""
    bond = bond_angstrom * _BOHR_PER_ANGSTROM
    return MolecularSystem(
        atomic_numbers=jnp.array([1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]]),
        basis_set="sto-3g",
    )


def _neural_functional(seed: int = 0) -> NeuralXCFunctional:
    """A small initialised neural XC functional for testing."""
    return NeuralXCFunctional(hidden_sizes=(16, 16), use_attention=False, rngs=nnx.Rngs(seed))


def _perturb_output_layer(
    functional: NeuralXCFunctional, *, bias_shift: float, kernel_shift: float
) -> None:
    """Move the output layer off its zero (LDA) initialisation for testing."""
    layer = functional.output_layer
    layer.bias.value = layer.bias.value + bias_shift  # type: ignore[reportOptionalMemberAccess]
    layer.kernel.value = layer.kernel.value + kernel_shift


def test_neural_xc_scf_converges_h2() -> None:
    """The H2 RKS SCF with the neural XC functional converges to a finite energy."""
    with jax.enable_x64(True):
        solver = SCFSolver(_h2_system(), neural_functional=_neural_functional(), max_iterations=80)
        result = solver.solve()
    assert result.converged
    assert bool(jnp.isfinite(result.total_energy))


def test_neural_xc_scf_density_is_idempotent() -> None:
    """The neural-XC closed-shell density satisfies ``D S D = 2 D``."""
    with jax.enable_x64(True):
        system = _h2_system()
        result = SCFSolver(
            system, neural_functional=_neural_functional(), max_iterations=80
        ).solve()
        basis = AtomicOrbitalBasis.from_molecular_system(system)
        overlap = JaxGaussianBackend(system, basis).overlap()
        density = result.density_matrix
        residual = float(jnp.max(jnp.abs(density @ overlap @ density - 2.0 * density)))
    assert residual < 1e-6


def test_neural_xc_potential_matches_finite_difference() -> None:
    r"""AD XC potential (rho and sigma channels) matches FD of the neural energy.

    Both functional derivatives of :math:`\rho\,\varepsilon_{xc}(\rho,\sigma)`
    are verified against central finite differences -- the density-gradient
    (:math:`\sigma`) channel is live, not zeroed.
    """
    with jax.enable_x64(True):
        functional = _neural_functional()
        # Move the network off its zero (LDA) initialisation so the sigma channel
        # is non-trivial.
        _perturb_output_layer(functional, bias_shift=0.3, kernel_shift=0.1)

        rho = jnp.array([0.05, 0.2, 0.5, 1.0, 2.0])
        sigma = jnp.array([0.01, 0.05, 0.1, 0.3, 0.5])
        v_rho, v_sigma = functional.xc_potential_components(rho, sigma)

        def energy(r: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(r * functional.energy_density_from_sigma(r, s))

        epsilon = 1e-6
        fd_rho = np.zeros(rho.shape)
        fd_sigma = np.zeros(sigma.shape)
        for i in range(rho.shape[0]):
            fd_rho[i] = (
                energy(rho.at[i].add(epsilon), sigma) - energy(rho.at[i].add(-epsilon), sigma)
            ) / (2.0 * epsilon)
            fd_sigma[i] = (
                energy(rho, sigma.at[i].add(epsilon)) - energy(rho, sigma.at[i].add(-epsilon))
            ) / (2.0 * epsilon)

    np.testing.assert_allclose(np.asarray(v_rho), fd_rho, atol=1e-5)
    np.testing.assert_allclose(np.asarray(v_sigma), fd_sigma, atol=1e-5)
    # The sigma channel must be genuinely non-zero (not the dead-channel bug).
    assert float(jnp.max(jnp.abs(v_sigma))) > 1e-4


def test_neural_xc_sigma_channel_is_live_for_gga_gradients() -> None:
    """``compute_functional_derivative`` uses live rho/sigma channels (no zeroing)."""
    with jax.enable_x64(True):
        functional = _neural_functional()
        # Perturb the kernel (not just the bias) so the dimensionless gradient
        # features feed the enhancement -- a bias-only network is gradient-blind.
        _perturb_output_layer(functional, bias_shift=0.4, kernel_shift=0.2)
        density = jnp.array([[0.1, 0.5, 1.0, 2.0]])
        gradients = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 3))
        v_rho = functional.compute_functional_derivative(density, gradients)
        # The same density with a different gradient must give a different
        # potential -- proving the gradient channel feeds the result.
        other = functional.compute_functional_derivative(density, 2.0 * gradients)
    assert v_rho.shape == density.shape
    assert float(jnp.max(jnp.abs(v_rho - other))) > 1e-6


@pytest.mark.slow
def test_learned_xc_fit_to_pbe_reduces_loss_monotonically() -> None:
    """A few optimiser steps fitting the neural XC to PBE energies reduce the loss.

    Differentiating the implicit-diff SCF total energy with respect to the neural
    XC parameters gives an exact ``dE/dtheta`` (the implicit function theorem), so
    the learned-XC training path works end to end. The loss must decrease
    monotonically over a handful of steps.
    """
    with jax.enable_x64(True):
        geometries = [0.6, 0.74, 0.9]
        references = [
            float(
                SCFSolver(_h2_system(b), functional="pbe", max_iterations=80).solve().total_energy
            )
            for b in geometries
        ]
        functional = _neural_functional()
        _, state = nnx.split(functional)
        solvers = [
            SCFSolver(_h2_system(b), neural_functional=functional, max_iterations=60)
            for b in geometries
        ]

        def loss(parameters: nnx.State) -> jnp.ndarray:
            total = jnp.asarray(0.0)
            for solver, reference in zip(solvers, references, strict=True):
                total = total + (solver.energy_from_state(parameters) - reference) ** 2
            return total / len(references)

        optimiser = optax.adam(3e-3)
        opt_state = optimiser.init(state)  # pyright: ignore[reportArgumentType]
        losses: list[float] = []
        for _ in range(7):
            value, grad = jax.value_and_grad(loss)(state)
            losses.append(float(value))
            updates, opt_state = optimiser.update(grad, opt_state)  # pyright: ignore[reportArgumentType]
            state = optax.apply_updates(state, updates)  # pyright: ignore[reportArgumentType]

    assert all(losses[i + 1] < losses[i] for i in range(len(losses) - 1))
    # The fit makes meaningful progress, not a negligible step.
    assert losses[-1] < 0.6 * losses[0]


def test_neural_xc_scf_energy_is_jittable() -> None:
    """The neural-XC SCF energy and its parameter gradient compile under jit."""
    with jax.enable_x64(True):
        functional = _neural_functional()
        _, state = nnx.split(functional)
        solver = SCFSolver(_h2_system(), neural_functional=functional, max_iterations=60)

        energy_fn = jax.jit(solver.energy_from_state)
        grad_fn = jax.jit(jax.grad(solver.energy_from_state))

        eager = float(solver.energy_from_state(state))
        jitted = float(energy_fn(state))
        gradient = grad_fn(state)
    assert jitted == eager or abs(jitted - eager) < 1e-8
    grad_leaves = jax.tree_util.tree_leaves(gradient)
    assert len(grad_leaves) > 0
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in grad_leaves)

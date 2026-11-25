r"""Position-traceable Kohn-Sham total energy, implicit-diff SCF and forces.

This module holds the differentiable core that powers analytic nuclear forces and
exchange-correlation-parameter gradients. Everything is a pure function of the
(possibly traced) nuclear ``positions``: the AO integrals, the molecular grid and
the exchange-correlation matrix are all rebuilt from ``positions`` so that
:func:`jax.grad` flows from the total energy down to the coordinates.

Two differentiation paths are provided for the self-consistent density:

* **Implicit (default)** -- the Roothaan/Kohn-Sham step ``D -> SCF_step(D)`` is
  solved by :func:`~opifex.neural.quantum.dft._fixed_point.anderson_fixed_point`,
  which wraps the converged fixed point in :func:`jax.lax.custom_root` using
  :func:`~opifex.core.solver.dense_tangent_solve` by default (or a caller-selected
  matrix-free tangent solver) so the gradient is exact via the implicit function
  theorem (Zhang & Chan 2022, ``arXiv:2207.13836``).
* **Unrolled** -- a fixed number of differentiable SCF steps, used only as a
  sanity cross-check of the implicit gradient in the tests.

The exchange-correlation matrix supports both the LDA (a single density-dependent
potential) and the PBE GGA (the ``v_rho`` and ``v_sigma`` pair contracted with
the AO gradients) functionals.

References:
----------
* X. Zhang, G. K.-L. Chan, *J. Chem. Phys.* **157**, 204801 (2022),
  arXiv:2207.13836 -- implicit differentiation of the SCF fixed point.
* J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).
* R. G. Parr, W. Yang, *Density-Functional Theory of Atoms and Molecules*,
  Oxford (1989), Ch. 7 -- the Kohn-Sham total-energy expression.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct
from jax import Array

from opifex.core.quantum.backend import JaxGaussianBackend
from opifex.core.quantum.basis import AtomicOrbitalBasis  # noqa: TC001
from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.solver.implicit import dense_tangent_solve, loop_status, TangentSolverFn
from opifex.core.solver.status import is_successful
from opifex.neural.quantum.dft._fixed_point import (
    anderson_fixed_point,
    AndersonAcceleration,
    FixedPointSolution,
)
from opifex.neural.quantum.dft.grid import MolecularGridTemplate  # noqa: TC001
from opifex.neural.quantum.dft.xc import (
    lda_energy_density,
    lda_exchange_correlation_potential,
    pbe_energy_density,
    pbe_exchange_correlation_potential,
)
from opifex.neural.quantum.neural_xc import NeuralXCFunctional  # noqa: TC001


class DensitySolve(struct.PyTreeNode):
    """A converged density together with how the solve that produced it went.

    Every field is pytree data, so this is returned through ``jit`` and batched under
    ``vmap`` with one outcome per element. It is what keeps the two forward paths --
    Anderson self-consistency and direct minimisation -- reporting the same measured
    quantities instead of one of them assuming its own success.

    Attributes:
        density: The converged closed-shell AO density matrix.
        n_iterations: Steps the solver actually took, as an ``int32`` array.
        status: Why the solve stopped; see :mod:`opifex.core.solver.status`.
    """

    density: Array
    n_iterations: Array
    status: Array

    @classmethod
    def from_solution(cls, solution: FixedPointSolution, density: Array) -> DensitySolve:
        """Construct a :class:`DensitySolve` from a :class:`FixedPointSolution`.

        Args:
            solution: Completed fixed-point solve carrying ``status`` and ``n_iterations``.
            density: Converged AO density matrix derived from ``solution``.

        Returns:
            The density paired with the solve's step count and status.
        """
        return cls(
            density=density,
            n_iterations=jnp.asarray(solution.n_iterations, jnp.int32),
            status=jnp.asarray(solution.status, jnp.int32),
        )

    @property
    def is_converged(self) -> Array:
        """Whether the solve reached a successful outcome, as a boolean array."""
        return is_successful(self.status)


class Functional(StrEnum):
    """Supported exchange-correlation functionals."""

    LDA = "lda"
    PBE = "pbe"
    NEURAL = "neural"


@dataclass(frozen=True, slots=True)
class NeuralXCSpec:
    """A learned XC functional split into a static graph and traced parameters.

    The implicit-diff SCF and the total-energy functions are pure functions of
    (possibly traced) arrays; an :class:`nnx.Module` cannot be passed through
    them directly. Splitting the :class:`~opifex.neural.quantum.neural_xc.NeuralXCFunctional`
    into its static ``graphdef`` and its parameter ``state`` (the documented
    ``nnx.split`` round-trip) lets the parameters flow as differentiable leaves
    through :func:`~opifex.neural.quantum.dft._fixed_point.anderson_fixed_point`
    and :func:`jax.grad`, so ``dE/dtheta`` is exact via the implicit function
    theorem -- the learned-XC training path.

    Attributes:
        graphdef: The static NNX graph definition of the functional.
        state: The functional's trainable ``nnx.Param`` state (traceable pytree leaves).
        rest_state: Optional non-parameter NNX state (such as ``RngState``) closed over
            outside implicit differentiation.
    """

    graphdef: nnx.GraphDef
    state: nnx.State
    rest_state: nnx.State | None = None

    @classmethod
    def from_functional(cls, functional: NeuralXCFunctional) -> NeuralXCSpec:
        """Split an evaluation view of a :class:`NeuralXCFunctional` into ``Param`` and rest.

        The SCF evaluates the functional, not a dropout sample of it, whatever mode the caller's
        module is in. ``nnx.view(..., deterministic=True)`` gives a view in evaluation mode that
        shares the caller's arrays and leaves the caller's module unchanged; splitting ``nnx.Param``
        from ``rest_state`` keeps non-floating ``RngState`` leaves out of differentiated arguments.
        """
        viewed = nnx.view(functional, deterministic=True, raise_if_not_found=False)
        graphdef, params, rest_state = nnx.split(viewed, nnx.Param, ...)
        return cls(graphdef=graphdef, state=params, rest_state=rest_state)

    def merge(self) -> NeuralXCFunctional:
        """Reconstruct the live :class:`NeuralXCFunctional` from the split."""
        if self.rest_state is None:
            return nnx.merge(self.graphdef, self.state)
        return nnx.merge(self.graphdef, self.state, self.rest_state)


@dataclass(frozen=True, slots=True, kw_only=True)
class _XCResult:
    """Exchange-correlation energy and Fock-matrix contribution."""

    energy: Array
    matrix: Array


def _density_on_grid(density: Array, ao_values: Array, ao_gradients: Array) -> tuple[Array, Array]:
    r"""Electron density and its Cartesian gradient on the quadrature grid.

    With :math:`\rho = \sum_{\mu\nu} D_{\mu\nu}\phi_\mu\phi_\nu`,

    .. math::
        \nabla\rho = 2 \sum_{\mu\nu} D_{\mu\nu}\phi_\mu\,\nabla\phi_\nu

    (using the symmetry of ``D``).

    Returns:
        ``(rho, grad_rho)`` with shapes ``(n_points,)`` and ``(n_points, 3)``.
    """
    rho = jnp.einsum("gm,mn,gn->g", ao_values, density, ao_values)
    grad_rho = 2.0 * jnp.einsum("gm,mn,gnc->gc", ao_values, density, ao_gradients)
    return rho, grad_rho


def _lda_xc(density: Array, ao_values: Array, weights: Array) -> _XCResult:
    """LDA exchange-correlation energy and matrix on the grid."""
    rho = jnp.einsum("gm,mn,gn->g", ao_values, density, ao_values)
    rho = jnp.clip(rho, 0.0, None)
    energy = jnp.sum(weights * rho * lda_energy_density(rho))
    potential = lda_exchange_correlation_potential(rho)
    matrix = jnp.einsum("g,gm,gn->mn", weights * potential, ao_values, ao_values)
    return _XCResult(energy=energy, matrix=matrix)


def _pbe_xc(density: Array, ao_values: Array, ao_gradients: Array, weights: Array) -> _XCResult:
    r"""PBE (GGA) exchange-correlation energy and matrix on the grid.

    The GGA Fock contribution has a local term from :math:`v_\rho` and a
    gradient term from :math:`v_\sigma`:

    .. math::
        V^{xc}_{\mu\nu} = \sum_g w_g \Big[ v_\rho\,\phi_\mu\phi_\nu
            + 2 v_\sigma\,\nabla\rho\cdot
              (\phi_\mu\nabla\phi_\nu + \phi_\nu\nabla\phi_\mu)\Big],

    symmetrised over :math:`\mu\nu` (Pople, Gill & Johnson 1992).
    """
    rho, grad_rho = _density_on_grid(density, ao_values, ao_gradients)
    rho = jnp.clip(rho, 0.0, None)
    sigma = jnp.sum(grad_rho**2, axis=-1)

    energy = jnp.sum(weights * rho * pbe_energy_density(rho, sigma))
    v_rho, v_sigma = pbe_exchange_correlation_potential(rho, sigma)

    local = jnp.einsum("g,gm,gn->mn", weights * v_rho, ao_values, ao_values)
    # gradient term: 2 v_sigma grad_rho . (phi_mu grad phi_nu).
    weighted_grad = (weights * 2.0 * v_sigma)[:, None] * grad_rho  # (g, 3)
    gradient_term = jnp.einsum("gc,gm,gnc->mn", weighted_grad, ao_values, ao_gradients)
    matrix = local + gradient_term + gradient_term.T
    return _XCResult(energy=energy, matrix=matrix)


def _neural_xc(
    density: Array,
    ao_values: Array,
    ao_gradients: Array,
    weights: Array,
    neural_spec: NeuralXCSpec,
) -> _XCResult:
    r"""Learned (GGA-form) exchange-correlation energy and matrix on the grid.

    Identical assembly to :func:`_pbe_xc` but with the energy density and the
    :math:`(v_\rho, v_\sigma)` potential pair coming from the merged neural
    functional. The neural parameters enter through ``neural_spec.state`` so the
    energy is differentiable with respect to them (the learned-XC training path).
    """
    functional = neural_spec.merge()
    rho, grad_rho = _density_on_grid(density, ao_values, ao_gradients)
    rho = jnp.clip(rho, 0.0, None)
    sigma = jnp.sum(grad_rho**2, axis=-1)

    energy = jnp.sum(weights * rho * functional.energy_density_from_sigma(rho, sigma))
    v_rho, v_sigma = functional.xc_potential_components(rho, sigma)

    local = jnp.einsum("g,gm,gn->mn", weights * v_rho, ao_values, ao_values)
    weighted_grad = (weights * 2.0 * v_sigma)[:, None] * grad_rho  # (g, 3)
    gradient_term = jnp.einsum("gc,gm,gnc->mn", weighted_grad, ao_values, ao_gradients)
    matrix = local + gradient_term + gradient_term.T
    return _XCResult(energy=energy, matrix=matrix)


def _build_xc(
    density: Array,
    ao_values: Array,
    ao_gradients: Array,
    weights: Array,
    functional: Functional,
    neural_spec: NeuralXCSpec | None = None,
) -> _XCResult:
    """Dispatch the exchange-correlation build on the functional."""
    if functional is Functional.NEURAL:
        if neural_spec is None:
            raise ValueError("Functional.NEURAL requires a NeuralXCSpec")
        return _neural_xc(density, ao_values, ao_gradients, weights, neural_spec)
    if functional is Functional.PBE:
        return _pbe_xc(density, ao_values, ao_gradients, weights)
    return _lda_xc(density, ao_values, weights)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Integrals:
    """The position-dependent quantities one SCF needs, built from ``positions``."""

    overlap: Array
    core_hamiltonian: Array
    eri: Array
    nuclear_repulsion: Array
    orthogonaliser: Array
    ao_values: Array
    ao_gradients: Array
    grid_weights: Array


def _symmetric_orthogonaliser(overlap: Array) -> Array:
    r"""Lowdin :math:`S^{-1/2}` via eigendecomposition of the overlap matrix."""
    eigenvalues, eigenvectors = jnp.linalg.eigh(overlap)
    return eigenvectors @ jnp.diag(eigenvalues ** (-0.5)) @ eigenvectors.T


def assemble_integrals(
    positions: Array,
    system: MolecularSystem,
    basis: AtomicOrbitalBasis,
    grid_template: MolecularGridTemplate,
    functional: Functional,
) -> _Integrals:
    """Build all position-dependent integrals/grid quantities from ``positions``.

    Args:
        positions: Nuclear positions in Bohr [Shape: (n_atoms, 3)].
        system: The molecular system (provides charges; positions are overridden).
        basis: The reference AO basis (recentred onto ``positions``).
        grid_template: The static molecular-grid template.
        functional: The exchange-correlation functional.

    Returns:
        The assembled :class:`_Integrals`.
    """
    moved_system = MolecularSystem(
        atomic_numbers=system.atomic_numbers,
        positions=positions,
        charge=system.charge,
        multiplicity=system.multiplicity,
        basis_set=system.basis_set,
    )
    moved_basis = basis.with_positions(positions)
    backend = JaxGaussianBackend(moved_system, moved_basis)

    overlap = backend.overlap()
    core_hamiltonian = backend.core_hamiltonian()
    eri = backend.electron_repulsion()
    nuclear_repulsion = backend.nuclear_repulsion()
    orthogonaliser = _symmetric_orthogonaliser(overlap)

    grid = grid_template.build(positions)
    if functional in (Functional.PBE, Functional.NEURAL):
        ao_values, ao_gradients = moved_basis.evaluate_with_gradients(grid.points)
    else:
        ao_values = moved_basis.evaluate(grid.points)
        ao_gradients = jnp.zeros((*ao_values.shape, 3), dtype=ao_values.dtype)
    return _Integrals(
        overlap=overlap,
        core_hamiltonian=core_hamiltonian,
        eri=eri,
        nuclear_repulsion=nuclear_repulsion,
        orthogonaliser=orthogonaliser,
        ao_values=ao_values,
        ao_gradients=ao_gradients,
        grid_weights=grid.weights,
    )


def _coulomb_matrix(eri: Array, density: Array) -> Array:
    r"""Coulomb matrix ``J_{mn} = sum_{ls} (mn|ls) D_{ls}``."""
    return jnp.einsum("mnls,ls->mn", eri, density)


def _density_from_fock(
    fock: Array, orthogonaliser: Array, n_occupied: int
) -> tuple[Array, Array, Array]:
    """Solve ``FC=SCe`` in the orthonormal basis; build the closed-shell density.

    Returns:
        ``(density, coefficients, orbital_energies)``.
    """
    fock_orthonormal = orthogonaliser.T @ fock @ orthogonaliser
    orbital_energies, orthonormal_coeffs = jnp.linalg.eigh(fock_orthonormal)
    coefficients = orthogonaliser @ orthonormal_coeffs
    occupied = coefficients[:, :n_occupied]
    density = 2.0 * occupied @ occupied.T
    return density, coefficients, orbital_energies


def build_fock(
    density: Array,
    integrals: _Integrals,
    functional: Functional,
    neural_spec: NeuralXCSpec | None = None,
) -> tuple[Array, Array, _XCResult]:
    """Build ``F = h_core + J + V_xc`` and return ``(fock, coulomb, xc)``."""
    coulomb = _coulomb_matrix(integrals.eri, density)
    xc = _build_xc(
        density,
        integrals.ao_values,
        integrals.ao_gradients,
        integrals.grid_weights,
        functional,
        neural_spec,
    )
    fock = integrals.core_hamiltonian + coulomb + xc.matrix
    return fock, coulomb, xc


def total_energy(
    density: Array,
    integrals: _Integrals,
    functional: Functional,
    neural_spec: NeuralXCSpec | None = None,
) -> Array:
    """Proper Kohn-Sham total energy for a given density and integrals."""
    coulomb = _coulomb_matrix(integrals.eri, density)
    xc = _build_xc(
        density,
        integrals.ao_values,
        integrals.ao_gradients,
        integrals.grid_weights,
        functional,
        neural_spec,
    )
    one_electron = jnp.sum(density * integrals.core_hamiltonian)
    hartree = 0.5 * jnp.sum(density * coulomb)
    return one_electron + hartree + xc.energy + integrals.nuclear_repulsion


def _scf_step(
    density: Array,
    integrals: _Integrals,
    functional: Functional,
    n_occupied: int,
    neural_spec: NeuralXCSpec | None = None,
) -> Array:
    """One Roothaan/Kohn-Sham step ``D -> D'`` (the SCF fixed-point map)."""
    fock, _, _ = build_fock(density, integrals, functional, neural_spec)
    new_density, _, _ = _density_from_fock(fock, integrals.orthogonaliser, n_occupied)
    return new_density


def scf_fixed_point(
    integrals: _Integrals,
    functional: Functional,
    n_occupied: int,
    *,
    tolerance: float,
    max_steps: int,
    neural_spec: NeuralXCSpec | None = None,
    initial_density: Array | None = None,
    tangent_solver: TangentSolverFn = dense_tangent_solve,
) -> FixedPointSolution:
    """Solve the Kohn-Sham density fixed point with pure-JAX Anderson acceleration.

    The single self-consistency engine shared by the differentiable energy path
    (:func:`converged_density_implicit`) and the forward solve
    (:meth:`opifex.neural.quantum.dft.scf.SCFSolver.solve`). Returns the full
    :class:`~opifex.neural.quantum.dft._fixed_point.FixedPointSolution` so callers
    can read the converged density (``.value``), iteration count
    (``.n_iterations``), and convergence status (``.status``). Implicit
    differentiation via :func:`jax.lax.custom_root` yields exact
    implicit-function-theorem gradients regardless of the forward iteration path.

    Args:
        integrals: The molecular integrals (overlap, core Hamiltonian, ERI, XC
            grid) the Fock build and orthogonalisation are computed from.
        functional: The exchange-correlation functional driving the Fock build.
        n_occupied: Number of doubly-occupied orbitals (electrons // 2).
        tolerance: Relative and absolute residual tolerance for convergence.
        max_steps: Maximum number of Anderson fixed-point iterations.
        neural_spec: Optional learned-XC specification for the ``NEURAL`` path.
        initial_density: Optional closed-shell density to seed the iteration.
            Defaults to the core-Hamiltonian density.
        tangent_solver: Linear solver ``(lin_op, rhs) -> v`` for implicit
            differentiation; defaults to :func:`~opifex.core.solver.dense_tangent_solve`.

    Returns:
        :class:`~opifex.neural.quantum.dft._fixed_point.FixedPointSolution`
        carrying ``.value``, ``.n_iterations``, and ``.status``.
    """
    if initial_density is None:
        initial = _density_from_fock(
            integrals.core_hamiltonian, integrals.orthogonaliser, n_occupied
        )[0]
    else:
        initial = initial_density
    solver = AndersonAcceleration(rtol=tolerance, atol=tolerance)

    def step(density: Array, _: None) -> Array:
        """Evaluate one Kohn-Sham SCF map iteration."""
        return _scf_step(density, integrals, functional, n_occupied, neural_spec)

    return anderson_fixed_point(
        step,
        solver,
        initial,
        max_steps=max_steps,
        tangent_solver=tangent_solver,
    )


def converged_density_implicit(
    integrals: _Integrals,
    functional: Functional,
    n_occupied: int,
    *,
    tolerance: float,
    max_steps: int,
    neural_spec: NeuralXCSpec | None = None,
    tangent_solver: TangentSolverFn = dense_tangent_solve,
) -> Array:
    r"""Converged density matrix via the implicit-function-theorem fixed point.

    Solves the Roothaan step :math:`D \to f(D)` with
    :class:`~opifex.neural.quantum.dft._fixed_point.AndersonAcceleration` and
    differentiates the bare step ``f`` at the converged fixed point via
    :func:`jax.lax.custom_root` + ``tangent_solver`` (defaulting to
    :func:`~opifex.core.solver.dense_tangent_solve`).
    """
    return scf_fixed_point(
        integrals,
        functional,
        n_occupied,
        tolerance=tolerance,
        max_steps=max_steps,
        neural_spec=neural_spec,
        tangent_solver=tangent_solver,
    ).value


def converged_density_unrolled(
    integrals: _Integrals,
    functional: Functional,
    n_occupied: int,
    *,
    n_steps: int,
    neural_spec: NeuralXCSpec | None = None,
) -> Array:
    """Converged density via a fixed number of differentiable SCF steps.

    Backpropagation flows through every step; used only to cross-check the
    implicit-diff gradient.
    """
    density = _density_from_fock(integrals.core_hamiltonian, integrals.orthogonaliser, n_occupied)[
        0
    ]
    for _ in range(n_steps):
        density = _scf_step(density, integrals, functional, n_occupied, neural_spec)
    return density


def _cayley_orthonormal(parameters: Array, reference: Array, n_orbitals: int) -> Array:
    r"""Orthonormal (in the overlap metric) occupied coefficients via QR.

    Direct-minimisation parametrisation (jrystal / DWD, arXiv:2411.05033): a free
    matrix is mapped to an orthonormal set of occupied orbitals. Working in the
    Lowdin-orthonormal basis (``reference`` is :math:`S^{-1/2}`), a thin QR of the
    free parameters yields orthonormal orthonormal-basis columns; back-transform
    with ``reference`` to recover overlap-orthonormal AO coefficients.
    """
    q_factor, _ = jnp.linalg.qr(parameters)
    occupied_orthonormal = q_factor[:, :n_orbitals]
    return reference @ occupied_orthonormal


class _DirectMinCarry(struct.PyTreeNode):
    """Internal carry for L-BFGS direct energy minimization."""

    params: Array
    opt_state: Any
    value: Array
    grad: Array
    step_count: Array
    converged: Array
    has_nonfinite: Array


_DIRECT_MAX_STEP_NORM = 0.2


def converged_density_direct(
    integrals: _Integrals,
    functional: Functional,
    n_occupied: int,
    n_ao: int,
    *,
    tolerance: float,
    max_steps: int,
    neural_spec: NeuralXCSpec | None = None,
) -> DensitySolve:
    r"""Converged density via direct L-BFGS energy minimisation over orbitals.

    SCF-free alternative (jrystal / DWD, ``arXiv:2411.05033``): minimises the
    Kohn-Sham energy directly over a QR-orthonormalised coefficient matrix using
    pure-JAX :func:`optax.lbfgs` + :func:`optax.scale_by_zoom_linesearch` +
    :func:`optax.value_and_grad_from_state` inside :func:`jax.lax.while_loop`.

    Args:
        integrals: The molecular integrals the energy is built from.
        functional: The exchange-correlation functional.
        n_occupied: Number of doubly-occupied orbitals (electrons // 2).
        n_ao: Number of atomic orbitals, sizing the coefficient matrix.
        tolerance: Relative and absolute convergence tolerance for the minimiser.
        max_steps: Maximum number of minimisation steps.
        neural_spec: Optional learned-XC specification for the ``NEURAL`` path.

    Returns:
        The density paired with the minimiser's step count and status.
    """
    reference = integrals.orthogonaliser
    dtype = reference.dtype
    tol_f = jnp.asarray(tolerance, dtype=dtype)

    def energy_fn(parameters: Array) -> Array:
        """Compute Kohn-Sham total energy from free orbital parameters."""
        coefficients = _cayley_orthonormal(parameters, reference, n_occupied)
        density = 2.0 * coefficients @ coefficients.T
        return total_energy(density, integrals, functional, neural_spec)

    max_ls = min(20, max(1, max_steps))
    optimizer = optax.lbfgs(
        memory_size=10,
        scale_init_precond=True,
        linesearch=optax.chain(
            optax.clip_by_global_norm(_DIRECT_MAX_STEP_NORM),
            optax.scale_by_zoom_linesearch(
                max_linesearch_steps=max_ls,
                max_learning_rate=1.0,
                initial_guess_strategy="one",
            ),
        ),
    )
    val_and_grad_fn = optax.value_and_grad_from_state(energy_fn)

    initial = jnp.eye(n_ao, dtype=dtype)
    init_opt_state = optimizer.init(initial)
    raw_init_val, raw_init_grad = val_and_grad_fn(initial, state=init_opt_state)
    init_val = jnp.asarray(raw_init_val, dtype=dtype)
    init_grad = cast("Array", raw_init_grad)
    init_scale = jnp.maximum(jnp.asarray(1.0, dtype=dtype), jnp.abs(init_val))
    init_nonfinite = ~jnp.isfinite(init_val) | ~jnp.all(jnp.isfinite(init_grad))
    init_conv = (~init_nonfinite) & (jnp.max(jnp.abs(init_grad)) <= tol_f * init_scale)

    def cond_fn(carry: _DirectMinCarry) -> Array:
        """Check whether direct L-BFGS orbital minimization should continue."""
        return (carry.step_count < max_steps) & (~carry.converged) & (~carry.has_nonfinite)

    def body_fn(carry: _DirectMinCarry) -> _DirectMinCarry:
        """Advance one L-BFGS orbital minimization step with zoom line search."""
        raw_updates, next_opt_state = optimizer.update(
            carry.grad,
            carry.opt_state,
            carry.params,
            value=carry.value,  # pyright: ignore[reportCallIssue]
            grad=carry.grad,  # pyright: ignore[reportCallIssue]
            value_fn=energy_fn,  # pyright: ignore[reportCallIssue]
        )
        updates = cast("Array", raw_updates)
        cand_params = cast("Array", optax.apply_updates(carry.params, updates))
        raw_cand_val, raw_cand_grad = val_and_grad_fn(cand_params, state=next_opt_state)
        cand_val = jnp.asarray(raw_cand_val, dtype=dtype)
        cand_grad = cast("Array", raw_cand_grad)
        step_nonfinite = ~jnp.isfinite(cand_val) | ~jnp.all(jnp.isfinite(cand_grad))

        grad_inf = jnp.max(jnp.abs(cand_grad))
        val_scale = jnp.maximum(jnp.asarray(1.0, dtype=dtype), jnp.abs(cand_val))
        grad_ok = grad_inf <= tol_f * val_scale

        next_params = jnp.where(step_nonfinite, carry.params, cand_params)
        return _DirectMinCarry(
            params=next_params,
            opt_state=next_opt_state,
            value=cand_val,
            grad=cand_grad,
            step_count=carry.step_count + jnp.asarray(1, dtype=jnp.int32),
            converged=(~step_nonfinite) & grad_ok,
            has_nonfinite=carry.has_nonfinite | step_nonfinite,
        )

    init_carry = _DirectMinCarry(
        params=initial,
        opt_state=init_opt_state,
        value=init_val,
        grad=init_grad,
        step_count=jnp.asarray(0, dtype=jnp.int32),
        converged=init_conv,
        has_nonfinite=init_nonfinite,
    )
    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    status = loop_status(
        has_nonfinite=final_carry.has_nonfinite,
        converged=final_carry.converged,
    )
    coefficients = _cayley_orthonormal(final_carry.params, reference, n_occupied)
    return DensitySolve(
        density=2.0 * coefficients @ coefficients.T,
        n_iterations=final_carry.step_count,
        status=status,
    )


__all__ = [
    "DensitySolve",
    "Functional",
    "NeuralXCSpec",
    "assemble_integrals",
    "build_fock",
    "converged_density_direct",
    "converged_density_implicit",
    "converged_density_unrolled",
    "scf_fixed_point",
    "total_energy",
]

r"""Native JAX Gaussian-integral engine (McMurchie-Davidson).

This module computes the one- and two-electron molecular integrals over the
contracted Cartesian-Gaussian basis of
:class:`~opifex.core.quantum.basis.AtomicOrbitalBasis` using the
McMurchie-Davidson (MMD) Hermite-Gaussian expansion. Everything is written in
JAX (``jnp``) so the integral tensors are differentiable with respect to the
nuclear positions and the whole pipeline is ``jit``-compatible.

The :class:`QCBackend` Protocol is the swappable seam (dependency inversion):
:class:`JaxGaussianBackend` is the native in-tree implementation, and a PySCF
adapter exists only in the tests as a validation oracle.

Method and references
---------------------
The implementation follows the standard McMurchie-Davidson scheme:

* L. E. McMurchie, E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978).
* T. Helgaker, P. Jorgensen, J. Olsen, *Molecular Electronic-Structure Theory*,
  Wiley (2000), Ch. 9 -- the Hermite expansion coefficients :math:`E_t^{ij}`
  (eq. 9.5.6), the overlap (9.5.41), kinetic (9.3.40-9.3.43) and the
  Hermite-Coulomb integrals :math:`R_{tuv}` (9.9.18-9.9.20) used for the
  nuclear-attraction and electron-repulsion integrals.
* The Boys function :math:`F_n(x)=\int_0^1 t^{2n} e^{-x t^2}\,dt` is evaluated in
  :mod:`opifex.core.quantum._boys` with a three-branch ``jnp.select`` (analytic
  ``x=0`` limit, ascending series, large-``x`` asymptotic; the MESS
  ``gammanu_select`` strategy) so it is accurate and AD-safe in both limits.

The full AO integral tensors are assembled by the batched flat-primitive harness
(:mod:`opifex.core.quantum._flat_harness` over
:class:`opifex.core.quantum.basis.FlatPrimitives`, the ``graphcore-research/mess``
batching pattern -- one ``vmap`` + ``segment_sum`` pass, ``jit``-compilable). The
recurrence indexing is cross-checked against the Joshua Goings "Integrals"
write-up of McMurchie-Davidson (which follows Helgaker); every integral is
validated against an eager per-primitive reference to ~1e-10 and against PySCF to
~1e-8 in the test suite.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

from opifex.core.quantum._boys import boys_function, boys_vector
from opifex.core.quantum._flat_harness import (
    electron_repulsion_tensor,
    one_electron_matrices,
)
from opifex.core.quantum.basis import AtomicOrbitalBasis  # noqa: TC001
from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001


@runtime_checkable
class QCBackend(Protocol):
    """Quantum-chemistry integral backend seam.

    Implementations provide the AO integral tensors and the nuclear-repulsion
    energy for a fixed :class:`MolecularSystem` / :class:`AtomicOrbitalBasis`.
    """

    def overlap(self) -> Array:
        """Return the AO overlap matrix ``S`` [Shape: (n_ao, n_ao)]."""
        ...

    def core_hamiltonian(self) -> Array:
        """Return the core Hamiltonian ``T + V`` [Shape: (n_ao, n_ao)]."""
        ...

    def electron_repulsion(self) -> Array:
        """Return the ERI tensor ``(ij|kl)`` [Shape: (n_ao,)*4] in chemist order."""
        ...

    def nuclear_repulsion(self) -> Array:
        """Return the scalar nuclear-repulsion energy ``E_nn``."""
        ...


def hermite_expansion(
    l_a: int,
    l_b: int,
    distance: Array,
    exp_a: Array,
    exp_b: Array,
) -> Array:
    r"""Hermite expansion coefficients :math:`E_t^{l_a l_b}` for one Cartesian axis.

    Implements the McMurchie-Davidson recurrence (Helgaker eq. 9.5.6-9.5.8)

    .. math::
        E_t^{i+1,j} = \frac{1}{2p} E_{t-1}^{ij}
                    + X_{PA} E_t^{ij} + (t+1) E_{t+1}^{ij},
        E_t^{i,j+1} = \frac{1}{2p} E_{t-1}^{ij}
                    + X_{PB} E_t^{ij} + (t+1) E_{t+1}^{ij},

    with :math:`E_0^{00} = \exp(-\mu X_{AB}^2)`, :math:`p=a+b`,
    :math:`\mu = ab/p`, :math:`X_{PA} = -b\,X_{AB}/p`,
    :math:`X_{PB} = a\,X_{AB}/p`.

    Args:
        l_a: Angular-momentum power on centre A for this axis.
        l_b: Angular-momentum power on centre B for this axis.
        distance: Signed separation ``X_AB = X_A - X_B`` along this axis.
        exp_a: Primitive exponent ``a`` on centre A.
        exp_b: Primitive exponent ``b`` on centre B.

    Returns:
        Array of length ``l_a + l_b + 1`` holding ``E_0, ..., E_{l_a+l_b}``.
    """
    total_p = exp_a + exp_b
    reduced = exp_a * exp_b / total_p
    pa = -exp_b * distance / total_p
    pb = exp_a * distance / total_p
    t_max = l_a + l_b

    # table[i, j, t] = E_t^{ij}; sized (l_a+1, l_b+1, t_max+1).
    table = jnp.zeros((l_a + 1, l_b + 1, t_max + 1), dtype=total_p.dtype)
    table = table.at[0, 0, 0].set(jnp.exp(-reduced * distance**2))

    half_inv_p = 1.0 / (2.0 * total_p)

    def shifted(tab: Array, i: int, j: int, offset: int) -> Array:
        """Return ``E_{t+offset}^{ij}`` aligned to index ``t``, zero-padded."""
        column = tab[i, j]
        if offset == 0:
            return column
        if offset == -1:
            return jnp.concatenate([jnp.zeros(1, column.dtype), column[:-1]])
        return jnp.concatenate([column[1:], jnp.zeros(1, column.dtype)])

    t_indices = jnp.arange(t_max + 1, dtype=total_p.dtype)

    # Increment i (centre A) with j = 0.
    for i in range(l_a):
        new_column = (
            half_inv_p * shifted(table, i, 0, -1)
            + pa * shifted(table, i, 0, 0)
            + (t_indices + 1.0) * shifted(table, i, 0, +1)
        )
        table = table.at[i + 1, 0].set(new_column)

    # Increment j (centre B) for every i.
    for i in range(l_a + 1):
        for j in range(l_b):
            new_column = (
                half_inv_p * shifted(table, i, j, -1)
                + pb * shifted(table, i, j, 0)
                + (t_indices + 1.0) * shifted(table, i, j, +1)
            )
            table = table.at[i, j + 1].set(new_column)

    return table[l_a, l_b]


def hermite_coulomb(max_total: int, alpha: Array, separation: Array) -> Array:
    r"""Hermite-Coulomb auxiliary integrals :math:`R_{tuv}`.

    Implements Helgaker eq. 9.9.18-9.9.20:

    .. math::
        R_{tuv}^n = \begin{cases}
            (-2\alpha)^n F_n(\alpha R_{PC}^2) & t=u=v=0 \\
            t\,R_{t-1,u,v}^{n+1} + X_{PC} R_{t-1,u,v}^{n+1} & \dots
        \end{cases}

    Args:
        max_total: Maximum of ``t + u + v`` required.
        alpha: Combined exponent ``alpha`` (``p`` for V, ``pq/(p+q)`` for ERI).
        separation: ``P - C`` vector [Shape: (3,)].

    Returns:
        Array ``R[t, u, v]`` of shape ``(max_total+1,)*3`` (entries with
        ``t+u+v > max_total`` are left at zero).
    """
    dx, dy, dz = separation[0], separation[1], separation[2]
    r2 = dx * dx + dy * dy + dz * dz
    boys = boys_vector(max_total, alpha * r2)
    powers = (-2.0 * alpha) ** jnp.arange(max_total + 1, dtype=alpha.dtype)
    # aux[n] = R_000^n = (-2 alpha)^n F_n.
    aux = powers * boys

    size = max_total + 1
    # r_table[t, u, v, n] built up the standard MMD way.
    r_table = jnp.zeros((size, size, size, size), dtype=alpha.dtype)
    r_table = r_table.at[0, 0, 0, :].set(aux)

    def lower(tab: Array, t: int, u: int, v: int) -> Array:
        """``R_{t,u,v}^{n+1}`` aligned to index ``n`` (last entry padded)."""
        return jnp.concatenate([tab[t, u, v, 1:], jnp.zeros(1, tab.dtype)])

    for total in range(1, max_total + 1):
        for t in range(total + 1):
            for u in range(total + 1 - t):
                v = total - t - u
                if t > 0:
                    term = (t - 1) * lower(r_table, t - 2, u, v) if t >= 2 else 0.0
                    new = term + dx * lower(r_table, t - 1, u, v)
                elif u > 0:
                    term = (u - 1) * lower(r_table, t, u - 2, v) if u >= 2 else 0.0
                    new = term + dy * lower(r_table, t, u - 1, v)
                else:
                    term = (v - 1) * lower(r_table, t, u, v - 2) if v >= 2 else 0.0
                    new = term + dz * lower(r_table, t, u, v - 1)
                r_table = r_table.at[t, u, v, :].set(new)

    return r_table[:, :, :, 0]


class JaxGaussianBackend:
    """Native McMurchie-Davidson AO integral backend.

    Builds the overlap, kinetic, nuclear-attraction, electron-repulsion and
    nuclear-repulsion quantities for a fixed molecular system and basis. All
    tensors are JAX arrays differentiable with respect to nuclear positions.

    Args:
        system: The molecular system (nuclear positions and charges).
        basis: The contracted-Gaussian AO basis.
    """

    def __init__(self, system: MolecularSystem, basis: AtomicOrbitalBasis) -> None:
        """Store the system and basis and build the flat-primitive view."""
        self._system = system
        self._basis = basis
        self._flat = basis.flat_primitives()
        self._nuclear_positions = jnp.asarray(system.positions)
        self._nuclear_charges = jnp.asarray(system.atomic_numbers).astype(
            self._nuclear_positions.dtype
        )

    @property
    def n_atomic_orbitals(self) -> int:
        """Number of atomic orbitals."""
        return self._basis.n_atomic_orbitals

    def _assemble_one_electron(self) -> tuple[Array, Array, Array]:
        """Assemble ``(S, T, V)`` with one batched ``vmap`` over primitive pairs.

        Uses the flat-primitive harness (MESS ``integrate_dense`` pattern): a
        single ``vmap`` over the upper-triangular primitive pairs followed by a
        double ``segment_sum`` contraction to AOs -- no Python shell loop, so the
        whole build traces once and ``jit``-compiles.
        """
        return one_electron_matrices(self._flat, self._nuclear_positions, self._nuclear_charges)

    def overlap(self) -> Array:
        """Return the AO overlap matrix ``S``."""
        overlap, _, _ = self._assemble_one_electron()
        return overlap

    def kinetic(self) -> Array:
        """Return the AO kinetic-energy matrix ``T``."""
        _, kinetic, _ = self._assemble_one_electron()
        return kinetic

    def nuclear_attraction(self) -> Array:
        """Return the AO nuclear-attraction matrix ``V``."""
        _, _, nuclear = self._assemble_one_electron()
        return nuclear

    def core_hamiltonian(self) -> Array:
        """Return the core Hamiltonian ``T + V``."""
        _, kinetic, nuclear = self._assemble_one_electron()
        return kinetic + nuclear

    def electron_repulsion(self) -> Array:
        """Return the ERI tensor ``(ij|kl)`` in chemist notation.

        Built with the flat-primitive harness (MESS ``eri_basis`` pattern): the
        8-fold-unique AO quartets are evaluated in chunked ``vmap`` traces over
        primitive quartets, contracted with ``segment_sum``, then scattered into
        the eight permutation-equivalent dense positions. One trace per chunk --
        the whole tensor ``jit``-compiles (the eager ``n_shells**4`` loop did
        not).
        """
        return electron_repulsion_tensor(self._flat)

    def nuclear_repulsion(self) -> Array:
        """Return the nuclear-repulsion energy ``sum_{A<B} Z_A Z_B / R_AB``.

        The squared inter-nuclear distance is computed only on the strict upper
        triangle (``i < j``); the diagonal, where ``R_AB = 0``, is replaced with
        a finite value *before* the square root so the gradient with respect to
        the nuclear positions stays finite (the standard ``jnp.where`` masked
        ``sqrt`` is gradient-unsafe at zero -- needed for analytic forces).
        """
        positions = self._nuclear_positions
        charges = self._nuclear_charges
        diff = positions[:, None, :] - positions[None, :, :]
        squared = jnp.sum(diff**2, axis=-1)
        charge_products = charges[:, None] * charges[None, :]
        n_atoms = positions.shape[0]
        upper = jnp.triu(jnp.ones((n_atoms, n_atoms), dtype=bool), k=1)
        # Guard the squared distance before the sqrt so no NaN gradient leaks in
        # from the (masked-out) diagonal / lower triangle.
        safe_squared = jnp.where(upper, squared, 1.0)
        distances = jnp.sqrt(safe_squared)
        contributions = jnp.where(upper, charge_products / distances, 0.0)
        return jnp.sum(contributions)


__all__ = [
    "JaxGaussianBackend",
    "QCBackend",
    "boys_function",
    "boys_vector",
    "hermite_coulomb",
    "hermite_expansion",
]

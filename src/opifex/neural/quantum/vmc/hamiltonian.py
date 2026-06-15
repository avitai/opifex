r"""Molecular local energy for variational Monte Carlo.

The local energy of a (real, log-domain) wavefunction at an electron
configuration ``r`` is

.. math::

    E_{\mathrm{loc}}(r) = -\tfrac12 \frac{\nabla^2 \psi}{\psi}(r) + V(r)
    = -\tfrac12 \Big(\nabla^2 \log|\psi| + \|\nabla \log|\psi|\|^2\Big) + V(r),

with the molecular Coulomb potential

.. math::

    V(r) = \underbrace{\sum_{i<j} \frac{1}{r_{ij}}}_{V_{ee}}
    - \underbrace{\sum_{i,A} \frac{Z_A}{r_{iA}}}_{V_{eN}}
    + \underbrace{\sum_{A<B} \frac{Z_A Z_B}{R_{AB}}}_{V_{NN}}.

The kinetic term uses either the native forward-Laplacian or the
``jvp``-over-``grad`` oracle (:mod:`~opifex.neural.quantum.vmc.laplacian`). All
functions are pure JAX -- ``jit`` / ``grad`` / ``vmap`` clean -- and follow the
DeepMind FermiNet ``hamiltonian.py`` formulae. The potential is cusp-safe: the
electron-electron distance diagonal is masked in
:func:`~._blocks.construct_input_features`, and same-site nuclear pairs never
appear because only the strict upper triangle is summed.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float  # noqa: TC002

from opifex.neural.quantum.vmc.laplacian import (
    forward_laplacian,
    jvp_grad_laplacian,
)


KineticMethod = Literal["forward", "oracle"]
_LAPLACIANS: dict[str, Callable[..., tuple[Array, Array, Array]]] = {
    "forward": forward_laplacian,
    "oracle": jvp_grad_laplacian,
}


def potential_electron_electron(
    positions: Float[Array, "nelectron ndim"],
) -> Array:
    """Electron-electron Coulomb repulsion ``sum_{i<j} 1 / r_ij``."""
    n = positions.shape[0]
    displacement = positions[:, None, :] - positions[None, :, :]
    # Mask the diagonal inside the norm so ||0|| has no undefined gradient.
    distance = jnp.linalg.norm(displacement + jnp.eye(n)[..., None], axis=-1)
    inverse = jnp.triu(1.0 / distance, k=1)
    return jnp.sum(inverse)


def potential_electron_nuclear(
    positions: Float[Array, "nelectron ndim"],
    atoms: Float[Array, "natom ndim"],
    charges: Float[Array, " natom"],
) -> Array:
    """Electron-nucleus Coulomb attraction ``-sum_{i,A} Z_A / r_iA``."""
    distance = jnp.linalg.norm(positions[:, None, :] - atoms[None, :, :], axis=-1)
    return -jnp.sum(charges[None, :] / distance)


def potential_nuclear_nuclear(
    atoms: Float[Array, "natom ndim"],
    charges: Float[Array, " natom"],
) -> Array:
    """Nucleus-nucleus Coulomb repulsion ``sum_{A<B} Z_A Z_B / R_AB``."""
    natom = atoms.shape[0]
    distance = jnp.linalg.norm(
        atoms[:, None, :] - atoms[None, :, :] + jnp.eye(natom)[..., None], axis=-1
    )
    pair_charges = charges[:, None] * charges[None, :]
    return jnp.sum(jnp.triu(pair_charges / distance, k=1))


def potential_energy(
    positions: Float[Array, "nelectron ndim"],
    *,
    atoms: Float[Array, "natom ndim"],
    charges: Float[Array, " natom"],
) -> Array:
    """Total molecular Coulomb potential ``V_ee + V_eN + V_NN``."""
    return (
        potential_electron_electron(positions)
        + potential_electron_nuclear(positions, atoms, charges)
        + potential_nuclear_nuclear(atoms, charges)
    )


def local_kinetic_energy(
    log_abs: Callable[[Array], Array],
    positions: Float[Array, "nelectron ndim"],
    *,
    method: KineticMethod = "forward",
) -> Array:
    r"""Local kinetic energy ``-1/2 (nabla^2 log|psi| + |nabla log|psi||^2)``.

    Args:
        log_abs: ``positions -> log|psi|`` for a single walker.
        positions: Electron coordinates of shape ``(nelectron, ndim)``.
        method: ``"forward"`` for the native forward-Laplacian (default) or
            ``"oracle"`` for the ``jvp``-over-``grad`` reference.

    Returns:
        The scalar local kinetic energy.
    """
    _, laplacian, gradient = _LAPLACIANS[method](log_abs, positions)
    return -0.5 * (laplacian + jnp.sum(gradient**2))


def local_energy(
    log_abs: Callable[[Array], Array],
    *,
    atoms: Float[Array, "natom ndim"],
    charges: Float[Array, " natom"],
    method: KineticMethod = "forward",
) -> Callable[[Array], Array]:
    r"""Build the local-energy function ``E_loc(positions)`` for a wavefunction.

    Args:
        log_abs: ``positions -> log|psi|`` for a single walker.
        atoms: Nuclear coordinates of shape ``(natom, ndim)``.
        charges: Nuclear charges of shape ``(natom,)``.
        method: Kinetic-energy Laplacian method (see
            :func:`local_kinetic_energy`).

    Returns:
        A pure function mapping a single walker's positions to its scalar local
        energy. ``vmap`` it over walkers for a batch.
    """
    atoms = jnp.asarray(atoms)
    charges = jnp.asarray(charges)

    def energy_fn(positions: Array) -> Array:
        kinetic = local_kinetic_energy(log_abs, positions, method=method)
        potential = potential_energy(positions, atoms=atoms, charges=charges)
        return kinetic + potential

    return energy_fn


__all__ = [
    "KineticMethod",
    "local_energy",
    "local_kinetic_energy",
    "potential_electron_electron",
    "potential_electron_nuclear",
    "potential_energy",
    "potential_nuclear_nuclear",
]

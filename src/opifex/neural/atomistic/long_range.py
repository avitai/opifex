r"""Long-range electrostatics as a pluggable add-on (Latent Ewald Summation).

This module implements the §1.4 *long-range-as-pluggable-add-on* seam for the
atomistic models: a per-atom *latent charge* readout plus a Coulomb long-range
energy, returned as an additive correction ``{"long_range_energy": ...}`` on top
of a short-range MLIP energy.

Method
------
**Latent Ewald Summation (LES)** -- Cheng 2025, "Latent Ewald summation for
machine learning of long-range interactions" (arXiv:2408.15165; see also
arXiv:2502.04668, arXiv:2512.18029 in the surrounding landscape). LES learns
*hidden* per-atom charges :math:`q_i` from local descriptors and evaluates the
electrostatic energy of those charges by Ewald summation, *without* charges
supervision. Here the latent charges are produced by the same
total-charge-conserving readout used by
:class:`~opifex.neural.atomistic.heads.charge.ChargeHead` /
:class:`~opifex.neural.atomistic.heads.dipole.DipoleHead`
(:func:`~opifex.neural.atomistic.heads.charge.conserve_total_charge`).

**Ewald summation** -- the periodic Coulomb energy is split into a
short-ranged real-space sum (``erfc``), a smooth reciprocal-space sum
(``exp(-k^2/4 eta^2)``), a per-atom self-energy correction, and a uniform
neutralising-background term for a charged cell (standard Ewald; e.g.
Allen & Tildesley, "Computer Simulation of Liquids"; Frenkel & Smit,
"Understanding Molecular Simulation"):

.. math::

   E = \underbrace{\tfrac12 \sum_{i\neq j}\sum_{\mathbf n}
       \frac{q_i q_j\,\operatorname{erfc}(\eta r_{ij,\mathbf n})}{r_{ij,\mathbf n}}}_{\text{real}}
     + \underbrace{\frac{2\pi}{V}\sum_{\mathbf k\neq 0}
       \frac{e^{-k^2/4\eta^2}}{k^2}\,\lvert S(\mathbf k)\rvert^2}_{\text{reciprocal}}
     - \underbrace{\frac{\eta}{\sqrt\pi}\sum_i q_i^2}_{\text{self}}
     - \underbrace{\frac{\pi}{2 V \eta^2}\Big(\sum_i q_i\Big)^2}_{\text{background}} ,

with structure factor :math:`S(\mathbf k)=\sum_i q_i e^{i\mathbf k\cdot\mathbf r_i}`.
The total energy is **independent of the splitting parameter** :math:`\eta`; this
invariance is the standard Ewald correctness check exercised in the tests.

**Free (non-periodic) systems** -- the long-range energy reduces to the bare
pairwise Coulomb sum :math:`\tfrac12\sum_{i\neq j} q_i q_j / r_{ij}`.

Reference implementation
------------------------
The reciprocal-space structure-factor sum follows ``../jax-md``
(``jax_md/_energy/electrostatics.py``: ``coulomb_recip_ewald``,
``structure_factor``, ``coulomb_direct``) -- Schoenholz & Cubuk 2020, "JAX-MD".
The standalone form here additionally carries the self-energy and net-charge
background terms (absent from jax_md's neighbour-list path), and enumerates a
fixed shell of real-space periodic images so the real-space sum converges for
small cells, keeping the whole computation ``jit`` / ``grad`` / ``vmap`` clean.

Units are Gaussian / atomic (:math:`1/4\pi\varepsilon_0 = 1`); a downstream
caller multiplies by the appropriate Coulomb constant for its unit system.
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.scipy.special import erfc
from jaxtyping import Array, Complex, Float  # noqa: TC002

from opifex.core.quantum.registry import register_property_head
from opifex.neural.atomistic.heads.charge import conserve_total_charge


if TYPE_CHECKING:
    from opifex.core.quantum.molecular_system import MolecularSystem


logger = logging.getLogger(__name__)

# Small distance floor so self/coincident pairs never divide by zero under grad.
_DISTANCE_FLOOR: float = 1e-12
# Default Ewald splitting parameter (1/length); the energy is independent of it.
_DEFAULT_ETA: float = 0.4
# Default number of reciprocal lattice points searched per axis (half-range).
_DEFAULT_RECIPROCAL_CUTOFF: int = 10
# Default real-space periodic-image shell (per axis) for the erfc sum.
_DEFAULT_REAL_IMAGE_SHELL: int = 2


def _free_coulomb_energy(
    charges: Float[Array, " n_atoms"],
    positions: Float[Array, "n_atoms 3"],
) -> Float[Array, ""]:
    r"""Return the bare pairwise Coulomb energy of a free (open) system.

    Computes :math:`\tfrac12 \sum_{i\neq j} q_i q_j / r_{ij}` from all-pairs
    distances, masking the self term. Differentiable and ``vmap`` clean.

    Args:
        charges: Per-atom charges of shape ``(n_atoms,)``.
        positions: Cartesian positions of shape ``(n_atoms, 3)``.

    Returns:
        The scalar long-range Coulomb energy.
    """
    separations = positions[:, None, :] - positions[None, :, :]
    distances = jnp.sqrt(jnp.sum(separations**2, axis=-1) + _DISTANCE_FLOOR)
    charge_products = charges[:, None] * charges[None, :]
    off_diagonal = 1.0 - jnp.eye(charges.shape[0], dtype=positions.dtype)
    inverse_distance = jnp.where(off_diagonal > 0, 1.0 / distances, 0.0)
    return 0.5 * jnp.sum(charge_products * inverse_distance * off_diagonal)


def _real_image_offsets(shell: int) -> Float[Array, "n_images 3"]:
    """Return the integer real-space image offsets within ``shell`` per axis."""
    grid = range(-shell, shell + 1)
    return jnp.asarray(list(itertools.product(grid, grid, grid)), dtype=float)


def _reciprocal_vectors(cell: Float[Array, "3 3"], cutoff: int) -> Float[Array, "n_k 3"]:
    r"""Return the non-zero reciprocal lattice vectors within ``cutoff`` per axis.

    The reciprocal lattice is :math:`B = 2\\pi (A^{-1})^T` for row-vector cell
    ``A`` (``MolecularSystem`` / ASE convention), so :math:`\\mathbf k =
    \\mathbf m B` for integer triplets :math:`\\mathbf m`. The zero vector is
    dropped (handled by the self/background terms). The integer-triplet grid is a
    static NumPy enumeration so the reciprocal sum stays ``jit``-clean.

    Args:
        cell: ``(3, 3)`` lattice matrix whose rows are the cell vectors.
        cutoff: Half-range of integer reciprocal indices searched per axis.

    Returns:
        Reciprocal lattice vectors of shape ``(n_k, 3)`` (zero vector removed).
    """
    indices = np.asarray(list(itertools.product(range(-cutoff, cutoff + 1), repeat=3)), dtype=float)
    non_zero = np.any(indices != 0.0, axis=-1)
    integer_triplets = jnp.asarray(indices[non_zero])
    reciprocal_basis = 2.0 * jnp.pi * jnp.linalg.inv(cell).T
    return integer_triplets @ reciprocal_basis


def _ewald_real_space(
    charges: Float[Array, " n_atoms"],
    positions: Float[Array, "n_atoms 3"],
    cell: Float[Array, "3 3"],
    eta: float,
    image_shell: int,
) -> Float[Array, ""]:
    r"""Real-space ``erfc`` part of the Ewald sum over a shell of cell images.

    Sums :math:`\tfrac12 \sum_{i,j,\mathbf n}
    q_i q_j \operatorname{erfc}(\eta r)/r` over the integer cell images in the
    shell, excluding the :math:`i=j` self term in the home cell. Following
    ``../jax-md`` ``coulomb_direct``.
    """
    offsets = _real_image_offsets(image_shell) @ cell  # (n_images, 3)
    # separations[i, j, n] = r_i - r_j + image_n
    separations = (
        positions[:, None, None, :] - positions[None, :, None, :] + offsets[None, None, :, :]
    )
    distances = jnp.sqrt(jnp.sum(separations**2, axis=-1) + _DISTANCE_FLOOR)
    charge_products = charges[:, None, None] * charges[None, :, None]
    # Mask the i == j self term in the home cell (zero offset) only.
    home = jnp.all(jnp.abs(offsets) < 1e-9, axis=-1)  # (n_images,)
    self_mask = jnp.eye(charges.shape[0], dtype=positions.dtype)[:, :, None] * home
    keep = 1.0 - self_mask
    contribution = charge_products * erfc(eta * distances) / distances * keep
    return 0.5 * jnp.sum(contribution)


def _structure_factor(
    reciprocal_vectors: Float[Array, "n_k 3"],
    charges: Float[Array, " n_atoms"],
    positions: Float[Array, "n_atoms 3"],
) -> Complex[Array, " n_k"]:
    r"""Charge structure factor :math:`S(\mathbf k)=\sum_i q_i e^{i\mathbf k\cdot\mathbf r_i}`.

    Following ``../jax-md`` ``structure_factor``.
    """
    phase = jnp.einsum("kd,id->ki", reciprocal_vectors, positions)
    return jnp.sum(charges[None, :] * jnp.exp(1j * phase), axis=-1)


def _ewald_reciprocal_space(
    charges: Float[Array, " n_atoms"],
    positions: Float[Array, "n_atoms 3"],
    cell: Float[Array, "3 3"],
    eta: float,
    reciprocal_cutoff: int,
) -> Float[Array, ""]:
    r"""Reciprocal-space part of the Ewald sum.

    Evaluates :math:`\frac{2\pi}{V}\sum_{\mathbf k\neq 0}
    \frac{e^{-k^2/4\eta^2}}{k^2}\lvert S(\mathbf k)\rvert^2` following
    ``../jax-md`` ``coulomb_recip_ewald``.
    """
    reciprocal_vectors = _reciprocal_vectors(cell, reciprocal_cutoff)
    k_squared = jnp.sum(reciprocal_vectors**2, axis=-1)
    structure = _structure_factor(reciprocal_vectors, charges, positions)
    volume = jnp.abs(jnp.linalg.det(cell))
    weights = jnp.exp(-k_squared / (4.0 * eta**2)) / k_squared
    summed = jnp.sum(weights * jnp.abs(structure) ** 2)
    return (2.0 * jnp.pi / volume) * summed


def _ewald_self_and_background(
    charges: Float[Array, " n_atoms"],
    cell: Float[Array, "3 3"],
    eta: float,
) -> Float[Array, ""]:
    r"""Self-energy and net-charge neutralising-background corrections.

    Returns :math:`-\frac{\eta}{\sqrt\pi}\sum_i q_i^2
    - \frac{\pi}{2 V \eta^2}\big(\sum_i q_i\big)^2` (standard Ewald). The second
    term vanishes for a charge-neutral cell.
    """
    volume = jnp.abs(jnp.linalg.det(cell))
    self_energy = (eta / jnp.sqrt(jnp.pi)) * jnp.sum(charges**2)
    net_charge = jnp.sum(charges)
    background = (jnp.pi / (2.0 * volume * eta**2)) * net_charge**2
    return -self_energy - background


def latent_ewald_energy(
    charges: Float[Array, " n_atoms"],
    positions: Float[Array, "n_atoms 3"],
    cell: Float[Array, "3 3"] | None = None,
    *,
    eta: float = _DEFAULT_ETA,
    reciprocal_cutoff: int = _DEFAULT_RECIPROCAL_CUTOFF,
    real_image_shell: int = _DEFAULT_REAL_IMAGE_SHELL,
) -> Float[Array, ""]:
    r"""Long-range Coulomb energy of latent charges (Latent Ewald Summation).

    Dispatches on ``cell``: a free (open) system uses the bare pairwise sum
    :math:`\tfrac12\sum_{i\neq j} q_i q_j/r_{ij}`; a periodic system uses the full
    Ewald sum (real + reciprocal + self + net-charge background). The periodic
    energy is independent of the splitting parameter ``eta`` once
    ``reciprocal_cutoff`` and ``real_image_shell`` are large enough. Units are
    Gaussian / atomic (:math:`1/4\pi\varepsilon_0 = 1`).

    Method: Latent Ewald Summation (Cheng 2025, arXiv:2408.15165); Ewald
    summation (Allen & Tildesley); reciprocal sum after ``../jax-md``
    ``coulomb_recip_ewald`` / ``structure_factor``.

    Args:
        charges: Per-atom (latent) charges of shape ``(n_atoms,)``.
        positions: Cartesian positions of shape ``(n_atoms, 3)``.
        cell: ``(3, 3)`` lattice matrix (rows are cell vectors) for a periodic
            system, or ``None`` for a free system.
        eta: Ewald splitting parameter (1/length); periodic only. The total
            energy is invariant to it within the truncation tolerance.
        reciprocal_cutoff: Half-range of integer reciprocal indices per axis.
        real_image_shell: Real-space periodic-image shell (per axis) for the
            ``erfc`` sum.

    Returns:
        The scalar long-range Coulomb energy.
    """
    if cell is None:
        return _free_coulomb_energy(charges, positions)
    real = _ewald_real_space(charges, positions, cell, eta, real_image_shell)
    reciprocal = _ewald_reciprocal_space(charges, positions, cell, eta, reciprocal_cutoff)
    correction = _ewald_self_and_background(charges, cell, eta)
    return real + reciprocal + correction


@register_property_head("long_range_energy")
class LatentEwaldHead(nnx.Module):
    r"""Latent-Ewald long-range energy head (pluggable electrostatics add-on).

    Predicts total-charge-conserving per-atom *latent* charges from the
    backbone's invariant ``"node_features"`` (reusing
    :func:`~opifex.neural.atomistic.heads.charge.conserve_total_charge`) and
    returns the long-range Coulomb energy of those charges via
    :func:`latent_ewald_energy` (free: direct ``1/r``; periodic: full Ewald).

    Method: Latent Ewald Summation (Cheng 2025, arXiv:2408.15165). The energy is
    an additive correction to a short-range MLIP energy (§1.4 long-range seam).

    Args:
        feature_dim: Width of the backbone's ``"node_features"`` embedding.
        hidden_dim: Hidden width of the per-atom latent-charge MLP. Defaults to
            ``feature_dim``.
        eta: Ewald splitting parameter for periodic systems.
        reciprocal_cutoff: Half-range of integer reciprocal indices per axis.
        real_image_shell: Real-space periodic-image shell (per axis).
        rngs: Random number generators (keyword-only) seeding the MLP weights.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int | None = None,
        eta: float = _DEFAULT_ETA,
        reciprocal_cutoff: int = _DEFAULT_RECIPROCAL_CUTOFF,
        real_image_shell: int = _DEFAULT_REAL_IMAGE_SHELL,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the per-atom latent-charge MLP and store the Ewald settings."""
        super().__init__()
        width = hidden_dim if hidden_dim is not None else feature_dim
        self.hidden = nnx.Linear(feature_dim, width, rngs=rngs)
        self.readout = nnx.Linear(width, 1, rngs=rngs)
        self.eta = eta
        self.reciprocal_cutoff = reciprocal_cutoff
        self.real_image_shell = real_image_shell

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """This head emits the scalar ``"long_range_energy"``."""
        return ("long_range_energy",)

    def latent_charges(
        self,
        system: MolecularSystem,
        embeddings: dict[str, Array],
    ) -> Float[Array, " n_atoms"]:
        """Predict total-charge-conserving per-atom latent charges.

        Args:
            system: The molecular system; its ``charge`` is the conserved total.
            embeddings: Must contain ``"node_features"`` of shape
                ``(n_atoms, feature_dim)``.

        Returns:
            Per-atom latent charges of shape ``(n_atoms,)`` summing to
            ``system.charge``.
        """
        node_features = embeddings["node_features"]
        raw_charges = self.readout(nnx.silu(self.hidden(node_features)))[:, 0]
        return conserve_total_charge(raw_charges, system.charge)

    def __call__(
        self,
        system: MolecularSystem,
        graph: tuple[Array, Array],
        embeddings: dict[str, Array],
    ) -> dict[str, Array]:
        r"""Return the long-range Coulomb energy of the latent charges.

        Args:
            system: The molecular system providing positions, ``cell`` (``None``
                for free systems), and the conserved total ``charge``.
            graph: The ``(senders, receivers)`` edge index (unused by this head).
            embeddings: Must contain ``"node_features"`` of shape
                ``(n_atoms, feature_dim)``.

        Returns:
            ``{"long_range_energy": Array}`` -- the scalar long-range energy.
        """
        del graph
        charges = self.latent_charges(system, embeddings)
        energy = latent_ewald_energy(
            charges,
            system.positions,
            cell=system.cell,
            eta=self.eta,
            reciprocal_cutoff=self.reciprocal_cutoff,
            real_image_shell=self.real_image_shell,
        )
        return {"long_range_energy": energy}


__all__ = ["LatentEwaldHead", "latent_ewald_energy"]

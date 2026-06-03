r"""Becke-partitioned molecular quadrature grid for real-space XC integration.

The LDA exchange-correlation energy and potential require integrating functions
of the electron density over all space. This module builds a real molecular
quadrature grid by the standard Becke scheme:

* one atom-centred spherical grid per nucleus -- a radial quadrature times an
  angular (product Gauss-Legendre in :math:`\cos\theta` and uniform :math:`\phi`)
  grid;
* Becke fuzzy-cell weights that partition unity among the atomic cells so that
  overlapping atomic grids combine into a single molecular quadrature.

The grid is deliberately simple (a coarse product angular grid rather than a
Lebedev grid) but is a genuine molecular quadrature, sufficient to reproduce the
PySCF total energy to better than ``1e-4`` Ha on small molecules.

References
----------
* A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988) -- the fuzzy-cell weight
  scheme (eq. 13-22) and the recommended three iterations of the smoothing
  polynomial.
* C. W. Murray, N. C. Handy, G. J. Laming, *Mol. Phys.* **78**, 997 (1993) --
  the Euler-Maclaurin / log radial transform family (here a Becke-style
  :math:`r = R_m (1+x)/(1-x)` map with Gauss-Chebyshev radial nodes).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001


# Bragg-Slater atomic radii (Angstrom) used in Becke's size-adjusted partition,
# converted to Bohr. Values for H-Ne (Slater, J. Chem. Phys. 41, 3199 (1964);
# H adjusted to 0.35 as Becke recommends).
_BRAGG_RADII_ANGSTROM: dict[int, float] = {
    1: 0.35,
    2: 0.28,
    3: 1.45,
    4: 1.05,
    5: 0.85,
    6: 0.70,
    7: 0.65,
    8: 0.60,
    9: 0.50,
    10: 0.38,
}
_ANGSTROM_TO_BOHR = 1.0 / 0.52917721067


@dataclass(frozen=True, slots=True, kw_only=True)
class MolecularGrid:
    """A molecular quadrature grid.

    Attributes:
        points: Cartesian quadrature points in Bohr [Shape: (n_points, 3)].
        weights: Quadrature weights [Shape: (n_points,)].
    """

    points: Array
    weights: Array

    @property
    def n_points(self) -> int:
        """Number of quadrature points."""
        return int(self.points.shape[0])


def _atomic_radius_bohr(atomic_number: int) -> float:
    """Return the Bragg-Slater radius in Bohr (default to carbon's if unknown)."""
    angstrom = _BRAGG_RADII_ANGSTROM.get(atomic_number, 0.70)
    return angstrom * _ANGSTROM_TO_BOHR


def _radial_quadrature(n_radial: int, radius_scale: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Becke radial nodes/weights via :math:`r = R_m (1+x)/(1-x)`.

    Uses Gauss-Chebyshev (second-kind) nodes :math:`x_i` on ``(-1, 1)`` mapped to
    ``r in (0, inf)``; the Jacobian ``dr/dx`` and the Chebyshev weight factor are
    folded into the returned radial weights together with the ``r^2`` volume
    element.
    """
    indices = np.arange(1, n_radial + 1)
    angle = indices * np.pi / (n_radial + 1)
    x = np.cos(angle)
    # Gauss-Chebyshev (2nd kind) weights for weight sqrt(1-x^2).
    cheb_weights = np.pi / (n_radial + 1) * np.sin(angle) ** 2
    # Remove the sqrt(1-x^2) factor to recover plain integration weights.
    plain_weights = cheb_weights / np.sqrt(1.0 - x**2)

    radius = radius_scale * (1.0 + x) / (1.0 - x)
    jacobian = radius_scale * 2.0 / (1.0 - x) ** 2
    radial_weights = plain_weights * jacobian * radius**2
    return radius, radial_weights


def _angular_quadrature(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """Product angular grid: Gauss-Legendre in cos(theta), uniform in phi."""
    cos_theta, theta_weights = np.polynomial.legendre.leggauss(n_theta)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    phi_weight = 2.0 * np.pi / n_phi

    directions = []
    weights = []
    for i in range(n_theta):
        for j in range(n_phi):
            directions.append(
                (
                    sin_theta[i] * np.cos(phi[j]),
                    sin_theta[i] * np.sin(phi[j]),
                    cos_theta[i],
                )
            )
            weights.append(theta_weights[i] * phi_weight)
    return np.array(directions), np.array(weights)


def _becke_partition_weights(
    points: np.ndarray, centers: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Becke fuzzy-cell weights for every grid point.

    Returns:
        A pair ``(cell_weights, normaliser)`` where ``cell_weights`` has shape
        ``(n_points, n_atoms)`` holding the unnormalised fuzzy-cell weight of each
        point for each atom, and ``normaliser`` has shape ``(n_points,)`` holding
        their per-point sum (the partition-of-unity denominator).
    """
    n_points = points.shape[0]
    n_atoms = centers.shape[0]

    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
    cell_weights = np.ones((n_points, n_atoms))

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            r_ij = np.linalg.norm(centers[i] - centers[j])
            mu = (distances[:, i] - distances[:, j]) / r_ij
            # Becke size adjustment (eq. A2-A3) using atomic radii.
            chi = radii[i] / radii[j]
            u = (chi - 1.0) / (chi + 1.0)
            a = u / (u * u - 1.0)
            a = np.clip(a, -0.5, 0.5)
            nu = mu + a * (1.0 - mu * mu)
            # Three iterations of the smoothing polynomial p(t) = 1.5 t - 0.5 t^3.
            p = nu
            for _ in range(3):
                p = 1.5 * p - 0.5 * p**3
            s = 0.5 * (1.0 - p)
            cell_weights[:, i] *= s

    normaliser = np.sum(cell_weights, axis=1)
    return cell_weights, normaliser


def build_molecular_grid(
    system: MolecularSystem,
    *,
    n_radial: int = 40,
    n_theta: int = 14,
    n_phi: int = 28,
) -> MolecularGrid:
    """Build a Becke-partitioned molecular quadrature grid.

    Args:
        system: The molecular system (atomic centres and elements).
        n_radial: Number of radial quadrature shells per atom.
        n_theta: Number of polar (Gauss-Legendre) angular nodes per shell.
        n_phi: Number of azimuthal (uniform) angular nodes per shell.

    Returns:
        The assembled molecular grid with points (Bohr) and weights.
    """
    centers = np.asarray(system.positions, dtype=np.float64)
    atomic_numbers = [int(z) for z in system.atomic_numbers]
    radii = np.array([_atomic_radius_bohr(z) for z in atomic_numbers])

    directions, angular_weights = _angular_quadrature(n_theta, n_phi)

    all_points: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    owner_index: list[int] = []

    for atom_index, center in enumerate(centers):
        radius, radial_weights = _radial_quadrature(n_radial, radii[atom_index])
        # Outer product of radial shells and angular directions.
        shell_points = center[None, None, :] + radius[:, None, None] * directions[None, :, :]
        shell_points = shell_points.reshape(-1, 3)
        shell_weights = (radial_weights[:, None] * angular_weights[None, :]).reshape(-1)
        all_points.append(shell_points)
        all_weights.append(shell_weights)
        owner_index.extend([atom_index] * shell_points.shape[0])

    points = np.concatenate(all_points, axis=0)
    base_weights = np.concatenate(all_weights, axis=0)
    owners = np.array(owner_index)

    cell_weights, normaliser = _becke_partition_weights(points, centers, radii)
    partition = cell_weights[np.arange(points.shape[0]), owners] / normaliser
    weights = base_weights * partition

    return MolecularGrid(points=jnp.asarray(points), weights=jnp.asarray(weights))


def _becke_smoothing(nu: Array) -> Array:
    """Three iterations of Becke's smoothing polynomial ``p = 1.5 t - 0.5 t^3``."""
    p = nu
    for _ in range(3):
        p = 1.5 * p - 0.5 * p**3
    return p


# Floor under the squared distance so ``sqrt`` and its gradient stay finite when
# a grid point coincides (numerically) with a nucleus.
_DISTANCE_FLOOR_SQUARED = 1.0e-24


def _safe_norm(displacement: Array) -> Array:
    """Euclidean norm along the last axis with an AD-safe floor at the origin."""
    return jnp.sqrt(jnp.sum(displacement**2, axis=-1) + _DISTANCE_FLOOR_SQUARED)


@dataclass(frozen=True, slots=True, kw_only=True)
class MolecularGridTemplate:
    """Static grid structure with a position-traceable assembly.

    Holds everything about the quadrature that does not depend on the nuclear
    coordinates -- the per-atom radial node displacements (the shell radii times
    the angular directions), the base quadrature weights, the per-point owning
    atom and the size-adjustment parameters of the Becke partition. The
    :meth:`build` method then places the grid for a (possibly traced) set of
    nuclear positions, recomputing only the centre-dependent Becke weights in
    JAX so that the resulting grid is differentiable with respect to ``R``.

    Attributes:
        node_displacements: Per-point displacement from its owning atom centre,
            ``r_i = R_owner + displacement_i`` [Shape: (n_points, 3)].
        base_weights: Radial-times-angular quadrature weights [Shape: (n_points,)].
        owners: Owning-atom index of each point [Shape: (n_points,)].
        radii: Bragg-Slater atomic radii in Bohr [Shape: (n_atoms,)].
        size_adjustment: Becke ``a_{ij}`` size-adjustment matrix
            [Shape: (n_atoms, n_atoms)].
    """

    node_displacements: Array
    base_weights: Array
    owners: Array
    radii: Array
    size_adjustment: Array

    @property
    def n_points(self) -> int:
        """Number of quadrature points."""
        return int(self.node_displacements.shape[0])

    def build(self, positions: Array) -> MolecularGrid:
        """Assemble the molecular grid for (possibly traced) nuclear positions.

        Args:
            positions: Nuclear positions in Bohr [Shape: (n_atoms, 3)].

        Returns:
            The placed :class:`MolecularGrid` with weights carrying gradients
            with respect to ``positions``.
        """
        centers = positions[self.owners]  # (n_points, 3)
        points = centers + self.node_displacements

        # Distances from each point to every nucleus: (n_points, n_atoms). A
        # grid point can sit (numerically) on its owning nucleus, where the
        # Euclidean-norm gradient ``x/||x||`` is singular; ``_safe_norm`` adds a
        # tiny floor under the square root so ``jax.grad`` stays finite there.
        distances = _safe_norm(points[:, None, :] - positions[None, :, :])
        pair_distance = _safe_norm(positions[:, None, :] - positions[None, :, :])
        n_atoms = positions.shape[0]
        identity = jnp.eye(n_atoms, dtype=bool)
        safe_pair = jnp.where(identity, 1.0, pair_distance)

        # mu_ij(point) = (d_i - d_j)/R_ij; nu = mu + a_ij(1-mu^2).
        mu = (distances[:, :, None] - distances[:, None, :]) / safe_pair[None, :, :]
        nu = mu + self.size_adjustment[None, :, :] * (1.0 - mu * mu)
        smoothed = _becke_smoothing(nu)
        cell_factor = 0.5 * (1.0 - smoothed)  # s(nu_ij), shape (n_points, i, j)
        # Diagonal (i == j) must not contribute to the product over j.
        cell_factor = jnp.where(identity[None, :, :], 1.0, cell_factor)
        cell_weights = jnp.prod(cell_factor, axis=2)  # (n_points, n_atoms)

        normaliser = jnp.sum(cell_weights, axis=1)
        owned = jnp.take_along_axis(cell_weights, self.owners[:, None], axis=1)[:, 0]
        partition = owned / normaliser
        weights = self.base_weights * partition
        return MolecularGrid(points=points, weights=weights)


def build_molecular_grid_traceable(
    system: MolecularSystem,
    *,
    n_radial: int = 40,
    n_theta: int = 14,
    n_phi: int = 28,
) -> MolecularGridTemplate:
    """Build the static template for a position-traceable molecular grid.

    The radial/angular nodes and the Becke size-adjustment parameters depend only
    on the (static) elements, so they are precomputed here; the
    coordinate-dependent placement is deferred to
    :meth:`MolecularGridTemplate.build`. This separation lets ``jax.grad`` flow
    through the grid for analytic nuclear forces with the GGA functional.

    Args:
        system: The molecular system (atomic centres and elements).
        n_radial: Number of radial quadrature shells per atom.
        n_theta: Number of polar (Gauss-Legendre) angular nodes per shell.
        n_phi: Number of azimuthal (uniform) angular nodes per shell.

    Returns:
        The :class:`MolecularGridTemplate` for the system.
    """
    atomic_numbers = [int(z) for z in system.atomic_numbers]
    radii = np.array([_atomic_radius_bohr(z) for z in atomic_numbers])
    directions, angular_weights = _angular_quadrature(n_theta, n_phi)

    displacement_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    owner_index: list[int] = []
    for atom_index in range(len(atomic_numbers)):
        radius, radial_weights = _radial_quadrature(n_radial, radii[atom_index])
        shell_displacements = (radius[:, None, None] * directions[None, :, :]).reshape(-1, 3)
        shell_weights = (radial_weights[:, None] * angular_weights[None, :]).reshape(-1)
        displacement_blocks.append(shell_displacements)
        weight_blocks.append(shell_weights)
        owner_index.extend([atom_index] * shell_displacements.shape[0])

    node_displacements = np.concatenate(displacement_blocks, axis=0)
    base_weights = np.concatenate(weight_blocks, axis=0)
    owners = np.array(owner_index)

    # Becke size adjustment a_ij from the atomic-radius ratio (eq. A2-A6).
    n_atoms = len(atomic_numbers)
    size_adjustment = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            chi = radii[i] / radii[j]
            u = (chi - 1.0) / (chi + 1.0)
            a = u / (u * u - 1.0)
            size_adjustment[i, j] = np.clip(a, -0.5, 0.5)

    return MolecularGridTemplate(
        node_displacements=jnp.asarray(node_displacements),
        base_weights=jnp.asarray(base_weights),
        owners=jnp.asarray(owners),
        radii=jnp.asarray(radii),
        size_adjustment=jnp.asarray(size_adjustment),
    )


__all__ = [
    "MolecularGrid",
    "MolecularGridTemplate",
    "build_molecular_grid",
    "build_molecular_grid_traceable",
]

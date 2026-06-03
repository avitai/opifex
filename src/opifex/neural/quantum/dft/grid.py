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


__all__ = [
    "MolecularGrid",
    "build_molecular_grid",
]

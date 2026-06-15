"""3D molecular geometry, periodic cell, and domain-with-exclusion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.geometry.csg.operations import difference
from opifex.geometry.csg.primitives import Circle


if TYPE_CHECKING:
    from opifex.geometry.csg.types import Point3D, Points3D, Shape2D


# 3D Molecular Geometry Support (preserving exact API)
class MolecularGeometry:
    """3D molecular geometry with atomic coordinates."""

    def __init__(self, atomic_symbols: list[str], positions: jax.Array) -> None:
        """Initialize molecular geometry.

        Args:
            atomic_symbols: List of atomic symbols (e.g., ['H', 'H', 'O'])
            positions: Atomic positions in Bohr, shape (N, 3)

        Raises:
            ValueError: If number of symbols doesn't match number of positions
        """
        positions = jnp.asarray(positions)

        if len(atomic_symbols) != positions.shape[0]:
            raise ValueError("Number of atomic symbols must match number of positions")

        self.atomic_symbols = atomic_symbols
        self.positions = positions
        self.n_atoms = len(atomic_symbols)

    def compute_distances(self) -> jax.Array:
        """Compute all pairwise interatomic distances."""
        # Compute pairwise distance matrix
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        return jnp.linalg.norm(diff, axis=2)

    def project_to_2d(self, plane: str = "xy") -> jax.Array:
        """Project 3D coordinates to 2D plane."""
        if plane == "xy":
            return self.positions[:, :2]
        if plane == "xz":
            return self.positions[:, [0, 2]]
        if plane == "yz":
            return self.positions[:, [1, 2]]
        raise ValueError("Plane must be 'xy', 'xz', or 'yz'")

    @classmethod
    def from_molecular_system(cls, molecular_system) -> MolecularGeometry:
        """Create molecular geometry from MolecularSystem."""
        # Extract atomic symbols
        atomic_symbols = cls._extract_atomic_symbols(molecular_system)

        # Extract positions
        positions = cls._extract_positions(molecular_system)

        if atomic_symbols is None or positions is None:
            # Fallback: inspect the molecular system object for debugging
            available_attrs = [attr for attr in dir(molecular_system) if not attr.startswith("_")]
            raise ValueError(
                f"Molecular system must have atomic symbols and positions. "
                f"Available attributes: {available_attrs}. "
                f"Found atomic_symbols: {atomic_symbols is not None}, "
                f"Found positions: {positions is not None}"
            )

        return cls(atomic_symbols, positions)

    @classmethod
    def _extract_atomic_symbols(cls, molecular_system):
        """Extract atomic symbols from molecular system."""
        # Check for atomic_symbols attribute
        if hasattr(molecular_system, "atomic_symbols"):
            return molecular_system.atomic_symbols
        if hasattr(molecular_system, "symbols"):
            return molecular_system.symbols

        # Check atoms attribute - fix nested if statements
        if (
            hasattr(molecular_system, "atoms")
            and isinstance(molecular_system.atoms, list)
            and len(molecular_system.atoms) > 0
            and isinstance(molecular_system.atoms[0], tuple)
        ):
            return [atom[0] for atom in molecular_system.atoms]

        if hasattr(molecular_system, "species"):
            return molecular_system.species

        # Convert atomic numbers to symbols (fallback)
        if hasattr(molecular_system, "atomic_numbers"):
            atomic_number_to_symbol = {
                1: "H",
                2: "He",
                3: "Li",
                4: "Be",
                5: "B",
                6: "C",
                7: "N",
                8: "O",
                9: "F",
                10: "Ne",
                11: "Na",
                12: "Mg",
                13: "Al",
                14: "Si",
                15: "P",
                16: "S",
                17: "Cl",
                18: "Ar",
            }
            return [
                atomic_number_to_symbol.get(num, f"X{num}")
                for num in molecular_system.atomic_numbers
            ]
        return None

    @classmethod
    def _extract_positions(cls, molecular_system):
        """Extract positions from molecular system."""
        if hasattr(molecular_system, "positions"):
            return molecular_system.positions
        if hasattr(molecular_system, "coords"):
            return molecular_system.coords
        if hasattr(molecular_system, "geometry"):
            return molecular_system.geometry

        # Check atoms attribute - fix nested if statements
        if (
            hasattr(molecular_system, "atoms")
            and isinstance(molecular_system.atoms, list)
            and len(molecular_system.atoms) > 0
            and isinstance(molecular_system.atoms[0], tuple)
        ):
            return jnp.array([atom[1] for atom in molecular_system.atoms])
        return None


class PeriodicCell:
    """Periodic boundary conditions for materials systems."""

    def __init__(self, lattice_vectors: jax.Array) -> None:
        """Initialize periodic cell.

        Args:
            lattice_vectors: 3x3 array of lattice vectors in Bohr
        """
        self.lattice_vectors = jnp.asarray(lattice_vectors)
        if self.lattice_vectors.shape != (3, 3):
            raise ValueError("Lattice vectors must be 3x3 array")

        # Precompute reciprocal lattice vectors for efficiency
        self.reciprocal_vectors = jnp.linalg.inv(self.lattice_vectors).T

    @property
    def volume(self) -> float:
        """Compute volume of the unit cell."""
        return float(jnp.abs(jnp.linalg.det(self.lattice_vectors)))

    def wrap_coordinates(self, positions: jax.Array) -> jax.Array:
        """Wrap coordinates into unit cell."""
        # Convert to fractional coordinates
        fractional = jnp.linalg.solve(self.lattice_vectors.T, positions.T).T
        # Wrap to [0, 1)
        fractional_wrapped = fractional % 1.0
        # Convert back to Cartesian
        return fractional_wrapped @ self.lattice_vectors

    def wrap_to_unit_cell(self, point: Point3D) -> Point3D:
        """Wrap a single point to unit cell [0, 1)³."""
        # Convert to fractional coordinates
        fractional = jnp.dot(point, self.reciprocal_vectors)
        # Wrap to [0, 1)
        wrapped_fractional = fractional - jnp.floor(fractional)
        # Convert back to Cartesian coordinates
        return jnp.dot(wrapped_fractional, self.lattice_vectors)

    def periodic_distance(self, point1: Point3D, point2: Point3D) -> jax.Array:
        """Compute minimum distance between points considering periodicity."""
        # Convert to fractional coordinates
        frac1 = jnp.dot(point1, self.reciprocal_vectors)
        frac2 = jnp.dot(point2, self.reciprocal_vectors)

        # Compute minimum image difference
        diff_frac = frac2 - frac1
        diff_frac = diff_frac - jnp.round(diff_frac)  # Wrap to [-0.5, 0.5)

        # Convert back to Cartesian and compute distance
        diff_cart = jnp.dot(diff_frac, self.lattice_vectors)
        return jnp.linalg.norm(diff_cart)

    def minimum_image_distance(self, pos1: jax.Array, pos2: jax.Array) -> jax.Array:
        """Compute minimum image distance between two positions."""
        # Convert to fractional coordinates
        frac1 = jnp.linalg.solve(self.lattice_vectors.T, pos1.T).T
        frac2 = jnp.linalg.solve(self.lattice_vectors.T, pos2.T).T

        # Compute fractional displacement
        frac_disp = frac2 - frac1
        # Apply minimum image convention
        frac_disp = frac_disp - jnp.round(frac_disp)

        # Convert back to Cartesian
        cart_disp = frac_disp @ self.lattice_vectors
        return jnp.linalg.norm(cart_disp)

    def find_neighbors(
        self, positions: Points3D, cutoff_radius: float
    ) -> list[tuple[int, int, float]]:
        """Find neighboring atoms within cutoff radius considering periodicity.

        Args:
            positions: Atomic positions, shape (N, 3)
            cutoff_radius: Cutoff distance for neighbors

        Returns:
            List of (atom1_idx, atom2_idx, distance) tuples
        """
        n_atoms = positions.shape[0]

        # Create all pairwise combinations using JAX vectorized operations
        i_indices, j_indices = jnp.meshgrid(jnp.arange(n_atoms), jnp.arange(n_atoms), indexing="ij")

        # Only consider upper triangular pairs (i < j)
        upper_tri_mask = i_indices < j_indices

        # Get valid pairs
        valid_i = i_indices[upper_tri_mask]
        valid_j = j_indices[upper_tri_mask]

        # Compute distances for all valid pairs using vmap
        def compute_pair_distance(i, j):
            """Return the periodic distance between atoms ``i`` and ``j``."""
            return self.periodic_distance(positions[i], positions[j])

        distances = jax.vmap(compute_pair_distance)(valid_i, valid_j)

        # Filter by cutoff radius
        within_cutoff = distances <= cutoff_radius

        # Extract results
        neighbor_i = valid_i[within_cutoff]
        neighbor_j = valid_j[within_cutoff]
        neighbor_distances = distances[within_cutoff]

        # Convert to list of tuples for compatibility
        neighbors = []
        for idx in range(neighbor_i.shape[0]):
            neighbors.append(
                (
                    int(neighbor_i[idx]),
                    int(neighbor_j[idx]),
                    float(neighbor_distances[idx]),
                )
            )

        return neighbors


# Additional utility functions with optimal design patterns
def create_computational_domain_with_molecular_exclusion(
    domain_shape: Shape2D,
    molecular_geometry: MolecularGeometry,
    exclusion_radius: float = 1.0,
) -> Shape2D:
    """Create computational domain with molecular exclusion zones."""
    # Project molecular geometry to 2D
    projected_positions = molecular_geometry.project_to_2d()

    # Create exclusion zones around atoms using simple loop (more compatible)
    result_domain = domain_shape

    # Apply exclusions one by one
    for i in range(projected_positions.shape[0]):
        pos = projected_positions[i]
        exclusion_zone = Circle(center=pos, radius=exclusion_radius)
        result_domain = difference(result_domain, exclusion_zone)

    return result_domain


def create_molecular_geometry_from_dft_problem(dft_problem) -> MolecularGeometry:
    """Create molecular geometry from DFT problem specification."""
    # Extract molecular information from DFT problem
    if hasattr(dft_problem, "molecular_system"):
        mol_sys = dft_problem.molecular_system
        return MolecularGeometry.from_molecular_system(mol_sys)
    raise ValueError("DFT problem must have molecular_system attribute")

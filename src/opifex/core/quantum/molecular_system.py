"""
Molecular system representation for quantum mechanical calculations.

This module provides the core MolecularSystem dataclass and related utilities
for quantum mechanical calculations in the Opifex framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


# Atomic number to symbol mapping
ATOMIC_SYMBOLS = {
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
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
}

# Symbol to atomic number mapping
SYMBOL_TO_ATOMIC_NUMBER = {v: k for k, v in ATOMIC_SYMBOLS.items()}

# Atomic unit conversions
BOHR_TO_ANGSTROM = 0.52917721067
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
HARTREE_TO_EV = 27.21138602
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV


@dataclass
class MolecularSystem:
    """
    Standard molecular system representation for Neural DFT.

    This class provides a comprehensive representation of molecular systems
    suitable for quantum mechanical calculations, following atomic units
    convention and supporting both molecular and periodic systems.

    Attributes:
        atomic_numbers: Nuclear charges for each atom [Shape: (N_atoms,)]
        positions: Atomic positions in Bohr (atomic units) [Shape: (N_atoms, 3)]
        charge: Total molecular charge (default: 0)
        multiplicity: Spin multiplicity (2S + 1) (default: 1)
        cell: Periodic cell vectors in Bohr [Shape: (3, 3)] (default: None)
        basis_set: Basis set specification (default: "def2-tzvp")
    """

    atomic_numbers: Array  # Shape: (N_atoms,)
    positions: Array  # Shape: (N_atoms, 3) in Bohr
    charge: int = 0
    multiplicity: int = 1
    cell: Array | None = None  # Shape: (3, 3) for periodic systems
    basis_set: str = "def2-tzvp"

    def __post_init__(self):
        """Validate molecular system after initialization."""
        # Validate array shapes
        if self.positions.shape[0] != len(self.atomic_numbers):
            raise ValueError("Number of positions must match number of atoms")
        if self.positions.shape[1] != 3:
            raise ValueError("Positions must be 3D coordinates")

        # Validate physical constraints
        if self.multiplicity < 1:
            raise ValueError("Multiplicity must be positive")

        # Use JAX-compatible validation that works with JIT
        # Only perform validation if not in a JIT context
        try:
            # Convert to Python scalars for validation to avoid tracing issues
            atomic_numbers_py = [int(z) for z in self.atomic_numbers]
            if any(z <= 0 for z in atomic_numbers_py):
                raise ValueError("Atomic numbers must be positive")
            if any(z > 118 for z in atomic_numbers_py):
                raise ValueError("Atomic numbers must be <= 118")
        except (
            jax.errors.TracerIntegerConversionError,
            jax.errors.ConcretizationTypeError,
        ):
            # Skip validation when in JIT context - assume inputs are valid
            pass

        # Validate periodic cell if provided
        if self.cell is not None:
            if self.cell.shape != (3, 3):
                raise ValueError("Periodic cell must be 3x3 matrix")
            # Convert determinant to Python scalar for validation
            det_value = float(jnp.linalg.det(self.cell))
            if det_value <= 0:
                raise ValueError("Periodic cell must have positive determinant")

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the system."""
        return len(self.atomic_numbers)

    @property
    def n_electrons(self) -> int | jax.Array:
        """Total number of electrons."""
        # Use JAX-compatible computation that works with JIT
        try:
            # Try to convert to Python int if not in JIT context
            return int(jnp.sum(self.atomic_numbers) - self.charge)
        except (
            jax.errors.TracerIntegerConversionError,
            jax.errors.ConcretizationTypeError,
        ):
            # In JIT context, return the JAX array directly
            # The caller should handle this appropriately
            return jnp.sum(self.atomic_numbers) - self.charge

    @property
    def is_periodic(self) -> bool:
        """Whether the system has periodic boundary conditions."""
        return self.cell is not None

    @property
    def symbols(self) -> list[str]:
        """List of atomic symbols."""
        return [ATOMIC_SYMBOLS.get(int(z), f"X{int(z)}") for z in self.atomic_numbers]

    @property
    def molecular_formula(self) -> str:
        """Molecular formula string."""
        # Use JAX-compatible approach for counting symbols
        symbols = self.symbols
        unique_symbols = sorted(set(symbols))

        formula_parts = []
        for symbol in unique_symbols:
            # Count occurrences using JAX operations where possible
            count = sum(1 for s in symbols if s == symbol)
            if count == 1:
                formula_parts.append(symbol)
            else:
                formula_parts.append(f"{symbol}{count}")

        return "".join(formula_parts)

    @property
    def total_nuclear_charge(self) -> float:
        """Total nuclear charge."""
        return float(jnp.sum(self.atomic_numbers))

    @property
    def center_of_mass(self) -> Array:
        """Center of mass in Bohr."""
        # Use atomic masses (approximated as atomic numbers for simplicity)
        # JAX X64 mode handles precision automatically
        masses = jnp.array(self.atomic_numbers)
        total_mass = jnp.sum(masses)
        return jnp.sum(masses[:, None] * self.positions, axis=0) / total_mass

    @property
    def center_of_charge(self) -> Array:
        """Center of nuclear charge in Bohr."""
        total_charge = jnp.sum(self.atomic_numbers)
        return (
            jnp.sum(self.atomic_numbers[:, None] * self.positions, axis=0)
            / total_charge
        )

    def distance_matrix(self) -> Array:
        """
        Compute pairwise distance matrix between atoms.

        Returns:
            Distance matrix in Bohr [Shape: (N_atoms, N_atoms)]
        """
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        return jnp.sqrt(jnp.sum(diff**2, axis=-1))

    def get_atomic_masses(self) -> Array:
        """
        Get atomic masses for all atoms.

        Returns:
            Atomic masses in atomic mass units [Shape: (N_atoms,)]
        """
        # Simple approximation: atomic mass ≈ atomic number
        # For more accurate calculations, use a proper atomic mass table
        # JAX X64 mode handles precision automatically
        return jnp.array(self.atomic_numbers)

    def get_positions_angstrom(self) -> Array:
        """
        Get atomic positions in Angstrom.

        Returns:
            Positions in Angstrom [Shape: (N_atoms, 3)]
        """
        return self.positions * BOHR_TO_ANGSTROM

    def translate(self, translation: Array) -> MolecularSystem:
        """
        Translate the molecular system.

        Args:
            translation: Translation vector in Bohr [Shape: (3,)]

        Returns:
            New MolecularSystem with translated positions
        """
        new_positions = self.positions + translation
        return MolecularSystem(
            atomic_numbers=self.atomic_numbers,
            positions=new_positions,
            charge=self.charge,
            multiplicity=self.multiplicity,
            cell=self.cell,
            basis_set=self.basis_set,
        )

    def center_at_origin(self) -> MolecularSystem:
        """
        Center the molecular system at the origin (center of mass).

        Returns:
            New MolecularSystem centered at origin
        """
        return self.translate(-self.center_of_mass)

    def detect_symmetry(self) -> dict[str, Any]:
        """
        Detect molecular symmetry (basic implementation).

        Returns:
            Dictionary containing symmetry information
        """
        # Basic symmetry detection - can be enhanced with more sophisticated algorithms
        symmetry_info = {
            "point_group": "C1",  # Default to no symmetry
            "has_inversion": False,
            "has_reflection": False,
            "rotation_axes": [],
        }

        # Check for linear molecule
        if self.n_atoms == 2:
            symmetry_info["point_group"] = "C∞v"

        # For more complex symmetry detection, implement proper point group analysis
        return symmetry_info

    def validate_quantum_system(self) -> bool:
        """
        Validate quantum mechanical consistency.

        Returns:
            True if system is quantum mechanically valid
        """
        # Check electron count parity with multiplicity
        n_unpaired = self.multiplicity - 1
        if (self.n_electrons - n_unpaired) % 2 != 0:
            return False

        # Check physical constraints
        if self.n_electrons < 0:
            return False

        # Check minimum distance between atoms (prevent nuclear overlap)
        if self.n_atoms > 1:
            distances = self.distance_matrix()
            # Mask diagonal elements using JAX-compatible operations
            mask = jnp.eye(self.n_atoms, dtype=bool)
            distances_masked = jnp.where(mask, jnp.inf, distances)
            min_distance = float(jnp.min(distances_masked))
            if min_distance < 0.1:  # Minimum distance of 0.1 Bohr
                return False

        return True

    def get_system_info(self) -> dict[str, Any]:
        """
        Get comprehensive system information.

        Returns:
            Dictionary with system properties
        """
        return {
            "n_atoms": self.n_atoms,
            "n_electrons": self.n_electrons,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "molecular_formula": self.molecular_formula,
            "total_nuclear_charge": self.total_nuclear_charge,
            "is_periodic": self.is_periodic,
            "basis_set": self.basis_set,
            "center_of_mass": self.center_of_mass,
            "center_of_charge": self.center_of_charge,
            "quantum_valid": self.validate_quantum_system(),
            "symmetry": self.detect_symmetry(),
        }

    def __str__(self) -> str:
        """String representation of the molecular system."""
        lines = [
            f"MolecularSystem: {self.molecular_formula}",
            f"Atoms: {self.n_atoms}, Electrons: {self.n_electrons}",
            f"Charge: {self.charge}, Multiplicity: {self.multiplicity}",
            f"Basis set: {self.basis_set}",
        ]

        if self.is_periodic:
            lines.append("Periodic boundary conditions: Yes")

        return "\n".join(lines)


def create_molecular_system(
    atoms: list[tuple[str, tuple[float, float, float]]],
    charge: int = 0,
    multiplicity: int = 1,
    basis_set: str = "def2-tzvp",
    cell: Array | None = None,
) -> MolecularSystem:
    """
    Create a molecular system from atomic symbols and coordinates.

    Args:
        atoms: List of (symbol, (x, y, z)) tuples in Angstrom
        charge: Total molecular charge
        multiplicity: Spin multiplicity
        basis_set: Basis set specification
        cell: Periodic cell vectors in Angstrom [Shape: (3, 3)]

    Returns:
        MolecularSystem instance with positions in Bohr
    """
    atomic_numbers = []
    positions_angstrom = []

    for symbol, coords in atoms:
        if symbol not in SYMBOL_TO_ATOMIC_NUMBER:
            raise ValueError(f"Unknown atomic symbol: {symbol}")

        atomic_numbers.append(SYMBOL_TO_ATOMIC_NUMBER[symbol])
        positions_angstrom.append(coords)

    # Convert to JAX arrays
    atomic_numbers_array = jnp.array(atomic_numbers)
    positions_bohr = jnp.array(positions_angstrom) * ANGSTROM_TO_BOHR

    # Convert cell to Bohr if provided
    cell_bohr = None
    if cell is not None:
        cell_bohr = cell * ANGSTROM_TO_BOHR

    return MolecularSystem(
        atomic_numbers=atomic_numbers_array,
        positions=positions_bohr,
        charge=charge,
        multiplicity=multiplicity,
        cell=cell_bohr,
        basis_set=basis_set,
    )


def create_water_molecule(
    oh_distance: float = 0.96, hoh_angle: float = 104.5
) -> MolecularSystem:
    """
    Create a water molecule with specified geometry.

    Args:
        oh_distance: O-H bond length in Angstrom
        hoh_angle: H-O-H angle in degrees

    Returns:
        MolecularSystem for water molecule
    """
    import math

    # Place oxygen at origin
    o_pos = (0.0, 0.0, 0.0)

    # Calculate hydrogen positions
    half_angle = math.radians(hoh_angle / 2)
    h1_pos = (
        oh_distance * math.sin(half_angle),
        oh_distance * math.cos(half_angle),
        0.0,
    )
    h2_pos = (
        -oh_distance * math.sin(half_angle),
        oh_distance * math.cos(half_angle),
        0.0,
    )

    atoms = [
        ("O", o_pos),
        ("H", h1_pos),
        ("H", h2_pos),
    ]

    return create_molecular_system(atoms, charge=0, multiplicity=1)


def create_methane_molecule(ch_distance: float = 1.09) -> MolecularSystem:
    """
    Create a methane molecule with tetrahedral geometry.

    Args:
        ch_distance: C-H bond length in Angstrom

    Returns:
        MolecularSystem for methane molecule
    """
    import math

    # Tetrahedral geometry - hydrogen positions around carbon at origin
    tetrahedral_angle = math.acos(-1 / 3)  # ~109.47 degrees

    # Place carbon at origin
    c_pos = (0.0, 0.0, 0.0)

    # Calculate hydrogen positions in tetrahedral arrangement
    positions = [
        (0.0, 0.0, ch_distance),
        (
            ch_distance * math.sin(tetrahedral_angle),
            0.0,
            ch_distance * math.cos(tetrahedral_angle),
        ),
        (
            -ch_distance * math.sin(tetrahedral_angle) * math.cos(math.pi / 3),
            ch_distance * math.sin(tetrahedral_angle) * math.sin(math.pi / 3),
            ch_distance * math.cos(tetrahedral_angle),
        ),
        (
            -ch_distance * math.sin(tetrahedral_angle) * math.cos(math.pi / 3),
            -ch_distance * math.sin(tetrahedral_angle) * math.sin(math.pi / 3),
            ch_distance * math.cos(tetrahedral_angle),
        ),
    ]

    atoms = [("C", c_pos)] + [("H", pos) for pos in positions]

    return create_molecular_system(atoms, charge=0, multiplicity=1)

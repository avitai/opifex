r"""Contracted-Gaussian atomic-orbital basis sets for molecular DFT.

This module turns a :class:`~opifex.core.quantum.molecular_system.MolecularSystem`
plus a basis-set name into an :class:`AtomicOrbitalBasis`: the flat list of
contracted Cartesian-Gaussian shells (one per ``(atom, l)`` block) together with
the atomic-orbital (AO) offset table required by the McMurchie-Davidson integral
engine in :mod:`opifex.core.quantum.backend`.

Only the **STO-3G** minimal basis is provided for now, with hardcoded
contracted-GTO exponents and contraction coefficients for H, C, N and O. The
numbers are the standard EMSL Basis Set Exchange / PySCF STO-3G values (Hehre,
Stewart & Pople, *J. Chem. Phys.* **51**, 2657 (1969)); they are validated
indirectly by the integral tests against ``pyscf.gto.M(...).intor(...)``.

Primitive normalisation convention
----------------------------------
Each primitive Cartesian Gaussian is individually normalised, and the contracted
combination is then renormalised so that the contracted AO has unit self-overlap
(the PySCF / standard quantum-chemistry convention). The renormalisation is
folded into the stored contraction coefficients so the integral engine never has
to renormalise again.

References
----------
* T. Helgaker, P. Jorgensen, J. Olsen, *Molecular Electronic-Structure Theory*,
  Wiley (2000), Ch. 9 (Gaussian basis functions and their normalisation).
* W. J. Hehre, R. F. Stewart, J. A. Pople, *J. Chem. Phys.* **51**, 2657 (1969)
  (the STO-NG contraction coefficients).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from opifex.core.quantum.molecular_system import ATOMIC_SYMBOLS, MolecularSystem


# ---------------------------------------------------------------------------
# STO-3G primitive data (EMSL / PySCF values).
#
# Each entry is a list of shells; each shell is ``(l, [(exponent, coeff), ...])``
# where ``coeff`` is the *contraction* coefficient against the individually
# normalised primitive (the STO-NG convention -- the same set of coefficients is
# reused across elements per l-block).
# ---------------------------------------------------------------------------
_STO3G: dict[str, list[tuple[int, list[tuple[float, float]]]]] = {
    "H": [
        (
            0,
            [
                (3.42525091, 0.15432897),
                (0.62391373, 0.53532814),
                (0.16885540, 0.44463454),
            ],
        ),
    ],
    "C": [
        (
            0,
            [
                (71.6168370, 0.15432897),
                (13.0450960, 0.53532814),
                (3.5305122, 0.44463454),
            ],
        ),
        (
            0,
            [
                (2.9412494, -0.09996723),
                (0.6834831, 0.39951283),
                (0.2222899, 0.70011547),
            ],
        ),
        (
            1,
            [
                (2.9412494, 0.15591627),
                (0.6834831, 0.60768372),
                (0.2222899, 0.39195739),
            ],
        ),
    ],
    "N": [
        (
            0,
            [
                (99.1061690, 0.15432897),
                (18.0523120, 0.53532814),
                (4.8856602, 0.44463454),
            ],
        ),
        (
            0,
            [
                (3.7804559, -0.09996723),
                (0.8784966, 0.39951283),
                (0.2857144, 0.70011547),
            ],
        ),
        (
            1,
            [
                (3.7804559, 0.15591627),
                (0.8784966, 0.60768372),
                (0.2857144, 0.39195739),
            ],
        ),
    ],
    "O": [
        (
            0,
            [
                (130.7093200, 0.15432897),
                (23.8088610, 0.53532814),
                (6.4436083, 0.44463454),
            ],
        ),
        (
            0,
            [
                (5.0331513, -0.09996723),
                (1.1695961, 0.39951283),
                (0.3803890, 0.70011547),
            ],
        ),
        (
            1,
            [
                (5.0331513, 0.15591627),
                (1.1695961, 0.60768372),
                (0.3803890, 0.39195739),
            ],
        ),
    ],
}

# Cartesian angular-momentum component lists, ordered to match PySCF's Cartesian
# ordering for s and p shells: s -> (x=y=z=0); p -> (x), (y), (z).
_CART_COMPONENTS: dict[int, tuple[tuple[int, int, int], ...]] = {
    0: ((0, 0, 0),),
    1: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
}


def _double_factorial(value: int) -> int:
    """Return the double factorial ``value!!`` for ``value >= -1``."""
    result = 1
    while value > 1:
        result *= value
        value -= 2
    return result


def _primitive_norm(exponent: float, angular: tuple[int, int, int]) -> float:
    r"""Normalisation constant of a single Cartesian primitive Gaussian.

    For a primitive ``x^l y^m z^n exp(-a r^2)`` the normalisation is

    .. math::
        N = (2a/\pi)^{3/4}
            \frac{(4a)^{(l+m+n)/2}}
                 {\sqrt{(2l-1)!!\,(2m-1)!!\,(2n-1)!!}}

    (Helgaker, Jorgensen & Olsen, eq. 9.2.4).
    """
    l_x, l_y, l_z = angular
    total = l_x + l_y + l_z
    numerator = (2.0 * exponent / np.pi) ** 0.75 * (4.0 * exponent) ** (total / 2.0)
    denominator = np.sqrt(
        _double_factorial(2 * l_x - 1)
        * _double_factorial(2 * l_y - 1)
        * _double_factorial(2 * l_z - 1)
    )
    return float(numerator / denominator)


@dataclass(frozen=True, slots=True, kw_only=True)
class GaussianShell:
    """A single contracted Cartesian-Gaussian shell centred on one atom.

    Attributes:
        atom_index: Index of the atom this shell is centred on.
        angular_momentum: Total angular momentum ``l`` (0 = s, 1 = p, ...).
        center: Cartesian centre in Bohr [Shape: (3,)].
        exponents: Primitive Gaussian exponents [Shape: (n_prim,)].
        coefficients: Contraction coefficients for *l-axis-aligned* primitives,
            already including primitive normalisation and contracted
            renormalisation [Shape: (n_prim,)].
        ao_offset: Index of the first AO produced by this shell in the flat AO
            ordering.
    """

    atom_index: int
    angular_momentum: int
    center: Array
    exponents: Array
    coefficients: Array
    ao_offset: int

    @property
    def n_primitives(self) -> int:
        """Number of primitive Gaussians in the contraction."""
        return int(self.exponents.shape[0])

    @property
    def n_cartesian(self) -> int:
        """Number of Cartesian AO components in this shell (1 for s, 3 for p)."""
        return len(_CART_COMPONENTS[self.angular_momentum])

    @property
    def cartesian_components(self) -> tuple[tuple[int, int, int], ...]:
        """Cartesian ``(l_x, l_y, l_z)`` powers for each AO in this shell."""
        return _CART_COMPONENTS[self.angular_momentum]


def _contracted_self_overlap(
    exponents: np.ndarray, normalised_coeffs: np.ndarray, angular: tuple[int, int, int]
) -> float:
    r"""Self-overlap of a contraction of normalised primitives along one axis.

    Uses the closed-form same-centre Gaussian overlap

    .. math::
        \langle a | b \rangle =
            \left(\frac{\pi}{a+b}\right)^{3/2}
            \frac{(2l-1)!!\,(2m-1)!!\,(2n-1)!!}{(2(a+b))^{l+m+n}}

    summed over primitive pairs with their normalisation constants folded in.
    """
    l_x, l_y, l_z = angular
    total = l_x + l_y + l_z
    angular_factor = (
        _double_factorial(2 * l_x - 1)
        * _double_factorial(2 * l_y - 1)
        * _double_factorial(2 * l_z - 1)
    )
    overlap = 0.0
    for i, exp_i in enumerate(exponents):
        for j, exp_j in enumerate(exponents):
            combined = exp_i + exp_j
            pair = (np.pi / combined) ** 1.5 * angular_factor / (2.0 * combined) ** total
            overlap += normalised_coeffs[i] * normalised_coeffs[j] * pair
    return float(overlap)


def _build_shell_coefficients(
    exponents: np.ndarray, raw_coeffs: np.ndarray, angular_momentum: int
) -> np.ndarray:
    """Fold primitive normalisation and contracted renormalisation into coeffs.

    The l-axis-aligned Cartesian component ``(l, 0, 0)`` is used as the reference
    for the contracted renormalisation; the per-primitive normalisation it
    carries is shared by every Cartesian component of the shell because primitive
    normalisation depends only on the total angular momentum for axis-aligned
    powers.
    """
    reference_angular = (angular_momentum, 0, 0)
    norms = np.array([_primitive_norm(float(exp), reference_angular) for exp in exponents])
    normalised = raw_coeffs * norms
    self_overlap = _contracted_self_overlap(exponents, normalised, reference_angular)
    return normalised / np.sqrt(self_overlap)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlatPrimitives:
    r"""Flat array-of-primitives view of an AO basis (MESS batching layout).

    A *primitive* here is one Cartesian component of one primitive Gaussian: a
    ``p``-shell with three primitives expands to nine flat primitives (three
    Cartesian components :math:`\times` three radial primitives). Stacking every
    primitive into 1-D arrays lets the McMurchie-Davidson integral kernels be
    evaluated with a single :func:`jax.vmap` over primitive pairs/quartets and
    contracted to AOs with :func:`jax.ops.segment_sum` (the
    ``graphcore-research/mess`` pattern), replacing the eager shell loops.

    The split between *traced* and *static* fields is deliberate: the geometry
    and Gaussian parameters (``center``, ``alpha``, ``coeff``) are JAX arrays so
    the integrals stay differentiable w.r.t. nuclear positions and the build is
    ``jit``-traceable, while the angular-momentum powers (``lmn``), the
    primitive->AO map (``orbital_index``) and ``max_total_l`` are NumPy / Python
    so they parametrise the (unrolled, static) recurrences without becoming
    tracers.

    Attributes:
        center: Primitive centres in Bohr [Shape: (n_prim, 3)] (traced).
        alpha: Primitive exponents [Shape: (n_prim,)] (traced).
        coeff: Contraction coefficients with primitive + contracted
            normalisation folded in [Shape: (n_prim,)] (traced).
        lmn: Cartesian angular-momentum powers [Shape: (n_prim, 3)]
            (NumPy-static int).
        orbital_index: Contracted-AO index of each primitive [Shape: (n_prim,)]
            (NumPy-static int).
        num_orbitals: Number of contracted Cartesian AOs.
        max_total_l: Maximum total angular momentum of any primitive (static;
            sizes the McMurchie-Davidson Hermite tables).
    """

    center: Array
    alpha: Array
    coeff: Array
    lmn: np.ndarray
    orbital_index: np.ndarray
    num_orbitals: int
    max_total_l: int

    @property
    def num_primitives(self) -> int:
        """Number of flat primitives (Cartesian-component granularity)."""
        return int(self.alpha.shape[0])


@dataclass(frozen=True, slots=True, kw_only=True)
class AtomicOrbitalBasis:
    """A contracted-Gaussian AO basis built from a molecular system.

    Attributes:
        shells: Flat tuple of contracted shells, in atom-major order.
        n_atomic_orbitals: Total number of Cartesian AOs.
        basis_name: The basis-set name this object was built from.
    """

    shells: tuple[GaussianShell, ...]
    n_atomic_orbitals: int
    basis_name: str

    def flat_primitives(self) -> FlatPrimitives:
        """Return the flat array-of-primitives view of this basis.

        Expands every contracted shell into its Cartesian components and
        primitive Gaussians, stacking centres/exponents/coefficients into 1-D
        JAX arrays (traced) and the angular powers / primitive->AO map into
        NumPy arrays (static). See :class:`FlatPrimitives` for the layout.

        Returns:
            The :class:`FlatPrimitives` view.
        """
        centers: list[Array] = []
        alphas: list[Array] = []
        coeffs: list[Array] = []
        lmn_rows: list[tuple[int, int, int]] = []
        orbital_index: list[int] = []
        ao_index = 0
        max_total_l = 0
        for shell in self.shells:
            n_prim = shell.n_primitives
            for power in shell.cartesian_components:
                centers.append(jnp.broadcast_to(shell.center, (n_prim, 3)))
                alphas.append(shell.exponents)
                coeffs.append(shell.coefficients)
                lmn_rows.extend([power] * n_prim)
                orbital_index.extend([ao_index] * n_prim)
                max_total_l = max(max_total_l, sum(power))
                ao_index += 1
        return FlatPrimitives(
            center=jnp.concatenate(centers, axis=0),
            alpha=jnp.concatenate(alphas, axis=0),
            coeff=jnp.concatenate(coeffs, axis=0),
            lmn=np.asarray(lmn_rows, dtype=np.int32),
            orbital_index=np.asarray(orbital_index, dtype=np.int32),
            num_orbitals=self.n_atomic_orbitals,
            max_total_l=int(max_total_l),
        )

    @classmethod
    def from_molecular_system(
        cls, system: MolecularSystem, basis_name: str = "sto-3g"
    ) -> AtomicOrbitalBasis:
        """Build an :class:`AtomicOrbitalBasis` for ``system``.

        Args:
            system: The molecular system providing atom centres and elements.
            basis_name: The basis-set name (only ``"sto-3g"`` is supported).

        Returns:
            The assembled AO basis.

        Raises:
            ValueError: If ``basis_name`` is unsupported or an element has no
                tabulated basis data.
        """
        normalised_name = basis_name.lower()
        if normalised_name != "sto-3g":
            raise ValueError(f"Unsupported basis set {basis_name!r}; only 'sto-3g' is available")

        positions = np.asarray(system.positions, dtype=np.float64)
        atomic_numbers = [int(z) for z in system.atomic_numbers]

        shells: list[GaussianShell] = []
        ao_offset = 0
        for atom_index, atomic_number in enumerate(atomic_numbers):
            symbol = ATOMIC_SYMBOLS.get(atomic_number)
            element_shells = _STO3G.get(symbol) if symbol is not None else None
            if element_shells is None:
                raise ValueError(
                    f"No STO-3G data for element Z={atomic_number}"
                    f" (symbol={symbol}); tabulated: {sorted(_STO3G)}"
                )
            center = jnp.asarray(positions[atom_index])
            for angular_momentum, primitives in element_shells:
                exponents = np.array([exp for exp, _ in primitives])
                raw_coeffs = np.array([coeff for _, coeff in primitives])
                coefficients = _build_shell_coefficients(exponents, raw_coeffs, angular_momentum)
                shell = GaussianShell(
                    atom_index=atom_index,
                    angular_momentum=angular_momentum,
                    center=center,
                    exponents=jnp.asarray(exponents),
                    coefficients=jnp.asarray(coefficients),
                    ao_offset=ao_offset,
                )
                shells.append(shell)
                ao_offset += shell.n_cartesian

        return cls(
            shells=tuple(shells),
            n_atomic_orbitals=ao_offset,
            basis_name=normalised_name,
        )

    @property
    def n_shells(self) -> int:
        """Number of contracted shells."""
        return len(self.shells)

    def with_positions(self, positions: Array) -> AtomicOrbitalBasis:
        """Return a copy with shell centres taken from ``positions``.

        The exponents, contraction coefficients, angular momenta and AO offsets
        (the static basis structure) are preserved; only the per-shell centre is
        re-sourced from ``positions[atom_index]``. Because the (possibly traced)
        ``positions`` array flows straight into each shell centre, the integrals
        built from the returned basis are differentiable with respect to the
        nuclear coordinates -- this is the seam analytic forces rely on.

        Args:
            positions: Nuclear positions in Bohr [Shape: (n_atoms, 3)].

        Returns:
            A new :class:`AtomicOrbitalBasis` centred at ``positions``.
        """
        moved = tuple(
            GaussianShell(
                atom_index=shell.atom_index,
                angular_momentum=shell.angular_momentum,
                center=positions[shell.atom_index],
                exponents=shell.exponents,
                coefficients=shell.coefficients,
                ao_offset=shell.ao_offset,
            )
            for shell in self.shells
        )
        return AtomicOrbitalBasis(
            shells=moved,
            n_atomic_orbitals=self.n_atomic_orbitals,
            basis_name=self.basis_name,
        )

    def evaluate(self, points: Array) -> Array:
        r"""Evaluate every AO at a set of Cartesian points.

        Each contracted Cartesian AO is

        .. math::
            \phi(r) = (x-X)^{l_x}(y-Y)^{l_y}(z-Z)^{l_z}
                      \sum_p c_p \, e^{-a_p |r-R|^2}.

        Args:
            points: Cartesian points in Bohr [Shape: (n_points, 3)].

        Returns:
            AO values [Shape: (n_points, n_atomic_orbitals)] in the flat AO
            ordering.
        """
        columns: list[Array] = []
        for shell in self.shells:
            offset = shell.center
            displacement = points - offset[None, :]
            r2 = jnp.sum(displacement**2, axis=-1)
            radial = jnp.sum(
                shell.coefficients[None, :] * jnp.exp(-shell.exponents[None, :] * r2[:, None]),
                axis=-1,
            )
            for power in shell.cartesian_components:
                angular = (
                    displacement[:, 0] ** power[0]
                    * displacement[:, 1] ** power[1]
                    * displacement[:, 2] ** power[2]
                )
                columns.append(angular * radial)
        return jnp.stack(columns, axis=-1)

    def evaluate_with_gradients(self, points: Array) -> tuple[Array, Array]:
        r"""Evaluate every AO and its Cartesian gradient at a set of points.

        Differentiating :meth:`evaluate` analytically gives, for each contracted
        AO :math:`\phi = g(r)\,R(r)` with angular factor
        :math:`g = \prod_c (r_c-X_c)^{l_c}` and radial sum
        :math:`R = \sum_p c_p e^{-a_p|r-R|^2}`,

        .. math::
            \frac{\partial\phi}{\partial r_c} =
                \frac{\partial g}{\partial r_c} R
              + g\,\frac{\partial R}{\partial r_c},\quad
            \frac{\partial R}{\partial r_c} =
                \sum_p c_p (-2 a_p)(r_c-X_c) e^{-a_p|r-R|^2},

        and :math:`\partial g/\partial r_c = l_c (r_c-X_c)^{l_c-1}
        \prod_{d\neq c}(r_d-X_d)^{l_d}` (zero when :math:`l_c=0`). The gradient
        of the density on a grid is then assembled from these AO gradients.

        Args:
            points: Cartesian points in Bohr [Shape: (n_points, 3)].

        Returns:
            A pair ``(values, gradients)`` where ``values`` has shape
            ``(n_points, n_atomic_orbitals)`` and ``gradients`` has shape
            ``(n_points, n_atomic_orbitals, 3)`` (the last axis is ``d/dr_c``).
        """
        value_columns: list[Array] = []
        gradient_columns: list[Array] = []
        for shell in self.shells:
            offset = shell.center
            displacement = points - offset[None, :]
            r2 = jnp.sum(displacement**2, axis=-1)
            exponentials = jnp.exp(-shell.exponents[None, :] * r2[:, None])
            radial = jnp.sum(shell.coefficients[None, :] * exponentials, axis=-1)
            # d(radial)/dr_c = sum_p c_p (-2 a_p)(r_c - X_c) exp(...).
            radial_factor = jnp.sum(
                shell.coefficients[None, :] * (-2.0 * shell.exponents[None, :]) * exponentials,
                axis=-1,
            )
            radial_gradient = radial_factor[:, None] * displacement  # (n_points, 3)

            for power in shell.cartesian_components:
                axis_powers = [displacement[:, axis] ** power[axis] for axis in range(3)]
                angular = axis_powers[0] * axis_powers[1] * axis_powers[2]
                value_columns.append(angular * radial)

                angular_gradient_axes = []
                for axis in range(3):
                    if power[axis] == 0:
                        derivative = jnp.zeros_like(displacement[:, axis])
                    else:
                        derivative = power[axis] * displacement[:, axis] ** (power[axis] - 1)
                        for other in range(3):
                            if other != axis:
                                derivative = derivative * axis_powers[other]
                    angular_gradient_axes.append(derivative)
                angular_gradient = jnp.stack(angular_gradient_axes, axis=-1)

                gradient = angular_gradient * radial[:, None] + angular[:, None] * radial_gradient
                gradient_columns.append(gradient)

        values = jnp.stack(value_columns, axis=-1)
        gradients = jnp.stack(gradient_columns, axis=1)
        return values, gradients


__all__ = [
    "AtomicOrbitalBasis",
    "FlatPrimitives",
    "GaussianShell",
]

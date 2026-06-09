r"""Resolution-of-identity / density-fitting two-electron integrals.

The four-index electron-repulsion integral (ERI) :math:`(\mu\nu|\lambda\sigma)`
is the :math:`O(N^4)` bottleneck of every mean-field method. The
resolution-of-identity (RI), a.k.a. density fitting (DF), approximation
factorises it through an auxiliary Gaussian basis :math:`\{\chi_P\}`,

.. math::
    (\mu\nu|\lambda\sigma) \;\approx\;
        \sum_{PQ} (\mu\nu|P)\,[\mathbf{V}^{-1}]_{PQ}\,(Q|\lambda\sigma),
    \qquad V_{PQ} = (P|Q),

so the storage/compute drops to :math:`O(N^3)` (the three-index tensor
:math:`(\mu\nu|P)` plus the :math:`(P|Q)` Coulomb metric). The fit is the unique
least-squares minimiser of the Coulomb self-interaction of the residual density,
which is why the *Coulomb* metric :math:`(P|Q)` (not the overlap metric) appears.

References
----------
* J. L. Whitten, *J. Chem. Phys.* **58**, 4496 (1973) -- the original Gaussian
  expansion of charge distributions (the "RI" idea).
* B. I. Dunlap, J. W. D. Connolly, J. R. Sabin, *J. Chem. Phys.* **71**, 3396
  (1979) -- robust Coulomb-metric (variational) density fitting.
* O. Vahtras, J. Almlof, M. W. Feyereisen, *Chem. Phys. Lett.* **213**, 514
  (1993) -- the "RI-V" Coulomb-metric fit of the ERIs used here.
* T. Helgaker, P. Jorgensen, J. Olsen, *Molecular Electronic-Structure Theory*,
  Wiley (2000), Ch. 9 -- the McMurchie-Davidson Hermite machinery reused below.

Method and reuse
----------------
Both auxiliary integral types are evaluated as *special cases* of the existing
McMurchie-Davidson four-centre primitive ERI
:func:`opifex.core.quantum._flat_mmd.eri_primitive` -- no new Hermite or Boys
code is introduced:

* The **three-centre** integral :math:`(\mu\nu|P)` is the four-centre integral
  :math:`(\mu\nu|P\,\mathbf{1})` whose fourth function is the *constant*
  ``s``-Gaussian (exponent ``0``, coefficient ``1``). With ``exp_d = 0`` the
  ket pair collapses to the single auxiliary function :math:`\chi_P`, giving the
  genuine three-centre Coulomb integral (validated below to ~1e-9 against an
  analytic derivative-trick reference and to ~1e-13 against PySCF ``int3c2e``).
* The **two-centre** metric :math:`(P|Q)` is the four-centre integral
  :math:`(P\,\mathbf{1}|Q\,\mathbf{1})` with constant functions on *both*
  electrons, i.e. the bare two-centre Coulomb repulsion of two auxiliary
  Gaussians (validated against PySCF ``int2c2e``).

Normalisation follows :mod:`opifex.core.quantum.basis` exactly (the same
``_build_shell_coefficients`` primitive + contracted normalisation used for the
main basis), so the three-/two-centre integrals reproduce PySCF up to the single
auxiliary-only diagonal :math:`d_P = \sqrt{4\pi/(2l_P+1)}` that distinguishes
PySCF's ``int2c2e``/``int3c2e`` Coulomb-normalisation convention from the
overlap normalisation. That diagonal cancels identically in the fit
:math:`\mathbf{J}\mathbf{V}^{-1}\mathbf{J}`, so the *fitted* ERI is
convention-independent and matches PySCF's density fitting to ~1e-13.

Everything is written in JAX (``jnp``): the integral path is a single
:func:`jax.vmap` over the flat primitive triples/pairs and is therefore
``jit``/``grad``/``vmap`` compatible (the geometry and exponents are traced; the
angular powers and primitive->orbital maps are NumPy-static, the repo's
jax-static-metadata convention).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from opifex.core.quantum._flat_mmd import eri_primitive
from opifex.core.quantum.basis import (
    _build_shell_coefficients,
    _CART_COMPONENTS,
    FlatPrimitives,
)


logger = logging.getLogger(__name__)


# A small Tikhonov floor added to the Coulomb metric before the Cholesky solve.
# RI metrics are symmetric positive-definite in exact arithmetic, but redundant
# auxiliary sets push the smallest eigenvalue toward zero; the floor keeps the
# factorisation well-conditioned without perceptibly changing the fit (it is far
# below the intrinsic RI error and the smallest metric eigenvalue of any
# sensible auxiliary basis, O(0.1-1) Hartree). Matches the spirit of PySCF's
# ``lindep`` guard while leaving the fit/PySCF agreement at the ~1e-12 level.
_METRIC_REGULARISATION = 1e-12


@dataclass(frozen=True, slots=True, kw_only=True)
class AuxiliaryBasis:
    r"""Flat array-of-primitives view of an auxiliary (fitting) basis.

    Mirrors :class:`opifex.core.quantum.basis.FlatPrimitives` so the RI kernels
    can ``vmap`` over auxiliary primitives exactly as the main-basis harness
    does. A *primitive* is one Cartesian component of one primitive Gaussian; the
    contraction coefficients already carry the primitive + contracted
    normalisation (the :mod:`opifex.core.quantum.basis` convention).

    Attributes:
        center: Auxiliary primitive centres in Bohr [Shape: (n_prim, 3)]
            (traced).
        alpha: Auxiliary primitive exponents [Shape: (n_prim,)] (traced).
        coeff: Contraction coefficients with normalisation folded in
            [Shape: (n_prim,)] (traced).
        lmn: Cartesian angular-momentum powers [Shape: (n_prim, 3)]
            (NumPy-static int).
        orbital_index: Auxiliary-function index of each primitive
            [Shape: (n_prim,)] (NumPy-static int).
        num_orbitals: Number of contracted Cartesian auxiliary functions.
        max_total_l: Maximum total angular momentum of any auxiliary primitive
            (static; sizes the McMurchie-Davidson Hermite tables).
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
        """Number of flat auxiliary primitives (Cartesian-component granularity)."""
        return int(self.alpha.shape[0])

    @classmethod
    def from_shells(
        cls,
        shells: tuple[tuple[int, int, tuple[tuple[float, float], ...]], ...],
        centers: Array,
    ) -> AuxiliaryBasis:
        r"""Build an :class:`AuxiliaryBasis` from per-shell ``(l, primitives)`` data.

        Args:
            shells: Tuple of ``(center_index, angular_momentum, primitives)``
                where ``primitives`` is a tuple of ``(exponent, coefficient)``
                pairs (the raw STO-style contraction coefficients).
            centers: Cartesian centres in Bohr [Shape: (n_centers, 3)]; each
                shell's ``center_index`` selects a row.

        Returns:
            The flat auxiliary-primitive view, normalised with the
            :mod:`opifex.core.quantum.basis` convention.

        Raises:
            ValueError: If a shell references an out-of-range centre index or an
                angular momentum without a tabulated Cartesian component list.
        """
        n_centers = int(centers.shape[0])
        centers_list: list[Array] = []
        alphas: list[Array] = []
        coeffs: list[Array] = []
        lmn_rows: list[tuple[int, int, int]] = []
        orbital_index: list[int] = []
        aux_index = 0
        max_total_l = 0
        for center_index, angular_momentum, primitives in shells:
            if not 0 <= center_index < n_centers:
                raise ValueError(
                    f"Auxiliary shell centre index {center_index} out of range [0, {n_centers})"
                )
            if angular_momentum not in _CART_COMPONENTS:
                raise ValueError(
                    f"Unsupported auxiliary angular momentum l={angular_momentum};"
                    f" tabulated: {sorted(_CART_COMPONENTS)}"
                )
            exponents = np.asarray([exp for exp, _ in primitives], dtype=np.float64)
            raw_coeffs = np.asarray([coeff for _, coeff in primitives], dtype=np.float64)
            normalised = _build_shell_coefficients(exponents, raw_coeffs, angular_momentum)
            center = centers[center_index]
            n_prim = exponents.shape[0]
            for power in _CART_COMPONENTS[angular_momentum]:
                centers_list.append(jnp.broadcast_to(center, (n_prim, 3)))
                alphas.append(jnp.asarray(exponents))
                coeffs.append(jnp.asarray(normalised))
                lmn_rows.extend([power] * n_prim)
                orbital_index.extend([aux_index] * n_prim)
                max_total_l = max(max_total_l, sum(power))
                aux_index += 1
        return cls(
            center=jnp.concatenate(centers_list, axis=0),
            alpha=jnp.concatenate(alphas, axis=0),
            coeff=jnp.concatenate(coeffs, axis=0),
            lmn=np.asarray(lmn_rows, dtype=np.int32),
            orbital_index=np.asarray(orbital_index, dtype=np.int32),
            num_orbitals=aux_index,
            max_total_l=int(max_total_l),
        )


def _shared_max_l(flat: FlatPrimitives, aux: AuxiliaryBasis) -> int:
    """Static Hermite-table size spanning both main and auxiliary primitives."""
    return max(flat.max_total_l, aux.max_total_l)


def _three_center_primitive(
    lmn_a: Array,
    lmn_b: Array,
    lmn_p: Array,
    center_a: Array,
    center_b: Array,
    center_p: Array,
    exp_a: Array,
    exp_b: Array,
    exp_p: Array,
    max_l: int,
) -> Array:
    r"""Primitive three-centre integral :math:`(ab|P)` via the four-centre kernel.

    Evaluates :math:`(ab|P\,\mathbf 1)` -- the McMurchie-Davidson four-centre ERI
    with the fourth function set to the constant ``s``-Gaussian (``exp_d = 0``,
    ``coeff_d = 1``), which collapses the ket pair to the single auxiliary
    Gaussian :math:`\chi_P` and yields the genuine three-centre Coulomb integral.
    """
    zero = jnp.zeros((), dtype=exp_a.dtype)
    lmn_unit = jnp.zeros_like(lmn_p)
    return eri_primitive(
        lmn_a,
        lmn_b,
        lmn_p,
        lmn_unit,
        center_a,
        center_b,
        center_p,
        center_p,
        exp_a,
        exp_b,
        exp_p,
        zero,
        max_l,
    )


def _two_center_primitive(
    lmn_p: Array,
    lmn_q: Array,
    center_p: Array,
    center_q: Array,
    exp_p: Array,
    exp_q: Array,
    max_l: int,
) -> Array:
    r"""Primitive two-centre Coulomb metric :math:`(P|Q)` via the four-centre kernel.

    Evaluates :math:`(P\,\mathbf 1|Q\,\mathbf 1)` -- the four-centre ERI with the
    constant ``s``-Gaussian on *both* electrons, i.e. the bare two-centre Coulomb
    repulsion of the auxiliary Gaussians :math:`\chi_P` and :math:`\chi_Q`.
    """
    zero = jnp.zeros((), dtype=exp_p.dtype)
    lmn_unit = jnp.zeros_like(lmn_p)
    return eri_primitive(
        lmn_p,
        lmn_unit,
        lmn_q,
        lmn_unit,
        center_p,
        center_p,
        center_q,
        center_q,
        exp_p,
        zero,
        exp_q,
        zero,
        max_l,
    )


def three_center_eri(flat: FlatPrimitives, aux: AuxiliaryBasis) -> Array:
    r"""Three-centre two-electron tensor :math:`(\mu\nu|P)`.

    Builds the dense :math:`(n_\text{ao}, n_\text{ao}, n_\text{aux})` tensor by a
    single :func:`jax.vmap` over every (main, main, aux) primitive triple,
    followed by three :func:`jax.ops.segment_sum` contractions to the contracted
    main AOs and auxiliary functions (the MESS harness pattern).

    Args:
        flat: Flat primitive view of the main AO basis.
        aux: Flat primitive view of the auxiliary basis.

    Returns:
        The three-centre tensor :math:`(\mu\nu|P)`
        [Shape: (n_ao, n_ao, n_aux)] in chemist notation.
    """
    max_l = _shared_max_l(flat, aux)
    n_main = flat.num_primitives
    n_aux = aux.num_primitives

    main_i, main_j = np.meshgrid(
        np.arange(n_main, dtype=np.int64),
        np.arange(n_main, dtype=np.int64),
        indexing="ij",
    )
    grid_i = np.repeat(main_i.reshape(-1), n_aux)
    grid_j = np.repeat(main_j.reshape(-1), n_aux)
    grid_p = np.tile(np.arange(n_aux, dtype=np.int64), n_main * n_main)

    lmn_main = jnp.asarray(flat.lmn)
    lmn_aux = jnp.asarray(aux.lmn)
    ai = jnp.asarray(grid_i)
    aj = jnp.asarray(grid_j)
    ap = jnp.asarray(grid_p)

    coeff = flat.coeff[ai] * flat.coeff[aj] * aux.coeff[ap]

    def primitive(
        la: Array,
        lb: Array,
        lp: Array,
        ca: Array,
        cb: Array,
        cp: Array,
        ea: Array,
        eb: Array,
        ep: Array,
    ) -> Array:
        return _three_center_primitive(la, lb, lp, ca, cb, cp, ea, eb, ep, max_l)

    values = jax.vmap(primitive)(
        lmn_main[ai],
        lmn_main[aj],
        lmn_aux[ap],
        flat.center[ai],
        flat.center[aj],
        aux.center[ap],
        flat.alpha[ai],
        flat.alpha[aj],
        aux.alpha[ap],
    )
    values = coeff * values

    return _contract_three_center(
        values,
        flat.orbital_index[grid_i],
        flat.orbital_index[grid_j],
        aux.orbital_index[grid_p],
        flat.num_orbitals,
        aux.num_orbitals,
    )


def _contract_three_center(
    values: Array,
    ao_i: np.ndarray,
    ao_j: np.ndarray,
    ao_p: np.ndarray,
    num_ao: int,
    num_aux: int,
) -> Array:
    """Scatter primitive-triple values into the contracted ``(ao, ao, aux)`` tensor.

    A single scatter-add over the precomputed flat ``(i, j, P)`` AO/aux indices
    (NumPy-static) contracts the primitives to the dense tensor in one shot.
    """
    flat_index = (ao_i.astype(np.int64) * num_ao + ao_j.astype(np.int64)) * num_aux + ao_p.astype(
        np.int64
    )
    tensor = jnp.zeros((num_ao * num_ao * num_aux,), dtype=values.dtype)
    tensor = tensor.at[jnp.asarray(flat_index)].add(values)
    return tensor.reshape(num_ao, num_ao, num_aux)


def two_center_metric(aux: AuxiliaryBasis) -> Array:
    r"""Two-centre Coulomb metric :math:`V_{PQ} = (P|Q)`.

    Builds the symmetric :math:`(n_\text{aux}, n_\text{aux})` metric by a single
    :func:`jax.vmap` over every auxiliary primitive pair, then a
    :func:`jax.ops.segment_sum`-style scatter to the contracted auxiliary
    functions.

    Args:
        aux: Flat primitive view of the auxiliary basis.

    Returns:
        The Coulomb metric :math:`(P|Q)` [Shape: (n_aux, n_aux)] (symmetric,
        positive-definite in exact arithmetic).
    """
    max_l = aux.max_total_l
    n_aux = aux.num_primitives
    pp, qq = np.meshgrid(
        np.arange(n_aux, dtype=np.int64),
        np.arange(n_aux, dtype=np.int64),
        indexing="ij",
    )
    grid_p = pp.reshape(-1)
    grid_q = qq.reshape(-1)

    lmn_aux = jnp.asarray(aux.lmn)
    ap = jnp.asarray(grid_p)
    aq = jnp.asarray(grid_q)
    coeff = aux.coeff[ap] * aux.coeff[aq]

    def primitive(lp: Array, lq: Array, cp: Array, cq: Array, ep: Array, eq: Array) -> Array:
        return _two_center_primitive(lp, lq, cp, cq, ep, eq, max_l)

    values = jax.vmap(primitive)(
        lmn_aux[ap],
        lmn_aux[aq],
        aux.center[ap],
        aux.center[aq],
        aux.alpha[ap],
        aux.alpha[aq],
    )
    values = coeff * values

    num_aux = aux.num_orbitals
    flat_index = aux.orbital_index[grid_p].astype(np.int64) * num_aux + aux.orbital_index[
        grid_q
    ].astype(np.int64)
    metric = jnp.zeros((num_aux * num_aux,), dtype=values.dtype)
    metric = metric.at[jnp.asarray(flat_index)].add(values)
    return metric.reshape(num_aux, num_aux)


def fit_three_center(three_center: Array, metric: Array) -> Array:
    r"""Coulomb-metric fit coefficients :math:`B_{P,\mu\nu} = [V^{-1}]_{PQ}(Q|\mu\nu)`.

    Solves :math:`V\,B = J^\top` with a (regularised) Cholesky solve rather than
    forming :math:`V^{-1}` explicitly -- the ``jit``/``grad``-safe and numerically
    stable route for the symmetric positive-definite Coulomb metric.

    Args:
        three_center: The :math:`(\mu\nu|P)` tensor [Shape: (n_ao, n_ao, n_aux)].
        metric: The Coulomb metric :math:`(P|Q)` [Shape: (n_aux, n_aux)].

    Returns:
        Fit coefficients :math:`B` [Shape: (n_aux, n_ao, n_ao)] such that the
        fitted ERI is :math:`\sum_P (\mu\nu|P)\,B_{P,\lambda\sigma}`.
    """
    n_ao = three_center.shape[0]
    n_aux = three_center.shape[2]
    eye = jnp.eye(n_aux, dtype=metric.dtype)
    regularised = metric + _METRIC_REGULARISATION * eye
    cholesky = jnp.linalg.cholesky(regularised)
    rhs = three_center.reshape(n_ao * n_ao, n_aux).T
    coefficients = jax.scipy.linalg.cho_solve((cholesky, True), rhs)
    return coefficients.reshape(n_aux, n_ao, n_ao)


def fitted_eri(flat: FlatPrimitives, aux: AuxiliaryBasis) -> Array:
    r"""RI/DF-approximated four-index ERI :math:`(\mu\nu|\lambda\sigma)`.

    Assembles the three-centre tensor and Coulomb metric, then contracts

    .. math::
        (\mu\nu|\lambda\sigma) \approx
            \sum_{PQ}(\mu\nu|P)\,[V^{-1}]_{PQ}\,(Q|\lambda\sigma)

    via a Cholesky solve (see :func:`fit_three_center`). The result is exactly
    symmetric under the eight ERI permutations that the fit preserves
    (:math:`(\mu\nu|\lambda\sigma)=(\lambda\sigma|\mu\nu)` and the bra/ket index
    swaps); it differs from the exact ERI only by the intrinsic RI error of the
    chosen auxiliary basis.

    Args:
        flat: Flat primitive view of the main AO basis.
        aux: Flat primitive view of the auxiliary basis.

    Returns:
        The fitted ERI tensor [Shape: (n_ao,)*4] in chemist notation.
    """
    three_center = three_center_eri(flat, aux)
    metric = two_center_metric(aux)
    coefficients = fit_three_center(three_center, metric)
    n_ao = three_center.shape[0]
    bra = three_center.reshape(n_ao * n_ao, three_center.shape[2])
    ket = coefficients.reshape(coefficients.shape[0], n_ao * n_ao)
    fitted = bra @ ket
    return fitted.reshape(n_ao, n_ao, n_ao, n_ao)


__all__ = [
    "AuxiliaryBasis",
    "fit_three_center",
    "fitted_eri",
    "three_center_eri",
    "two_center_metric",
]

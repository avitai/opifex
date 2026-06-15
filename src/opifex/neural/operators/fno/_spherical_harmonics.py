"""Real spherical harmonic transform (SHT) in JAX.

This module provides a faithful JAX port of the orthonormalized real-valued
spherical harmonic transform from NVIDIA ``torch-harmonics`` for use inside the
Spherical Fourier Neural Operator (SFNO). It replaces the 2D-FFT approximation
that previously stood in for a genuine SHT.

The transform pair is

* **forward (analysis)** -- a real FFT over longitude ``phi`` followed, for each
  azimuthal order ``m``, by a latitude quadrature ``sum_j w_j P_l^m(cos theta_j) ...``
  yielding complex coefficients over ``(l, m)``;
* **inverse (synthesis)** -- a contraction of the coefficients with the associated
  Legendre table over ``l`` followed by an inverse real FFT over ``phi``.

The associated Legendre table ``P_l^m(cos theta_j)`` and the quadrature weights
are precomputed once with NumPy/SciPy at construction time (static), exactly
porting ``torch_harmonics.legendre.legpoly``. The forward/inverse transforms
themselves are pure ``jnp.einsum`` + ``jnp.fft.rfft``/``jnp.fft.irfft`` and are
therefore ``jit`` / ``grad`` / ``vmap`` compatible.

Normalization matches ``torch-harmonics`` with ``norm="ortho"`` and the
Condon-Shortley phase enabled, i.e. orthonormal real spherical harmonics.

References
----------
- Bonev et al. 2023, "Spherical Fourier Neural Operators" (arXiv:2306.03838).
- ``torch_harmonics/sht.py`` (``RealSHT.forward`` lines 119-132,
  ``InverseRealSHT.forward`` lines 215-230).
- ``torch_harmonics/legendre.py`` (``legpoly`` lines 46-119,
  ``clm`` lines 40-42, ``_precompute_legpoly`` lines 122-149).
- ``torch_harmonics/quadrature.py`` (``legendre_gauss_weights`` lines 144-172).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float  # noqa: TC002 — kept eager per opifex convention
from scipy.special import roots_legendre


# Supported latitude quadrature grids. Gauss-Legendre is the torch-harmonics
# default for an exact SHT (``sht.py`` line 88).
_GRID_LEGENDRE_GAUSS = "legendre-gauss"
_SUPPORTED_GRIDS = (_GRID_LEGENDRE_GAUSS,)


def _legpoly(mmax: int, lmax: int, x: np.ndarray) -> np.ndarray:
    r"""Compute ``(-1)^m c_l^m P_l^m(x)`` for orthonormal real spherical harmonics.

    Faithful NumPy port of ``torch_harmonics.legendre.legpoly`` (``legendre.py``
    lines 46-119) for ``norm="ortho"``, ``inverse=False`` and the Condon-Shortley
    phase enabled. The returned tensor has shape ``(mmax, lmax, len(x))``.

    The three-term recurrence is sequential in the degree ``l`` (each ``l`` reads
    ``l-1`` and ``l-2``); for fixed ``l`` all orders ``m`` are independent and are
    updated as a single vectorized slice, mirroring the upstream implementation.

    Args:
        mmax: Maximum azimuthal order ``m`` (non-inclusive).
        lmax: Maximum spherical harmonic degree ``l`` (non-inclusive).
        x: Sample positions ``cos(theta)`` of shape ``(nlat,)``.

    Returns:
        Associated Legendre table of shape ``(mmax, lmax, nlat)``.
    """
    n_modes = max(mmax, lmax)
    vandermonde = np.zeros((n_modes, n_modes, len(x)), dtype=np.float64)

    # norm == "ortho" -> norm_factor == 1.0 (legendre.py lines 88-91).
    vandermonde[0, 0, :] = 1.0 / math.sqrt(4 * math.pi)

    # Diagonal and sub-diagonal seeds: sequential in l but only O(n_modes) ops.
    for degree in range(1, n_modes):
        vandermonde[degree - 1, degree, :] = (
            math.sqrt(2 * degree + 1) * x * vandermonde[degree - 1, degree - 1, :]
        )
        vandermonde[degree, degree, :] = (
            np.sqrt((2 * degree + 1) * (1 + x) * (1 - x) / 2 / degree)
            * vandermonde[degree - 1, degree - 1, :]
        )

    # Three-term recurrence, vectorized across m for each fixed l
    # (legendre.py lines 100-104).
    for degree in range(2, n_modes):
        orders = np.arange(0, degree - 1, dtype=np.float64)
        a_lm = np.sqrt((2 * degree - 1) / (degree - orders) * (2 * degree + 1) / (degree + orders))
        b_lm = np.sqrt(
            (degree + orders - 1)
            / (degree - orders)
            * (2 * degree + 1)
            / (2 * degree - 3)
            * (degree - orders - 1)
            / (degree + orders)
        )
        vandermonde[: degree - 1, degree, :] = (
            a_lm[:, None] * x[None, :] * vandermonde[: degree - 1, degree - 1, :]
            - b_lm[:, None] * vandermonde[: degree - 1, degree - 2, :]
        )

    vandermonde = vandermonde[:mmax, :lmax]

    # Condon-Shortley phase (-1)^m (legendre.py lines 116-117).
    vandermonde[1::2] *= -1

    return vandermonde


@dataclass(frozen=True, slots=True)
class SphericalHarmonicBasis:
    """Precomputed real spherical harmonic transform on a fixed lat/lon grid.

    Construction precomputes the associated Legendre table and quadrature weights
    with NumPy/SciPy (static). :meth:`forward` and :meth:`inverse` are pure JAX
    and are safe under ``jit`` / ``grad`` / ``vmap``.

    The coefficient layout is ``(..., lmax, mmax)`` with non-negative orders ``m``
    only (the negative orders are redundant for a real field and are recovered by
    the real FFT), matching ``torch_harmonics.RealSHT``.

    Args:
        nlat: Number of latitude (colatitude) grid points.
        nlon: Number of longitude grid points.
        lmax: Maximum spherical harmonic degree ``+ 1`` (non-inclusive). If
            ``None``, defaults to ``nlat`` (Gauss-Legendre exactness, ``sht.py``).
        mmax: Maximum azimuthal order ``+ 1`` (non-inclusive). If ``None``, defaults
            to ``lmax`` (triangular truncation, ``truncation.py``).
        grid: Latitude quadrature grid. Only ``"legendre-gauss"`` is supported.

    Raises:
        ValueError: If ``grid`` is not a supported quadrature grid.
    """

    nlat: int
    nlon: int
    lmax: int
    mmax: int
    # Forward weights P_l^m(cos theta_j) * w_j, shape (mmax, lmax, nlat). Stored as
    # a concrete NumPy constant (not a jax.Array): the basis is cached across calls
    # (_get_spherical_basis), so a jax.Array built here would -- on first use inside
    # a jit trace -- be a trace-scoped constant that then leaks into every later
    # trace (UnexpectedTracerError). NumPy data is trace-agnostic and is rematerialised
    # as a fresh per-trace constant by forward()/inverse().
    _forward_weights: Float[np.ndarray, "mmax lmax nlat"]
    # Inverse Legendre table P_l^m(cos theta_j), shape (mmax, lmax, nlat).
    _legendre: Float[np.ndarray, "mmax lmax nlat"]

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int | None = None,
        mmax: int | None = None,
        grid: str = _GRID_LEGENDRE_GAUSS,
    ) -> None:
        if grid not in _SUPPORTED_GRIDS:
            raise ValueError(f"Unsupported SHT grid {grid!r}; supported grids: {_SUPPORTED_GRIDS}")

        # Triangular truncation matching torch_harmonics.truncation.truncate_sht:
        # Gauss-Legendre exactness gives lmax == nlat, then lmax = mmax = min(...).
        resolved_lmax = lmax if lmax is not None else nlat
        resolved_mmax = mmax if mmax is not None else resolved_lmax
        resolved_lmax = min(resolved_lmax, resolved_mmax, nlat)
        resolved_mmax = resolved_lmax

        # Gauss-Legendre nodes/weights on [-1, 1] (quadrature.py:legendre_gauss_weights).
        cost, weights = roots_legendre(nlat)
        # torch-harmonics flips arccos(cost) so latitudes ascend, and flips the
        # weights to match (sht.py line 98, quadrature.py lines 104-105).
        order = np.argsort(np.arccos(cost))
        cost = cost[order]
        weights = weights[order]

        legendre = _legpoly(resolved_mmax, resolved_lmax, cost)  # (m, l, nlat)
        # Fold quadrature weights into the forward operator (sht.py line 105).
        forward_weights = np.einsum("mlk,k->mlk", legendre, weights)

        object.__setattr__(self, "nlat", nlat)
        object.__setattr__(self, "nlon", nlon)
        object.__setattr__(self, "lmax", resolved_lmax)
        object.__setattr__(self, "mmax", resolved_mmax)
        # Keep the precomputed tables as concrete NumPy constants (not jax.Array):
        # the cached basis must stay trace-agnostic (see the field comments).
        object.__setattr__(self, "_forward_weights", np.asarray(forward_weights, dtype=np.float32))
        object.__setattr__(self, "_legendre", np.asarray(legendre, dtype=np.float32))

    def forward(
        self, field: Float[Array, "*batch nlat nlon"]
    ) -> Complex[Array, "*batch lmax mmax"]:
        """Forward (analysis) real SHT applied to the last two axes.

        Mirrors ``torch_harmonics.RealSHT.forward`` (``sht.py`` lines 119-132):
        a longitude real FFT scaled by ``2 pi`` (``norm="forward"``), followed by a
        latitude quadrature contraction with the precomputed Legendre weights.

        Args:
            field: Real spherical field with trailing shape ``(nlat, nlon)``.

        Returns:
            Complex spherical harmonic coefficients with trailing shape
            ``(lmax, mmax)``.
        """
        # Real FFT over longitude; norm="forward" divides by nlon, the 2*pi factor
        # turns the discrete sum into the longitude integral (sht.py line 120).
        spectrum = 2.0 * math.pi * jnp.fft.rfft(field, axis=-1, norm="forward")
        spectrum = spectrum[..., : self.mmax]  # keep orders 0..mmax-1

        weights = self._forward_weights.astype(spectrum.real.dtype)
        # spectrum trailing axes are (nlat=k, order=m); contract latitude k against
        # the (m, l, k) Legendre weights -> (..., l, m) (sht.py lines 123-130).
        coeff_real = jnp.einsum("...km,mlk->...lm", spectrum.real, weights)
        coeff_imag = jnp.einsum("...km,mlk->...lm", spectrum.imag, weights)
        return jax_complex(coeff_real, coeff_imag)

    def inverse(
        self, coeffs: Complex[Array, "*batch lmax mmax"]
    ) -> Float[Array, "*batch nlat nlon"]:
        """Inverse (synthesis) real SHT producing a field on the last two axes.

        Mirrors ``torch_harmonics.InverseRealSHT.forward`` (``sht.py`` lines
        215-230): a Legendre contraction over degree ``l`` followed by an inverse
        real FFT over longitude (``norm="forward"``).

        Args:
            coeffs: Complex spherical harmonic coefficients with trailing shape
                ``(lmax, mmax)``.

        Returns:
            Real spherical field with trailing shape ``(nlat, nlon)``.
        """
        legendre = self._legendre.astype(coeffs.real.dtype)
        # Contract degree l against (m, l, k) -> (..., k, m) (sht.py lines 223-224).
        field_real = jnp.einsum("...lm,mlk->...km", coeffs.real, legendre)
        field_imag = jnp.einsum("...lm,mlk->...km", coeffs.imag, legendre)
        spectrum = jax_complex(field_real, field_imag)
        # Inverse real FFT over longitude (sht.py line 228).
        return jnp.fft.irfft(spectrum, n=self.nlon, axis=-1, norm="forward")


def jax_complex(real: Array, imag: Array) -> Array:
    """Assemble a complex array from real and imaginary parts.

    Args:
        real: Real part.
        imag: Imaginary part (matching shape and dtype).

    Returns:
        Complex array ``real + 1j * imag``.
    """
    return jnp.asarray(real) + 1j * jnp.asarray(imag)


__all__ = ["SphericalHarmonicBasis"]

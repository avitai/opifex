r"""Radial bases and cutoff envelopes for E(3)-equivariant networks.

Interatomic-distance edge features are *invariant* scalars (they depend only on
``|r_i - r_j|``), so these objects return plain arrays rather than
:class:`~opifex.neural.equivariant.IrrepsArray`; downstream they are tagged as
``0e`` channels.

* :class:`BesselBasis` -- the Bessel basis of MACE (Batatia et al. 2022, "MACE:
  Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate
  Force Fields", arXiv:2206.07697), eq. 7.
* :class:`GaussianBasis` -- Gaussians on evenly spaced centres.
* :func:`polynomial_cutoff` -- the polynomial envelope of MACE, eq. 8.

The cosine envelope :func:`cosine_cutoff` follows Behler (J. Chem. Phys. 134,
074106, 2011), the original ACSF cutoff function.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002


class BesselBasis(nnx.Module):
    r"""Bessel radial basis ``b_n(r) = sqrt(2/r_c) sin(n pi r / r_c) / r``.

    MACE eq. 7 (Batatia et al. 2022, arXiv:2206.07697).  The frequencies
    ``n pi / r_c`` for ``n = 1..num_basis`` are stored as a (non-trainable) buffer;
    the basis is a smooth, complete set of invariant radial features.
    """

    def __init__(self, num_basis: int, cutoff: float, *, rngs: nnx.Rngs | None = None) -> None:
        """Build the Bessel basis.

        Args:
            num_basis: Number of Bessel functions ``n = 1..num_basis``.
            cutoff: Cutoff radius ``r_c`` (must be positive).
            rngs: Unused (the basis has no learnable parameters); accepted for a
                uniform constructor signature across equivariant modules.
        """
        super().__init__()
        if num_basis < 1:
            raise ValueError(f"num_basis must be >= 1, got {num_basis}")
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        del rngs
        self.num_basis = num_basis
        self.cutoff = cutoff
        self._frequencies = jnp.asarray(math.pi / cutoff * np.linspace(1.0, num_basis, num_basis))
        self._prefactor = math.sqrt(2.0 / cutoff)

    def __call__(self, radius: Float[Array, ...]) -> Float[Array, "... num_basis"]:
        """Evaluate the Bessel basis at the given distances.

        Args:
            radius: Interatomic distances of shape ``(...)`` (positive).

        Returns:
            Array of shape ``(..., num_basis)``.
        """
        scaled = radius[..., None] * self._frequencies
        return self._prefactor * (jnp.sin(scaled) / radius[..., None])


class GaussianBasis(nnx.Module):
    r"""Gaussian radial basis ``g_n(r) = exp(-(r - mu_n)^2 / (2 sigma^2))``.

    Gaussian radial features.  Centres ``mu_n`` are
    evenly spaced on ``[0, r_c]`` and the width is the centre spacing
    (``sigma = r_c / (num_basis - 1)``), giving the coefficient
    ``coeff = -0.5 / sigma^2``.
    """

    def __init__(self, num_basis: int, cutoff: float, *, rngs: nnx.Rngs | None = None) -> None:
        """Build the Gaussian basis.

        Args:
            num_basis: Number of Gaussians (centres on ``[0, r_c]``); must be
                ``>= 2`` so the spacing is well defined.
            cutoff: Cutoff radius ``r_c`` (positive); sets the centre range.
            rngs: Unused (no learnable parameters); accepted for a uniform
                constructor signature.
        """
        super().__init__()
        if num_basis < 2:
            raise ValueError(f"num_basis must be >= 2, got {num_basis}")
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        del rngs
        self.num_basis = num_basis
        self.cutoff = cutoff
        self._centres = jnp.asarray(np.linspace(0.0, cutoff, num_basis))
        self._coefficient = -0.5 / (cutoff / (num_basis - 1)) ** 2

    def __call__(self, radius: Float[Array, ...]) -> Float[Array, "... num_basis"]:
        """Evaluate the Gaussian basis at the given distances.

        Args:
            radius: Interatomic distances of shape ``(...)``.

        Returns:
            Array of shape ``(..., num_basis)``.
        """
        shifted = radius[..., None] - self._centres
        return jnp.exp(self._coefficient * shifted**2)


class PiecewiseLinearBasis(nnx.Module):
    r"""Piecewise-linear (hat) radial basis on ``[0, cutoff]``.

    An isotropic piecewise-linear filter basis for discrete-continuous (DISCO) convolutions
    (Ocampo, Price & McEwen 2023, ``arXiv:2209.13603``). The layer builds ``num_basis`` hat
    functions, which have collocation spacing ``dr = 2 * cutoff / (num_basis + 1)``
    and half-width ``dr``: ``phi_k(r) = max(0, 1 - |r - r_k| / dr)`` restricted to ``r <= cutoff``,
    with centres ``r_k = k * dr`` (odd ``num_basis``) or ``(k + 0.5) * dr`` (even). Each function is
    continuous and compactly supported, so no separate cutoff envelope is needed.
    """

    def __init__(self, num_basis: int, cutoff: float, *, rngs: nnx.Rngs | None = None) -> None:
        """Build the piecewise-linear basis.

        Args:
            num_basis: Number of hat functions (>= 1).
            cutoff: Support radius ``r_c`` (positive).
            rngs: Unused (the basis is parameter-free); accepted for interface uniformity.
        """
        super().__init__()
        del rngs
        if num_basis < 1:
            raise ValueError(f"num_basis must be >= 1, got {num_basis}")
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        self.num_basis = num_basis
        self.cutoff = cutoff
        self._dr = 2.0 * cutoff / (num_basis + 1)
        offset = 0.0 if num_basis % 2 == 1 else 0.5
        self._centres = jnp.asarray((np.arange(num_basis) + offset) * self._dr)

    def __call__(self, radius: Float[Array, ...]) -> Float[Array, "... num_basis"]:
        """Evaluate the hat basis at the given distances.

        Args:
            radius: Distances of shape ``(...)``.

        Returns:
            Array of shape ``(..., num_basis)``.
        """
        distance = jnp.abs(radius[..., None] - self._centres)
        values = 1.0 - distance / self._dr
        support = (distance <= self._dr) & (radius[..., None] <= self.cutoff)
        return jnp.where(support, values, 0.0)


def polynomial_cutoff(radius: Float[Array, ...], cutoff: float, *, p: int = 6) -> Float[Array, ...]:
    r"""Smooth polynomial cutoff envelope decaying from ``1`` to ``0`` on ``[0, r_c]``.

    MACE eq. 8 (Batatia et al. 2022, arXiv:2206.07697).  The envelope
    and its first ``p`` derivatives vanish at ``r_c``::

        f(r) = 1 - (p+1)(p+2)/2 (r/r_c)^p
                 + p(p+2) (r/r_c)^{p+1}
                 - p(p+1)/2 (r/r_c)^{p+2}

    masked to zero for ``r >= r_c``.

    Args:
        radius: Distances of shape ``(...)``.
        cutoff: Cutoff radius ``r_c`` (positive).
        p: Polynomial order controlling smoothness (default ``6``).

    Returns:
        Envelope values of shape ``(...)`` in ``[0, 1]``.
    """
    ratio = radius / cutoff
    envelope = (
        1.0
        - (p + 1.0) * (p + 2.0) / 2.0 * ratio**p
        + p * (p + 2.0) * ratio ** (p + 1)
        - p * (p + 1.0) / 2.0 * ratio ** (p + 2)
    )
    return envelope * (radius < cutoff)


def cosine_cutoff(radius: Float[Array, ...], cutoff: float) -> Float[Array, ...]:
    r"""Behler cosine cutoff envelope ``0.5 (cos(pi r / r_c) + 1)``.

    From Behler (J. Chem. Phys. 134, 074106, 2011), the original atom-centred
    symmetry-function cutoff.  Decays smoothly from ``1`` at ``r = 0`` to ``0``
    at ``r = r_c`` (with vanishing first derivative there) and is masked to zero
    beyond ``r_c``.

    Args:
        radius: Distances of shape ``(...)``.
        cutoff: Cutoff radius ``r_c`` (positive).

    Returns:
        Envelope values of shape ``(...)`` in ``[0, 1]``.
    """
    envelope = 0.5 * (jnp.cos(jnp.pi * radius / cutoff) + 1.0)
    return envelope * (radius < cutoff)

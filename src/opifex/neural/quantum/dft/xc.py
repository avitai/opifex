r"""Local-density-approximation (LDA) exchange-correlation functional.

Provides the spin-unpolarised LDA exchange-correlation energy density used by the
restricted Kohn-Sham solver in :mod:`opifex.neural.quantum.dft.scf`:

* **Slater (Dirac) exchange** -- the uniform-electron-gas exchange energy per
  particle :math:`\varepsilon_x(\rho) = -C_x\,\rho^{1/3}` with
  :math:`C_x = \tfrac34 (3/\pi)^{1/3}`.
* **VWN5 correlation** -- the Vosko-Wilk-Nusair (1980) parametrisation of the
  Ceperley-Alder uniform-gas correlation energy (their fit V; libxc/PySCF code 7,
  selected by ``'lda,vwn'``).

Everything is written in JAX so the functional is differentiable and the
exchange-correlation potential :math:`v_{xc} = d(\rho\varepsilon_{xc})/d\rho` is
obtained by automatic differentiation rather than a hand-coded derivative.

References
----------
* P. A. M. Dirac, *Proc. Cambridge Philos. Soc.* **26**, 376 (1930) (exchange).
* S. H. Vosko, L. Wilk, M. Nusair, *Can. J. Phys.* **58**, 1200 (1980),
  eq. 4.4 and Table 5 (paramagnetic fit) -- the VWN5 correlation parametrisation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


# Slater exchange constant C_x = (3/4) (3/pi)^(1/3).
_EXCHANGE_CONSTANT = 0.75 * (3.0 / jnp.pi) ** (1.0 / 3.0)

# VWN5 paramagnetic fit parameters (Vosko-Wilk-Nusair 1980, Table 5).
# These reproduce the Ceperley-Alder uniform-gas correlation energy.
_VWN_A = 0.0310907
_VWN_B = 3.72744
_VWN_C = 12.9352
_VWN_X0 = -0.10498

# Density floor to keep r_s and logarithms finite in vacuum regions.
_DENSITY_FLOOR = 1.0e-12


def slater_exchange_energy_density(density: Array) -> Array:
    r"""Slater exchange energy per particle :math:`\varepsilon_x(\rho)`.

    Args:
        density: Total electron density ``rho`` [any shape].

    Returns:
        Exchange energy per particle ``-C_x rho^(1/3)`` (same shape).
    """
    safe = jnp.clip(density, _DENSITY_FLOOR, None)
    return -_EXCHANGE_CONSTANT * safe ** (1.0 / 3.0)


def vwn_correlation_energy_density(density: Array) -> Array:
    r"""VWN5 correlation energy per particle :math:`\varepsilon_c(\rho)`.

    Implements the Vosko-Wilk-Nusair (1980) eq. 4.4 closed form for the
    paramagnetic (spin-unpolarised) electron gas:

    .. math::
        \varepsilon_c = A\Big[
            \ln\frac{x^2}{X(x)}
            + \frac{2b}{Q}\arctan\frac{Q}{2x+b}
            - \frac{b x_0}{X(x_0)}\Big(
                \ln\frac{(x-x_0)^2}{X(x)}
                + \frac{2(b+2x_0)}{Q}\arctan\frac{Q}{2x+b}\Big)\Big],

    with :math:`x=\sqrt{r_s}`, :math:`X(x)=x^2+bx+c`,
    :math:`Q=\sqrt{4c-b^2}` and the Wigner-Seitz radius
    :math:`r_s=(3/4\pi\rho)^{1/3}`.

    Args:
        density: Total electron density ``rho`` [any shape].

    Returns:
        Correlation energy per particle (same shape).
    """
    safe = jnp.clip(density, _DENSITY_FLOOR, None)
    r_s = (3.0 / (4.0 * jnp.pi * safe)) ** (1.0 / 3.0)
    x = jnp.sqrt(r_s)

    big_x = x * x + _VWN_B * x + _VWN_C
    big_x0 = _VWN_X0 * _VWN_X0 + _VWN_B * _VWN_X0 + _VWN_C
    q = jnp.sqrt(4.0 * _VWN_C - _VWN_B * _VWN_B)

    atan_term = jnp.arctan(q / (2.0 * x + _VWN_B))
    first = jnp.log(x * x / big_x) + 2.0 * _VWN_B / q * atan_term
    second = (
        _VWN_B
        * _VWN_X0
        / big_x0
        * (jnp.log((x - _VWN_X0) ** 2 / big_x) + 2.0 * (_VWN_B + 2.0 * _VWN_X0) / q * atan_term)
    )
    return _VWN_A * (first - second)


def lda_energy_density(density: Array) -> Array:
    r"""LDA exchange-correlation energy per particle :math:`\varepsilon_{xc}`.

    Args:
        density: Total electron density ``rho`` [any shape].

    Returns:
        ``epsilon_x + epsilon_c`` (same shape).
    """
    return slater_exchange_energy_density(density) + vwn_correlation_energy_density(density)


def lda_exchange_correlation_potential(density: Array) -> Array:
    r"""LDA XC potential :math:`v_{xc} = d(\rho\,\varepsilon_{xc})/d\rho`.

    Computed by automatic differentiation of the XC energy density
    :math:`\rho\,\varepsilon_{xc}(\rho)`.

    Args:
        density: Total electron density ``rho`` [Shape: (n_points,)].

    Returns:
        XC potential at each point [Shape: (n_points,)].
    """

    def energy_density(rho: Array) -> Array:
        return rho * lda_energy_density(rho)

    return jax.vmap(jax.grad(energy_density))(density)


__all__ = [
    "lda_energy_density",
    "lda_exchange_correlation_potential",
    "slater_exchange_energy_density",
    "vwn_correlation_energy_density",
]

r"""LDA and PBE (GGA) exchange-correlation functionals.

Provides the spin-unpolarised exchange-correlation energy densities used by the
restricted Kohn-Sham solver in :mod:`opifex.neural.quantum.dft.scf`.

**Local density approximation (LDA)**

* **Slater (Dirac) exchange** -- the uniform-electron-gas exchange energy per
  particle :math:`\varepsilon_x(\rho) = -C_x\,\rho^{1/3}` with
  :math:`C_x = \tfrac34 (3/\pi)^{1/3}`.
* **VWN5 correlation** -- the Vosko-Wilk-Nusair (1980) parametrisation of the
  Ceperley-Alder uniform-gas correlation energy (their fit V; libxc/PySCF code 7,
  selected by ``'lda,vwn'``).

**Generalised gradient approximation (PBE)**

* **PBE exchange** -- the uniform exchange times the enhancement factor
  :math:`F_x(s) = 1 + \kappa - \kappa/(1+\mu s^2/\kappa)` with
  :math:`\kappa=0.804`, :math:`\mu=0.2195149727645171` and the reduced gradient
  :math:`s = |\nabla\rho|/(2 k_F\rho)`, :math:`k_F=(3\pi^2\rho)^{1/3}` (PRL 77,
  3865 (1996), eq. 13-14).
* **PBE correlation** -- the PW92 uniform correlation plus the gradient
  correction :math:`H(r_s,t)` (eq. 7-8). The uniform part is the Perdew-Wang
  (1992) ``lda_c_pw_mod`` fit (:math:`A=0.0310907`), which is what libxc's
  ``gga_c_pbe`` uses internally.

Everything is written in JAX so the functionals are differentiable and the
exchange-correlation potential is obtained by automatic differentiation rather
than a hand-coded derivative. For the LDA the potential is
:math:`v_{xc} = d(\rho\varepsilon_{xc})/d\rho`; for the GGA it is the pair
:math:`(\partial(\rho\varepsilon)/\partial\rho,\;
\partial(\rho\varepsilon)/\partial\sigma)` with :math:`\sigma=|\nabla\rho|^2`.

References
----------
* P. A. M. Dirac, *Proc. Cambridge Philos. Soc.* **26**, 376 (1930) (exchange).
* S. H. Vosko, L. Wilk, M. Nusair, *Can. J. Phys.* **58**, 1200 (1980),
  eq. 4.4 and Table 5 (paramagnetic fit) -- the VWN5 correlation parametrisation.
* J. P. Perdew, Y. Wang, *Phys. Rev. B* **45**, 13244 (1992), Table I --
  the uniform-gas correlation fit (``A=0.0310907`` ``pw_mod`` variant).
* J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996),
  eq. 7-8 (correlation ``H``) and eq. 13-14 (exchange factor ``F_x``).
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

# PBE exchange parameters (PRL 77, 3865 (1996)).
_PBE_KAPPA = 0.8040
_PBE_MU = 0.2195149727645171

# PBE correlation parameters: gamma = (1 - ln 2)/pi^2, beta from eq. (9).
_PBE_GAMMA = (1.0 - jnp.log(2.0)) / jnp.pi**2
_PBE_BETA = 0.06672455060314922

# Perdew-Wang (1992) uniform-gas correlation fit constants. These are the
# ``lda_c_pw_mod`` values libxc's ``gga_c_pbe`` uses (A=0.0310907, not 0.031091).
_PW92_A = 0.0310907
_PW92_ALPHA1 = 0.21370
_PW92_BETA1 = 7.5957
_PW92_BETA2 = 3.5876
_PW92_BETA3 = 1.6382
_PW92_BETA4 = 0.49294

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


def pw92_correlation_energy_density(density: Array) -> Array:
    r"""Perdew-Wang (1992) uniform-gas correlation energy per particle.

    Implements the closed-form fit (Perdew & Wang 1992, eq. 10)

    .. math::
        \varepsilon_c^{\text{unif}}(r_s) = -2A(1+\alpha_1 r_s)\,
            \ln\!\Big(1 + \frac{1}{2A(\beta_1 r_s^{1/2}+\beta_2 r_s
            +\beta_3 r_s^{3/2}+\beta_4 r_s^2)}\Big),

    with the spin-unpolarised ``pw_mod`` constants (``A=0.0310907``) used by
    libxc's ``gga_c_pbe``. Provides the uniform reference for PBE correlation.

    Args:
        density: Total electron density ``rho`` [any shape].

    Returns:
        Uniform-gas correlation energy per particle (same shape).
    """
    safe = jnp.clip(density, _DENSITY_FLOOR, None)
    r_s = (3.0 / (4.0 * jnp.pi * safe)) ** (1.0 / 3.0)
    rs_half = jnp.sqrt(r_s)
    denominator = (
        2.0
        * _PW92_A
        * (
            _PW92_BETA1 * rs_half
            + _PW92_BETA2 * r_s
            + _PW92_BETA3 * r_s * rs_half
            + _PW92_BETA4 * r_s * r_s
        )
    )
    return -2.0 * _PW92_A * (1.0 + _PW92_ALPHA1 * r_s) * jnp.log1p(1.0 / denominator)


def pbe_exchange_energy_density(density: Array, sigma: Array) -> Array:
    r"""PBE exchange energy per particle :math:`\varepsilon_x^{\text{PBE}}`.

    The uniform-gas exchange is enhanced by the gradient-dependent factor
    :math:`F_x(s)` (PRL 77, 3865 (1996), eq. 13-14):

    .. math::
        \varepsilon_x^{\text{PBE}} = \varepsilon_x^{\text{unif}}\,F_x(s),\quad
        F_x(s) = 1 + \kappa - \frac{\kappa}{1 + \mu s^2/\kappa},\quad
        s = \frac{|\nabla\rho|}{2 k_F \rho},\; k_F = (3\pi^2\rho)^{1/3}.

    Args:
        density: Total electron density ``rho`` [any shape].
        sigma: Squared density gradient ``|grad rho|^2`` [same shape].

    Returns:
        PBE exchange energy per particle (same shape).
    """
    safe = jnp.clip(density, _DENSITY_FLOOR, None)
    safe_sigma = jnp.clip(sigma, 0.0, None)
    exchange_unif = -_EXCHANGE_CONSTANT * safe ** (1.0 / 3.0)
    k_fermi = (3.0 * jnp.pi**2 * safe) ** (1.0 / 3.0)
    s_squared = safe_sigma / (4.0 * k_fermi**2 * safe**2)
    enhancement = 1.0 + _PBE_KAPPA - _PBE_KAPPA / (1.0 + _PBE_MU * s_squared / _PBE_KAPPA)
    return exchange_unif * enhancement


def pbe_correlation_energy_density(density: Array, sigma: Array) -> Array:
    r"""PBE correlation energy per particle :math:`\varepsilon_c^{\text{PBE}}`.

    Adds the gradient correction :math:`H` to the PW92 uniform correlation
    (PRL 77, 3865 (1996), eq. 7-8; here for the unpolarised case
    :math:`\phi=1`):

    .. math::
        H = \gamma\,\ln\!\Big[1 + \frac{\beta}{\gamma} t^2
            \frac{1+At^2}{1+At^2+A^2t^4}\Big],\quad
        A = \frac{\beta}{\gamma}\Big[e^{-\varepsilon_c^{\text{unif}}/\gamma}-1
            \Big]^{-1},\quad
        t = \frac{|\nabla\rho|}{2 k_s \rho},\; k_s = \sqrt{4 k_F/\pi}.

    Args:
        density: Total electron density ``rho`` [any shape].
        sigma: Squared density gradient ``|grad rho|^2`` [same shape].

    Returns:
        PBE correlation energy per particle (same shape).
    """
    safe = jnp.clip(density, _DENSITY_FLOOR, None)
    safe_sigma = jnp.clip(sigma, 0.0, None)
    correlation_unif = pw92_correlation_energy_density(safe)
    k_fermi = (3.0 * jnp.pi**2 * safe) ** (1.0 / 3.0)
    k_screen = jnp.sqrt(4.0 * k_fermi / jnp.pi)
    t_squared = safe_sigma / (4.0 * k_screen**2 * safe**2)

    a_factor = (_PBE_BETA / _PBE_GAMMA) / jnp.expm1(-correlation_unif / _PBE_GAMMA)
    at2 = a_factor * t_squared
    ratio = (1.0 + at2) / (1.0 + at2 + at2 * at2)
    gradient_correction = _PBE_GAMMA * jnp.log1p((_PBE_BETA / _PBE_GAMMA) * t_squared * ratio)
    return correlation_unif + gradient_correction


def pbe_energy_density(density: Array, sigma: Array) -> Array:
    r"""PBE exchange-correlation energy per particle :math:`\varepsilon_{xc}`.

    Args:
        density: Total electron density ``rho`` [any shape].
        sigma: Squared density gradient ``|grad rho|^2`` [same shape].

    Returns:
        ``epsilon_x^PBE + epsilon_c^PBE`` (same shape).
    """
    return pbe_exchange_energy_density(density, sigma) + pbe_correlation_energy_density(
        density, sigma
    )


def pbe_exchange_correlation_potential(density: Array, sigma: Array) -> tuple[Array, Array]:
    r"""GGA XC potential components for PBE, by automatic differentiation.

    Returns the two functional derivatives of the XC energy density
    :math:`\rho\,\varepsilon_{xc}(\rho,\sigma)` needed to assemble the GGA Fock
    contribution:

    .. math::
        v_\rho = \frac{\partial(\rho\varepsilon_{xc})}{\partial\rho},\qquad
        v_\sigma = \frac{\partial(\rho\varepsilon_{xc})}{\partial\sigma}.

    Both are obtained with :func:`jax.grad` rather than a hand-coded derivative.

    Args:
        density: Total electron density ``rho`` [Shape: (n_points,)].
        sigma: Squared density gradient ``|grad rho|^2`` [Shape: (n_points,)].

    Returns:
        A pair ``(v_rho, v_sigma)`` each [Shape: (n_points,)].
    """

    def energy_density(rho: Array, sig: Array) -> Array:
        return rho * pbe_energy_density(rho, sig)

    grad_rho = jax.vmap(jax.grad(energy_density, argnums=0))(density, sigma)
    grad_sigma = jax.vmap(jax.grad(energy_density, argnums=1))(density, sigma)
    return grad_rho, grad_sigma


__all__ = [
    "lda_energy_density",
    "lda_exchange_correlation_potential",
    "pbe_correlation_energy_density",
    "pbe_energy_density",
    "pbe_exchange_correlation_potential",
    "pbe_exchange_energy_density",
    "pw92_correlation_energy_density",
    "slater_exchange_energy_density",
    "vwn_correlation_energy_density",
]

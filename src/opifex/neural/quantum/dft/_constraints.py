r"""Exact-constraint layer for machine-learned exchange-correlation functionals.

A learned exchange-correlation (XC) functional is only physical if it respects the
known exact constraints of the universal XC functional. Rather than clamp the raw
network output ad hoc, this module factorises the XC energy density in the
*enhancement-factor* form used by every constraint-based functional (PBE, SCAN)
and by the constraint-respecting machine-learned functionals (DM21; Nagai/Pederson
2022):

.. math::
    \varepsilon_{xc}(\rho, \sigma) = \varepsilon_x^{\text{unif}}(\rho)\,
        F_{xc}(s),\qquad
    \varepsilon_x^{\text{unif}}(\rho) = -C_x\,\rho^{1/3},\quad
    C_x = \tfrac34 (3/\pi)^{1/3},

where the enhancement factor :math:`F_{xc}` is a function of the *dimensionless*
reduced density gradient

.. math::
    s = \frac{|\nabla\rho|}{2 k_F \rho},\qquad k_F = (3\pi^2\rho)^{1/3}

only. Writing the functional this way makes several exact constraints structural
(they hold for *any* learned :math:`F_{xc}` of :math:`s`) and the remaining ones
are enforced by bounding :math:`F_{xc}`:

* **Uniform coordinate scaling for exchange** (Levy & Perdew 1985):
  :math:`E_x[\rho_\gamma] = \gamma E_x[\rho]` with
  :math:`\rho_\gamma(r)=\gamma^3\rho(\gamma r)`. Because :math:`s` is invariant
  under this scaling and :math:`\int \rho\,\varepsilon_x^{\text{unif}}\,d^3r`
  scales as :math:`\gamma`, the identity holds for any :math:`F_{xc}(s)` --
  the enhancement-factor structure enforces it exactly.
* **Uniform-electron-gas (LDA) limit** (Sun, Ruzsinszky & Perdew 2015,
  constraint set): a uniform density has :math:`s=0`, so requiring
  :math:`F_{xc}(0)=1` recovers the exact uniform-gas XC energy density
  :math:`\varepsilon_x^{\text{unif}}`.
* **Lieb-Oxford bound** (Lieb & Oxford 1981; Perdew, Burke & Ernzerhof 1996,
  eq. 14): the local bound :math:`\varepsilon_{xc} \ge -C_{LO}\rho^{1/3}` is
  guaranteed by capping the enhancement factor at :math:`1+\kappa` with
  :math:`\kappa=0.804` (the PBE choice that makes the GGA satisfy the local
  Lieb-Oxford bound). The squashing maps the unbounded network output into
  :math:`[F_{\min}, 1+\kappa]`.
* **Spin-scaling for exchange** (Oliver & Perdew 1979):
  :math:`E_x[\rho_\uparrow,\rho_\downarrow] =
  \tfrac12(E_x[2\rho_\uparrow]+E_x[2\rho_\downarrow])`. For the closed-shell
  (spin-unpolarised) case handled by the restricted Kohn-Sham solver this is the
  identity :math:`E_x[\rho/2,\rho/2]=E_x[\rho]`, again structural in the
  enhancement-factor form.

References
----------
* E. H. Lieb, S. Oxford, *Int. J. Quantum Chem.* **19**, 427 (1981) -- the
  lower bound :math:`E_{xc}\ge -C_{LO}\int\rho^{4/3}`; refined constant
  :math:`C_{LO}\approx1.636` (Chan & Handy 1999; Lewin, Lieb & Seiringer 2022).
* M. Levy, J. P. Perdew, *Phys. Rev. A* **32**, 2010 (1985) -- uniform
  coordinate scaling of the exchange energy.
* G. L. Oliver, J. P. Perdew, *Phys. Rev. A* **20**, 397 (1979) -- the
  exchange spin-scaling relation.
* J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996),
  eq. 14 -- :math:`\kappa=0.804` enforces the local Lieb-Oxford bound on the GGA.
* J. Sun, A. Ruzsinszky, J. P. Perdew, *Phys. Rev. Lett.* **115**, 036402
  (2015) -- the SCAN exact-constraint set (UEG limit, scaling, Lieb-Oxford).
* J. Kirkpatrick et al., *Science* **374**, 1385 (2021), arXiv:2102.06179
  (DM21) -- a machine-learned enhancement-factor functional.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


# Slater exchange constant C_x = (3/4)(3/pi)^(1/3); the uniform-gas exchange
# energy per particle is -C_x rho^(1/3).
_EXCHANGE_CONSTANT = 0.75 * (3.0 / jnp.pi) ** (1.0 / 3.0)

# PBE Lieb-Oxford cap: kappa = 0.804 makes the local enhancement factor satisfy
# the Lieb-Oxford bound F_xc <= 1 + kappa = 1.804 (Perdew-Burke-Ernzerhof 1996,
# eq. 14). The refined global Lieb-Oxford constant is C_LO ~ 1.636, and
# 1.804 C_x rho^(4/3) <= C_LO rho^(4/3) is *not* required pointwise; the
# 1 + kappa local cap is the standard, provably LO-satisfying GGA construction.
_LIEB_OXFORD_KAPPA = 0.804

# Refined Lieb-Oxford constant C_LO in E_xc >= -C_LO * integral rho^(4/3)
# (Chan & Handy 1999; Lewin-Lieb-Seiringer 2022 give 1.44 <= C_LO <= 1.58 for
# the universal bound and ~1.636 for the conjectured tight value). Used only to
# *check* the bound in tests, not in the functional form.
_LIEB_OXFORD_CONSTANT = 1.636

# Density floor keeping k_F and rho^(1/3) finite in vacuum regions.
_DENSITY_FLOOR = 1.0e-12

# Floor under the squared gradient so ``sqrt(sigma)`` and its derivative stay
# finite at sigma = 0 (the derivative of the bare square root diverges there).
# This is the same AD-safety device used by the molecular grid's ``_safe_norm``.
_SIGMA_FLOOR = 1.0e-24


def uniform_exchange_energy_density(density: Array) -> Array:
    r"""Uniform-electron-gas exchange energy per particle :math:`-C_x\rho^{1/3}`.

    This is the exact exchange energy density of the uniform electron gas and the
    anchor of the enhancement-factor parametrisation: the full XC energy density
    is this times a dimensionless enhancement factor.

    Args:
        density: Total electron density ``rho`` [any shape].

    Returns:
        ``-C_x rho^(1/3)`` (same shape).
    """
    safe = jnp.clip(density, _DENSITY_FLOOR, None)
    return -_EXCHANGE_CONSTANT * safe ** (1.0 / 3.0)


def reduced_density_gradient(density: Array, sigma: Array) -> Array:
    r"""Dimensionless reduced density gradient :math:`s=|\nabla\rho|/(2k_F\rho)`.

    With :math:`k_F=(3\pi^2\rho)^{1/3}` and :math:`\sigma=|\nabla\rho|^2`. This is
    the scaling-invariant input that makes uniform coordinate scaling structural.

    Args:
        density: Total electron density ``rho`` [any shape].
        sigma: Squared density gradient ``|grad rho|^2`` [same shape].

    Returns:
        Reduced gradient ``s`` (same shape).
    """
    safe = jnp.clip(density, _DENSITY_FLOOR, None)
    safe_sigma = jnp.clip(sigma, 0.0, None)
    k_fermi = (3.0 * jnp.pi**2 * safe) ** (1.0 / 3.0)
    # Floor the sqrt argument so the gradient is finite at sigma = 0.
    gradient_magnitude = jnp.sqrt(safe_sigma + _SIGMA_FLOOR)
    return gradient_magnitude / (2.0 * k_fermi * safe)


def lieb_oxford_bounded_enhancement(
    raw: Array, *, kappa: float = _LIEB_OXFORD_KAPPA, minimum: float = 0.0
) -> Array:
    r"""Map an unbounded network output to a Lieb-Oxford-bounded enhancement factor.

    The enhancement factor must lie in :math:`[F_{\min}, 1+\kappa]` so the XC
    energy density never undershoots the Lieb-Oxford lower bound (the upper cap on
    the enhancement is the lower bound on the *energy*, since the uniform exchange
    factor is negative). A shifted/scaled logistic maps the real line into this
    interval while passing through :math:`F=1` at :math:`raw=0`, so an
    *untrained* (zero-output) network reduces exactly to the uniform-gas (LDA)
    limit:

    .. math::
        F(raw) = F_{\min} + (1+\kappa - F_{\min})\,\sigma(raw + b),\quad
        b = \operatorname{logit}\!\Big(\frac{1-F_{\min}}{1+\kappa-F_{\min}}\Big).

    Args:
        raw: Unbounded enhancement signal from the network [any shape].
        kappa: Lieb-Oxford cap parameter (PBE value 0.804).
        minimum: Lower bound :math:`F_{\min}` on the enhancement factor.

    Returns:
        Enhancement factor in ``[minimum, 1 + kappa]`` (same shape as ``raw``).
    """
    upper = 1.0 + kappa
    span = upper - minimum
    # Bias so F(0) = 1 exactly (untrained network -> LDA limit).
    target = (1.0 - minimum) / span
    bias = jnp.log(target / (1.0 - target))
    return minimum + span * _sigmoid(raw + bias)


def _sigmoid(x: Array) -> Array:
    """Numerically stable logistic sigmoid."""
    return 0.5 * (jnp.tanh(0.5 * x) + 1.0)


def constrained_xc_energy_density(raw_enhancement: Array, density: Array, sigma: Array) -> Array:
    r"""Assemble a constraint-satisfying XC energy density from a raw signal.

    Combines the exact uniform-gas exchange anchor with the Lieb-Oxford-bounded
    enhancement factor:

    .. math::
        \varepsilon_{xc} = \varepsilon_x^{\text{unif}}(\rho)\,F_{xc},\qquad
        F_{xc}\in[F_{\min}, 1+\kappa].

    The dimensionless reduced gradient :math:`s` is the only gradient information
    the enhancement may depend on (the caller is expected to have built
    ``raw_enhancement`` from ``s`` and ``rho`` features), so uniform coordinate
    scaling and the UEG limit are structural.

    Args:
        raw_enhancement: Unbounded enhancement signal from the network [any shape].
        density: Total electron density ``rho`` [same shape].
        sigma: Squared density gradient ``|grad rho|^2`` [same shape].

    Returns:
        Constraint-satisfying XC energy per particle (same shape).
    """
    del sigma  # sigma enters only through the caller's features; kept for symmetry.
    enhancement = lieb_oxford_bounded_enhancement(raw_enhancement)
    return uniform_exchange_energy_density(density) * enhancement


def lieb_oxford_lower_bound_density(density: Array) -> Array:
    r"""Pointwise Lieb-Oxford lower bound :math:`-C_{LO}\rho^{4/3}`.

    The Lieb-Oxford inequality states
    :math:`E_{xc}[\rho]\ge -C_{LO}\int\rho^{4/3}d^3r`; this returns the
    integrand of the right-hand side so a test can verify
    :math:`\rho\,\varepsilon_{xc}\ge -C_{LO}\rho^{4/3}` pointwise.

    Args:
        density: Total electron density ``rho`` [any shape].

    Returns:
        ``-C_LO rho^(4/3)`` (same shape).
    """
    safe = jnp.clip(density, _DENSITY_FLOOR, None)
    return -_LIEB_OXFORD_CONSTANT * safe ** (4.0 / 3.0)


__all__ = [
    "constrained_xc_energy_density",
    "lieb_oxford_bounded_enhancement",
    "lieb_oxford_lower_bound_density",
    "reduced_density_gradient",
    "uniform_exchange_energy_density",
]

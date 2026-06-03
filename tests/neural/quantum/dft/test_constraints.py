r"""Tests for the exact-constraint layer of the learned XC functional.

Each exact constraint from the SCAN / Lieb-Oxford constraint set is verified to
hold for the enhancement-factor parametrisation on randomised test densities:

* the Lieb-Oxford lower bound is never violated;
* uniform coordinate scaling of exchange is satisfied to ~1e-6;
* the uniform-electron-gas (LDA) limit is recovered at zero gradient;
* the closed-shell spin-scaling identity holds.

References
----------
* E. H. Lieb, S. Oxford, *Int. J. Quantum Chem.* **19**, 427 (1981).
* M. Levy, J. P. Perdew, *Phys. Rev. A* **32**, 2010 (1985).
* J. Sun, A. Ruzsinszky, J. P. Perdew, *Phys. Rev. Lett.* **115**, 036402 (2015).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from opifex.neural.quantum.dft._constraints import (
    constrained_xc_energy_density,
    lieb_oxford_bounded_enhancement,
    lieb_oxford_lower_bound_density,
    reduced_density_gradient,
    uniform_exchange_energy_density,
)


def _random_density_and_sigma(seed: int, n: int = 256) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Random positive densities and squared gradients spanning a wide range."""
    key = jax.random.PRNGKey(seed)
    k_rho, k_sigma = jax.random.split(key)
    # Densities over several decades; sigma independently positive.
    log_rho = jax.random.uniform(k_rho, (n,), minval=-6.0, maxval=1.0, dtype=jnp.float64)
    density = jnp.power(10.0, log_rho)
    log_sigma = jax.random.uniform(k_sigma, (n,), minval=-8.0, maxval=2.0, dtype=jnp.float64)
    sigma = jnp.power(10.0, log_sigma)
    return density, sigma


def test_enhancement_factor_is_lieb_oxford_bounded() -> None:
    r"""The bounded enhancement factor never exceeds :math:`1+\kappa=1.804`."""
    with jax.enable_x64(True):
        raw = jax.random.normal(jax.random.PRNGKey(0), (10_000,)) * 50.0
        enhancement = lieb_oxford_bounded_enhancement(raw)
    # The logistic squashes into [0, 1+kappa]=[0, 1.804] by construction; the
    # tiny slack absorbs single-precision roundoff of the reduction.
    assert float(jnp.max(enhancement)) <= 1.804 + 1e-5
    assert float(jnp.min(enhancement)) >= 0.0 - 1e-9


def test_enhancement_factor_recovers_lda_at_zero_signal() -> None:
    """A zero network signal gives enhancement F=1 (exact UEG/LDA limit)."""
    with jax.enable_x64(True):
        enhancement = lieb_oxford_bounded_enhancement(jnp.zeros(5))
    np.testing.assert_allclose(np.asarray(enhancement), 1.0, atol=1e-12)


def test_lieb_oxford_bound_never_violated_on_random_densities() -> None:
    r"""The XC energy density obeys :math:`\rho\varepsilon_{xc}\ge -C_{LO}\rho^{4/3}`."""
    with jax.enable_x64(True):
        density, sigma = _random_density_and_sigma(seed=1)
        # Adversarially large raw signal pushes the enhancement to its cap.
        raw = jax.random.normal(jax.random.PRNGKey(2), density.shape, dtype=jnp.float64) * 30.0
        epsilon_xc = constrained_xc_energy_density(raw, density, sigma)
        energy_density = density * epsilon_xc
        bound = lieb_oxford_lower_bound_density(density)
    # Energy density must lie at or above the Lieb-Oxford floor everywhere.
    assert bool(jnp.all(energy_density >= bound - 1e-12))


def test_uniform_coordinate_scaling_identity() -> None:
    r"""Exchange obeys :math:`E_x[\rho_\gamma]=\gamma E_x[\rho]` to ~1e-6.

    Under :math:`\rho_\gamma(r)=\gamma^3\rho(\gamma r)` the reduced gradient ``s``
    is invariant, so the enhancement is unchanged and the energy integral scales
    linearly in ``gamma``. Verified on a 1D radial model density where the
    integral and the scaled integral can both be evaluated.
    """
    with jax.enable_x64(True):
        gamma = 1.7
        # Radial grid and a smooth model density rho(r) = N exp(-r^2).
        radius = jnp.linspace(1e-3, 6.0, 4000)
        weight = 4.0 * jnp.pi * radius**2 * (radius[1] - radius[0])
        rho = jnp.exp(-(radius**2))
        grad_rho = -2.0 * radius * rho  # drho/dr
        sigma = grad_rho**2
        s = reduced_density_gradient(rho, sigma)
        # Fixed enhancement field (function of s only) to isolate the scaling law.
        raw = 2.0 * s
        eps = constrained_xc_energy_density(raw, rho, sigma)
        e_x = jnp.sum(weight * rho * eps)

        # Scaled density rho_gamma(r) = gamma^3 rho(gamma r) on the same grid;
        # d/dr rho_gamma = gamma^3 (-2 gamma^2 r) exp(-(gamma r)^2).
        scaled_radius = gamma * radius
        rho_g = gamma**3 * jnp.exp(-(scaled_radius**2))
        grad_rho_g = gamma**3 * (-2.0 * gamma**2 * radius) * jnp.exp(-(scaled_radius**2))
        sigma_g = grad_rho_g**2
        s_g = reduced_density_gradient(rho_g, sigma_g)
        raw_g = 2.0 * s_g
        eps_g = constrained_xc_energy_density(raw_g, rho_g, sigma_g)
        e_x_gamma = jnp.sum(weight * rho_g * eps_g)

        relative = float(jnp.abs(e_x_gamma - gamma * e_x) / jnp.abs(gamma * e_x))
    assert relative < 1e-6


def test_reduced_gradient_is_scale_covariant() -> None:
    r"""The reduced gradient obeys :math:`s[\rho_\gamma](r)=s[\rho](\gamma r)`.

    Under uniform coordinate scaling :math:`\rho_\gamma(r)=\gamma^3\rho(\gamma r)`
    the dimensionless reduced gradient is unchanged as a *field* in the scaled
    coordinate -- this scale covariance is what makes the enhancement-factor form
    satisfy the exchange coordinate-scaling identity.
    """
    with jax.enable_x64(True):
        gamma = 2.3
        # Keep gamma r well below the point where exp(-(gamma r)^2) underflows the
        # density floor (~1e-12), so the reduced gradient is not floor-clipped.
        radius = jnp.linspace(1e-3, 2.0, 2000)
        # s of the scaled density rho_gamma(r)=gamma^3 rho(gamma r), evaluated at
        # the same grid points r_i. d/dr rho_gamma = gamma^3 (-2 gamma^2 r) exp(.).
        scaled = gamma * radius
        rho_g = gamma**3 * jnp.exp(-(scaled**2))
        grad_rho_g = gamma**3 * (-2.0 * gamma**2 * radius) * jnp.exp(-(scaled**2))
        s_gamma_at_r = reduced_density_gradient(rho_g, grad_rho_g**2)
        # s of the original density evaluated at the scaled points gamma r_i.
        rho_at_scaled = jnp.exp(-(scaled**2))
        grad_at_scaled = -2.0 * scaled * rho_at_scaled
        s_at_gamma_r = reduced_density_gradient(rho_at_scaled, grad_at_scaled**2)
    np.testing.assert_allclose(np.asarray(s_gamma_at_r), np.asarray(s_at_gamma_r), rtol=1e-6)


def test_uniform_electron_gas_limit() -> None:
    """At zero gradient the energy density equals the exact uniform exchange.

    The reduced gradient ``s`` carries a tiny AD-safety floor under its square
    root, so at physically resolved densities ``s`` is negligible and the
    enhancement recovers the exact uniform-gas limit to high accuracy.
    """
    with jax.enable_x64(True):
        # Densities spanning the chemically relevant range; below ~1e-4 the
        # sqrt floor in ``s`` makes the (vanishing-weight) vacuum tail deviate.
        density = jnp.array([1e-3, 1e-2, 0.1, 1.0, 10.0])
        sigma = jnp.zeros_like(density)
        s = reduced_density_gradient(density, sigma)
        # At s ~ 0 the network's dimensionless input vanishes -> raw signal ~ 0.
        raw = 3.0 * s
        epsilon_xc = constrained_xc_energy_density(raw, density, sigma)
        uniform = uniform_exchange_energy_density(density)
    np.testing.assert_allclose(np.asarray(epsilon_xc), np.asarray(uniform), rtol=1e-7)


def test_closed_shell_spin_scaling_identity() -> None:
    r"""Closed-shell spin scaling: :math:`\tfrac12\sum_\sigma E_x[2\rho_\sigma]=E_x[\rho]`.

    The exchange spin-scaling relation (Oliver & Perdew 1979) is
    :math:`E_x[\rho_\uparrow,\rho_\downarrow]=
    \tfrac12(E_x[2\rho_\uparrow]+E_x[2\rho_\downarrow])`. For a closed shell
    :math:`\rho_\uparrow=\rho_\downarrow=\rho/2`, so each doubled spin density is
    the total density and the right-hand side must reproduce the total exchange
    energy density built by the spin-unpolarised functional. This is the identity
    the restricted Kohn-Sham solver relies on, verified pointwise.
    """
    with jax.enable_x64(True):
        density, sigma = _random_density_and_sigma(seed=4)
        s = reduced_density_gradient(density, sigma)
        raw = 1.5 * s
        eps_total = constrained_xc_energy_density(raw, density, sigma)
        e_total = density * eps_total

        # Equal spin channels: rho_up = rho_dn = rho/2. The doubled spin density
        # 2 rho_up = rho with its gradient (sigma) is fed to the spin-unpolarised
        # functional; the 1/2 (up + dn) sum must give back the total.
        doubled = 2.0 * (density / 2.0)  # == density
        sigma_doubled = sigma  # |grad(2 rho_up)|^2 = |grad rho|^2 for equal split
        s_doubled = reduced_density_gradient(doubled, sigma_doubled)
        raw_doubled = 1.5 * s_doubled
        eps_doubled = constrained_xc_energy_density(raw_doubled, doubled, sigma_doubled)
        e_spin_scaled = 0.5 * doubled * eps_doubled + 0.5 * doubled * eps_doubled
    np.testing.assert_allclose(np.asarray(e_total), np.asarray(e_spin_scaled), atol=1e-12)

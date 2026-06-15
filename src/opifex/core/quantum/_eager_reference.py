r"""Eager per-primitive MMD reference assembly (test cross-check oracle).

The production integral engine in :mod:`opifex.core.quantum.backend` assembles
every AO integral with a single batched ``vmap`` + ``segment_sum`` pass (the
MESS pattern). This module keeps a small, deliberately straightforward *eager*
assembly built directly from the validated per-primitive McMurchie-Davidson
kernels (the ``_primitive_*`` functions in ``backend``): it loops over AO
pairs/quartets explicitly and only ``vmap``-s the inner primitive contraction.
It exists so the batched engine can be cross-checked against the same underlying
math without depending on PySCF, and is imported by the test suite rather than
by any runtime path. The AO-by-AO control flow is an entirely separate code path
from the batched harness, so agreement is a genuine independent check.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from opifex.core.quantum.backend import hermite_coulomb, hermite_expansion
from opifex.core.quantum.basis import AtomicOrbitalBasis  # noqa: TC001
from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001


def _axis_overlap(la: int, lb: int, rab_axis: Array, exp_a: Array, exp_b: Array) -> Array:
    r"""One-axis overlap factor :math:`E_0^{l_a l_b}\sqrt{\pi/p}`."""
    total_p = exp_a + exp_b
    e0 = hermite_expansion(la, lb, rab_axis, exp_a, exp_b)[0]
    return e0 * jnp.sqrt(jnp.pi / total_p)


def _primitive_overlap_kinetic(
    ang_a: tuple[int, int, int],
    ang_b: tuple[int, int, int],
    rab: Array,
    exp_a: Array,
    exp_b: Array,
) -> tuple[Array, Array]:
    r"""Primitive overlap and kinetic integrals (3D, normalised primitives).

    The overlap is the product of the three one-axis overlaps. The kinetic
    integral uses the standard recurrence (Helgaker eq. 9.3.40)

    .. math::
        T_{ij} = -\tfrac12 \big[ j(j-1) S_{i,j-2}
                 - 2b(2j+1) S_{ij} + 4b^2 S_{i,j+2} \big]

    applied per axis and summed, with the other two axes contributing their
    plain overlap factors.
    """
    ax, ay, az = ang_a
    bx, by, bz = ang_b
    sx = _axis_overlap(ax, bx, rab[0], exp_a, exp_b)
    sy = _axis_overlap(ay, by, rab[1], exp_a, exp_b)
    sz = _axis_overlap(az, bz, rab[2], exp_a, exp_b)
    overlap = sx * sy * sz

    def axis_kinetic(la: int, lb: int, rab_axis: Array) -> Array:
        term = -2.0 * exp_b * (2 * lb + 1) * _axis_overlap(la, lb, rab_axis, exp_a, exp_b)
        term = term + 4.0 * exp_b**2 * _axis_overlap(la, lb + 2, rab_axis, exp_a, exp_b)
        if lb >= 2:
            term = term + lb * (lb - 1) * _axis_overlap(la, lb - 2, rab_axis, exp_a, exp_b)
        return -0.5 * term

    tx = axis_kinetic(ax, bx, rab[0]) * sy * sz
    ty = sx * axis_kinetic(ay, by, rab[1]) * sz
    tz = sx * sy * axis_kinetic(az, bz, rab[2])
    return overlap, tx + ty + tz


def _primitive_nuclear(
    ang_a: tuple[int, int, int],
    ang_b: tuple[int, int, int],
    rab: Array,
    exp_a: Array,
    exp_b: Array,
    gaussian_center: Array,
    nuclear_positions: Array,
    nuclear_charges: Array,
    max_total: int,
) -> Array:
    r"""Primitive nuclear-attraction integral summed over nuclei.

    .. math::
        V = -\frac{2\pi}{p} \sum_C Z_C
            \sum_{tuv} E_t^{ab,x} E_u^{ab,y} E_v^{ab,z} R_{tuv}(p, P-C)
    """
    total_p = exp_a + exp_b
    ex = hermite_expansion(ang_a[0], ang_b[0], rab[0], exp_a, exp_b)
    ey = hermite_expansion(ang_a[1], ang_b[1], rab[1], exp_a, exp_b)
    ez = hermite_expansion(ang_a[2], ang_b[2], rab[2], exp_a, exp_b)

    def per_nucleus(position: Array, charge: Array) -> Array:
        r_table = hermite_coulomb(max_total, total_p, gaussian_center - position)
        # Contract E_t E_u E_v R_tuv.
        t_len = ang_a[0] + ang_b[0] + 1
        u_len = ang_a[1] + ang_b[1] + 1
        v_len = ang_a[2] + ang_b[2] + 1
        sub = r_table[:t_len, :u_len, :v_len]
        contracted = jnp.einsum("t,u,v,tuv->", ex, ey, ez, sub)
        return charge * contracted

    contributions = jax.vmap(per_nucleus)(nuclear_positions, nuclear_charges)
    return -2.0 * jnp.pi / total_p * jnp.sum(contributions)


def _primitive_eri(
    ang_a: tuple[int, int, int],
    ang_b: tuple[int, int, int],
    ang_c: tuple[int, int, int],
    ang_d: tuple[int, int, int],
    rab: Array,
    rcd: Array,
    exp_a: Array,
    exp_b: Array,
    exp_c: Array,
    exp_d: Array,
    center_p: Array,
    center_q: Array,
    max_total: int,
) -> Array:
    r"""Primitive two-electron repulsion integral ``(ab|cd)`` (MMD).

    Helgaker eq. 9.9.33:

    .. math::
        (ab|cd) = \frac{2\pi^{5/2}}{pq\sqrt{p+q}}
            \sum_{tuv} E_{tuv}^{ab}
            \sum_{\tau\nu\phi} (-1)^{\tau+\nu+\phi}
            E_{\tau\nu\phi}^{cd} R_{t+\tau,u+\nu,v+\phi}(\alpha, P-Q)
    """
    p = exp_a + exp_b
    q = exp_c + exp_d
    alpha = p * q / (p + q)

    e_ab_x = hermite_expansion(ang_a[0], ang_b[0], rab[0], exp_a, exp_b)
    e_ab_y = hermite_expansion(ang_a[1], ang_b[1], rab[1], exp_a, exp_b)
    e_ab_z = hermite_expansion(ang_a[2], ang_b[2], rab[2], exp_a, exp_b)
    e_cd_x = hermite_expansion(ang_c[0], ang_d[0], rcd[0], exp_c, exp_d)
    e_cd_y = hermite_expansion(ang_c[1], ang_d[1], rcd[1], exp_c, exp_d)
    e_cd_z = hermite_expansion(ang_c[2], ang_d[2], rcd[2], exp_c, exp_d)

    r_table = hermite_coulomb(max_total, alpha, center_p - center_q)

    t_len = ang_a[0] + ang_b[0] + 1
    u_len = ang_a[1] + ang_b[1] + 1
    v_len = ang_a[2] + ang_b[2] + 1
    tau_len = ang_c[0] + ang_d[0] + 1
    nu_len = ang_c[1] + ang_d[1] + 1
    phi_len = ang_c[2] + ang_d[2] + 1

    sign_tau = (-1.0) ** jnp.arange(tau_len, dtype=p.dtype)
    sign_nu = (-1.0) ** jnp.arange(nu_len, dtype=p.dtype)
    sign_phi = (-1.0) ** jnp.arange(phi_len, dtype=p.dtype)
    e_cd_x_signed = e_cd_x[:tau_len] * sign_tau
    e_cd_y_signed = e_cd_y[:nu_len] * sign_nu
    e_cd_z_signed = e_cd_z[:phi_len] * sign_phi

    total = 0.0
    for t in range(t_len):
        for u in range(u_len):
            for v in range(v_len):
                ket = jnp.einsum(
                    "a,b,c,abc->",
                    e_cd_x_signed,
                    e_cd_y_signed,
                    e_cd_z_signed,
                    r_table[t : t + tau_len, u : u + nu_len, v : v + phi_len],
                )
                total = total + e_ab_x[t] * e_ab_y[u] * e_ab_z[v] * ket

    prefactor = 2.0 * jnp.pi**2.5 / (p * q * jnp.sqrt(p + q))
    return prefactor * total


def _ao_table(basis: AtomicOrbitalBasis) -> list[tuple]:
    """Flatten the basis to ``(center, exponents, coeffs, lmn)`` per Cartesian AO."""
    table: list[tuple] = []
    for shell in basis.shells:
        for power in shell.cartesian_components:
            table.append((shell.center, shell.exponents, shell.coefficients, power))
    return table


def eager_one_electron(
    system: MolecularSystem, basis: AtomicOrbitalBasis
) -> tuple[Array, Array, Array]:
    """Assemble ``(S, T, V)`` AO-by-AO; the primitive sum is ``vmap``-vectorised."""
    aos = _ao_table(basis)
    n_ao = len(aos)
    positions = jnp.asarray(system.positions)
    charges = jnp.asarray(system.atomic_numbers).astype(positions.dtype)
    overlap = np.zeros((n_ao, n_ao))
    kinetic = np.zeros((n_ao, n_ao))
    nuclear = np.zeros((n_ao, n_ao))
    for i, (ca, ea, cfa, la) in enumerate(aos):
        for j, (cb, eb, cfb, lb) in enumerate(aos):
            rab = ca - cb
            max_total = sum(la) + sum(lb)
            grid_a = jnp.repeat(ea, eb.shape[0])
            grid_b = jnp.tile(eb, ea.shape[0])
            weight = jnp.repeat(cfa, eb.shape[0]) * jnp.tile(cfb, ea.shape[0])

            def stv(exp_a, exp_b, _la=la, _lb=lb, _rab=rab, _mt=max_total, _ca=ca, _cb=cb):
                s_p, t_p = _primitive_overlap_kinetic(_la, _lb, _rab, exp_a, exp_b)
                gc = (exp_a * _ca + exp_b * _cb) / (exp_a + exp_b)
                v_p = _primitive_nuclear(_la, _lb, _rab, exp_a, exp_b, gc, positions, charges, _mt)
                return jnp.stack([s_p, t_p, v_p])

            vals = jax.vmap(stv)(grid_a, grid_b)
            contracted = jnp.sum(weight[:, None] * vals, axis=0)
            overlap[i, j] = float(contracted[0])
            kinetic[i, j] = float(contracted[1])
            nuclear[i, j] = float(contracted[2])
    return jnp.asarray(overlap), jnp.asarray(kinetic), jnp.asarray(nuclear)


def eager_eri(system: MolecularSystem, basis: AtomicOrbitalBasis) -> Array:
    """Assemble ``(ij|kl)`` AO-quartet-by-quartet; primitive sum ``vmap``-vectorised."""
    del system
    aos = _ao_table(basis)
    n_ao = len(aos)
    eri = np.zeros((n_ao, n_ao, n_ao, n_ao))
    for i, (ca, ea, cfa, la) in enumerate(aos):
        for j, (cb, eb, cfb, lb) in enumerate(aos):
            rab = ca - cb
            for k, (cc, ec, cfc, lc) in enumerate(aos):
                for ll, (cd, ed, cfd, ld) in enumerate(aos):
                    rcd = cc - cd
                    max_total = sum(la) + sum(lb) + sum(lc) + sum(ld)
                    n_b, n_c, n_d = eb.shape[0], ec.shape[0], ed.shape[0]
                    inner = n_b * n_c * n_d
                    ga = jnp.repeat(ea, inner)
                    gb = jnp.tile(jnp.repeat(eb, n_c * n_d), ea.shape[0])
                    gc_ = jnp.tile(jnp.repeat(ec, n_d), ea.shape[0] * n_b)
                    gd = jnp.tile(ed, ea.shape[0] * n_b * n_c)
                    weight = (
                        jnp.repeat(cfa, inner)
                        * jnp.tile(jnp.repeat(cfb, n_c * n_d), ea.shape[0])
                        * jnp.tile(jnp.repeat(cfc, n_d), ea.shape[0] * n_b)
                        * jnp.tile(cfd, ea.shape[0] * n_b * n_c)
                    )

                    def prim(
                        exp_a,
                        exp_b,
                        exp_c,
                        exp_d,
                        _la=la,
                        _lb=lb,
                        _lc=lc,
                        _ld=ld,
                        _rab=rab,
                        _rcd=rcd,
                        _mt=max_total,
                        _ca=ca,
                        _cb=cb,
                        _cc=cc,
                        _cd=cd,
                    ):
                        cp = (exp_a * _ca + exp_b * _cb) / (exp_a + exp_b)
                        cq = (exp_c * _cc + exp_d * _cd) / (exp_c + exp_d)
                        return _primitive_eri(
                            _la,
                            _lb,
                            _lc,
                            _ld,
                            _rab,
                            _rcd,
                            exp_a,
                            exp_b,
                            exp_c,
                            exp_d,
                            cp,
                            cq,
                            _mt,
                        )

                    vals = jax.vmap(prim)(ga, gb, gc_, gd)
                    eri[i, j, k, ll] = float(jnp.sum(weight * vals))
    return jnp.asarray(eri)


__all__ = ["eager_eri", "eager_one_electron"]

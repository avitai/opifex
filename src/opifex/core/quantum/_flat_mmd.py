r"""Fixed-size, ``vmap``-friendly McMurchie-Davidson primitive kernels.

The validated McMurchie-Davidson (MMD) Hermite-expansion math lives in
:mod:`opifex.core.quantum.backend` (``hermite_expansion``, ``hermite_coulomb``,
the ``_primitive_*`` kernels). Those functions take the angular momenta
``l_a``/``l_b`` as Python ints (loop bounds) and therefore cannot be ``vmap``-ed
across primitives carrying *different* angular powers in a single trace.

This module re-expresses the same recurrences with the loop bounds fixed to a
static ``max_l`` (the basis maximum), so a single ``jax.vmap`` over the flat
primitive pytree (:class:`opifex.core.quantum.basis.FlatPrimitives`) evaluates
every primitive pair/quartet regardless of its angular powers. The actual
``(l_x, l_y, l_z)`` powers arrive as *dynamic* integer arrays and are used only
to *index/select* into the fixed-size Hermite tables -- the recurrence itself is
unrolled at trace time over ``max_l`` and is identical to the eager kernels, so
the results are bit-compatible (validated to ~1e-10 against the eager backend).

References mirror the eager backend: Helgaker, Jorgensen & Olsen, *Molecular
Electronic-Structure Theory* (2000), Ch. 9; McMurchie & Davidson, *J. Comput.
Phys.* **26**, 218 (1978). The batching layout follows
``graphcore-research/mess`` (Helal et al., arXiv:2406.03121).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from opifex.core.quantum._boys import boys_vector


def hermite_table(
    max_l: int,
    distance: Array,
    exp_a: Array,
    exp_b: Array,
) -> Array:
    r"""Full Hermite expansion table :math:`E_t^{ij}` for one axis to ``max_l``.

    Builds the McMurchie-Davidson table ``E[i, j, t]`` for all
    ``0 <= i, j <= max_l`` and ``0 <= t <= i + j`` in one shot (the recurrence
    body uses only static loop bounds, so this is fully ``vmap``-able over the
    exponents/distance). The per-axis powers are selected afterwards by indexing
    ``E[l_a, l_b]``.

    Args:
        max_l: Maximum angular-momentum power per axis (static).
        distance: Signed separation ``X_AB`` along this axis.
        exp_a: Primitive exponent on centre A.
        exp_b: Primitive exponent on centre B.

    Returns:
        Array ``E[i, j, t]`` of shape ``(max_l+1, max_l+1, 2*max_l+1)``.
    """
    total_p = exp_a + exp_b
    reduced = exp_a * exp_b / total_p
    pa = -exp_b * distance / total_p
    pb = exp_a * distance / total_p
    t_max = 2 * max_l
    half_inv_p = 1.0 / (2.0 * total_p)
    t_indices = jnp.arange(t_max + 1, dtype=total_p.dtype)

    table = jnp.zeros((max_l + 1, max_l + 1, t_max + 1), dtype=total_p.dtype)
    table = table.at[0, 0, 0].set(jnp.exp(-reduced * distance**2))

    def shifted(column: Array, offset: int) -> Array:
        """Return ``E_{t+offset}`` aligned to index ``t`` (zero-padded)."""
        if offset == 0:
            return column
        if offset == -1:
            return jnp.concatenate([jnp.zeros(1, column.dtype), column[:-1]])
        return jnp.concatenate([column[1:], jnp.zeros(1, column.dtype)])

    # Increment i (centre A) with j = 0.
    for i in range(max_l):
        column = table[i, 0]
        new_column = (
            half_inv_p * shifted(column, -1)
            + pa * shifted(column, 0)
            + (t_indices + 1.0) * shifted(column, +1)
        )
        table = table.at[i + 1, 0].set(new_column)

    # Increment j (centre B) for every i.
    for i in range(max_l + 1):
        for j in range(max_l):
            column = table[i, j]
            new_column = (
                half_inv_p * shifted(column, -1)
                + pb * shifted(column, 0)
                + (t_indices + 1.0) * shifted(column, +1)
            )
            table = table.at[i, j + 1].set(new_column)

    return table


def hermite_coulomb_table(max_total: int, alpha: Array, separation: Array) -> Array:
    r"""Hermite-Coulomb auxiliary integrals :math:`R_{tuv}` to ``max_total``.

    Identical recurrence to :func:`opifex.core.quantum.backend.hermite_coulomb`
    but with the table sized by the *static* ``max_total`` so it is
    ``vmap``-able; entries with ``t + u + v > max_total`` are left at zero.

    Args:
        max_total: Maximum of ``t + u + v`` (static).
        alpha: Combined exponent.
        separation: ``P - C`` vector [Shape: (3,)].

    Returns:
        Array ``R[t, u, v]`` of shape ``(max_total+1,)*3``.
    """
    dx, dy, dz = separation[0], separation[1], separation[2]
    r2 = dx * dx + dy * dy + dz * dz
    boys = boys_vector(max_total, alpha * r2)
    powers = (-2.0 * alpha) ** jnp.arange(max_total + 1, dtype=alpha.dtype)
    aux = powers * boys

    size = max_total + 1
    r_table = jnp.zeros((size, size, size, size), dtype=alpha.dtype)
    r_table = r_table.at[0, 0, 0, :].set(aux)

    def lower(t: int, u: int, v: int) -> Array:
        """``R_{t,u,v}^{n+1}`` aligned to index ``n`` (last entry padded)."""
        return jnp.concatenate([r_table[t, u, v, 1:], jnp.zeros(1, r_table.dtype)])

    for total in range(1, max_total + 1):
        for t in range(total + 1):
            for u in range(total + 1 - t):
                v = total - t - u
                if t > 0:
                    term = (t - 1) * lower(t - 2, u, v) if t >= 2 else 0.0
                    new = term + dx * lower(t - 1, u, v)
                elif u > 0:
                    term = (u - 1) * lower(t, u - 2, v) if u >= 2 else 0.0
                    new = term + dy * lower(t, u - 1, v)
                else:
                    term = (v - 1) * lower(t, u, v - 2) if v >= 2 else 0.0
                    new = term + dz * lower(t, u, v - 1)
                r_table = r_table.at[t, u, v, :].set(new)

    return r_table[:, :, :, 0]


def _select_axis(table: Array, l_a: Array, l_b: Array) -> Array:
    """Select the ``E[l_a, l_b, :]`` column from a fixed-size Hermite table."""
    return table[l_a, l_b]


def overlap_primitive(
    lmn_a: Array,
    lmn_b: Array,
    center_a: Array,
    center_b: Array,
    exp_a: Array,
    exp_b: Array,
    max_l: int,
) -> Array:
    r"""Primitive overlap integral via the fixed-size Hermite tables.

    .. math:: S = \prod_{c\in\{x,y,z\}} E_0^{l_a^c l_b^c}\sqrt{\pi/p}.
    """
    total_p = exp_a + exp_b
    rab = center_a - center_b
    norm = jnp.sqrt(jnp.pi / total_p)
    out = jnp.asarray(1.0, dtype=total_p.dtype)
    for axis in range(3):
        table = hermite_table(max_l, rab[axis], exp_a, exp_b)
        out = out * _select_axis(table, lmn_a[axis], lmn_b[axis])[0] * norm
    return out


def overlap_with_shift(
    lmn_a: Array,
    lmn_b: Array,
    center_a: Array,
    center_b: Array,
    exp_a: Array,
    exp_b: Array,
    axis: int,
    shift: int,
    max_l: int,
) -> Array:
    """Overlap with ``lmn_b[axis]`` shifted by ``shift`` (for the kinetic term).

    The shift can push ``l_b`` to ``max_l + 2``; the Hermite table is sized to
    ``max_l + 2`` so the shifted column is always available, and a ``jnp.where``
    zeroes the (unphysical) negative-power case.
    """
    total_p = exp_a + exp_b
    rab = center_a - center_b
    norm = jnp.sqrt(jnp.pi / total_p)
    out = jnp.asarray(1.0, dtype=total_p.dtype)
    for ax in range(3):
        lb = lmn_b[ax] + (shift if ax == axis else 0)
        lb_safe = jnp.where(lb < 0, 0, lb)
        table = hermite_table(max_l + 2, rab[ax], exp_a, exp_b)
        column = _select_axis(table, lmn_a[ax], lb_safe)[0]
        column = jnp.where(lb < 0, 0.0, column)
        out = out * column * norm
    return out


def kinetic_primitive(
    lmn_a: Array,
    lmn_b: Array,
    center_a: Array,
    center_b: Array,
    exp_a: Array,
    exp_b: Array,
    max_l: int,
) -> Array:
    r"""Primitive kinetic-energy integral (Helgaker eq. 9.3.40), per axis summed.

    .. math::
        T = -\tfrac12 \sum_{c} \big[ l_b^c(l_b^c-1) S_{l_b^c-2}
            - 2b(2 l_b^c + 1) S_{l_b^c} + 4b^2 S_{l_b^c+2} \big].
    """
    total = jnp.asarray(0.0, dtype=(exp_a + exp_b).dtype)
    for axis in range(3):
        lb = lmn_b[axis]
        s0 = overlap_with_shift(lmn_a, lmn_b, center_a, center_b, exp_a, exp_b, axis, 0, max_l)
        s_plus = overlap_with_shift(lmn_a, lmn_b, center_a, center_b, exp_a, exp_b, axis, 2, max_l)
        s_minus = overlap_with_shift(
            lmn_a, lmn_b, center_a, center_b, exp_a, exp_b, axis, -2, max_l
        )
        term = lb * (lb - 1) * s_minus - 2.0 * exp_b * (2 * lb + 1) * s0 + 4.0 * exp_b**2 * s_plus
        total = total + (-0.5) * term
    return total


def nuclear_primitive(
    lmn_a: Array,
    lmn_b: Array,
    center_a: Array,
    center_b: Array,
    exp_a: Array,
    exp_b: Array,
    nuclear_positions: Array,
    nuclear_charges: Array,
    max_l: int,
) -> Array:
    r"""Primitive nuclear-attraction integral summed over nuclei.

    .. math::
        V = -\frac{2\pi}{p}\sum_C Z_C
            \sum_{tuv} E_t^x E_u^y E_v^z R_{tuv}(p, P-C).
    """
    total_p = exp_a + exp_b
    rab = center_a - center_b
    gaussian_center = (exp_a * center_a + exp_b * center_b) / total_p
    max_total = 2 * max_l

    ex = _select_axis(hermite_table(max_l, rab[0], exp_a, exp_b), lmn_a[0], lmn_b[0])
    ey = _select_axis(hermite_table(max_l, rab[1], exp_a, exp_b), lmn_a[1], lmn_b[1])
    ez = _select_axis(hermite_table(max_l, rab[2], exp_a, exp_b), lmn_a[2], lmn_b[2])

    def per_nucleus(position: Array, charge: Array) -> Array:
        r_table = hermite_coulomb_table(max_total, total_p, gaussian_center - position)
        contracted = jnp.einsum("t,u,v,tuv->", ex, ey, ez, r_table)
        return charge * contracted

    contributions = jax.vmap(per_nucleus)(nuclear_positions, nuclear_charges)
    return -2.0 * jnp.pi / total_p * jnp.sum(contributions)


def eri_primitive(
    lmn_a: Array,
    lmn_b: Array,
    lmn_c: Array,
    lmn_d: Array,
    center_a: Array,
    center_b: Array,
    center_c: Array,
    center_d: Array,
    exp_a: Array,
    exp_b: Array,
    exp_c: Array,
    exp_d: Array,
    max_l: int,
) -> Array:
    r"""Primitive two-electron repulsion integral ``(ab|cd)`` (Helgaker 9.9.33).

    .. math::
        (ab|cd) = \frac{2\pi^{5/2}}{pq\sqrt{p+q}}
            \sum_{tuv} E_{tuv}^{ab}
            \sum_{\tau\nu\phi}(-1)^{\tau+\nu+\phi}
            E_{\tau\nu\phi}^{cd} R_{t+\tau,u+\nu,v+\phi}(\alpha, P-Q).
    """
    p = exp_a + exp_b
    q = exp_c + exp_d
    alpha = p * q / (p + q)
    rab = center_a - center_b
    rcd = center_c - center_d
    center_p = (exp_a * center_a + exp_b * center_b) / p
    center_q = (exp_c * center_c + exp_d * center_d) / q

    e_ab_x = _select_axis(hermite_table(max_l, rab[0], exp_a, exp_b), lmn_a[0], lmn_b[0])
    e_ab_y = _select_axis(hermite_table(max_l, rab[1], exp_a, exp_b), lmn_a[1], lmn_b[1])
    e_ab_z = _select_axis(hermite_table(max_l, rab[2], exp_a, exp_b), lmn_a[2], lmn_b[2])
    e_cd_x = _select_axis(hermite_table(max_l, rcd[0], exp_c, exp_d), lmn_c[0], lmn_d[0])
    e_cd_y = _select_axis(hermite_table(max_l, rcd[1], exp_c, exp_d), lmn_c[1], lmn_d[1])
    e_cd_z = _select_axis(hermite_table(max_l, rcd[2], exp_c, exp_d), lmn_c[2], lmn_d[2])

    max_total = 4 * max_l
    r_table = hermite_coulomb_table(max_total, alpha, center_p - center_q)

    half = 2 * max_l + 1
    sign = (-1.0) ** jnp.arange(half, dtype=p.dtype)
    e_cd_x_signed = e_cd_x * sign
    e_cd_y_signed = e_cd_y * sign
    e_cd_z_signed = e_cd_z * sign

    # Contract R[t+tau, u+nu, v+phi] against both Hermite vectors with static
    # shapes (vmap-safe): gather the windowed (t,u,v,tau,nu,phi) tensor, contract
    # the ket (tau,nu,phi) then the bra (t,u,v) via two einsums.
    window = _coulomb_window(r_table, half)
    ket = jnp.einsum("a,b,c,tuvabc->tuv", e_cd_x_signed, e_cd_y_signed, e_cd_z_signed, window)
    total = jnp.einsum("t,u,v,tuv->", e_ab_x, e_ab_y, e_ab_z, ket)

    prefactor = 2.0 * jnp.pi**2.5 / (p * q * jnp.sqrt(p + q))
    return prefactor * total


def _coulomb_window(r_table: Array, half: int) -> Array:
    """Gather ``R[t+tau, u+nu, v+phi]`` into a ``(t,u,v,tau,nu,phi)`` tensor.

    ``r_table`` has shape ``(4*max_l+1,)*3`` and ``half = 2*max_l + 1``. The
    returned tensor lets the ket contraction be a single static-shape einsum.
    """
    base = jnp.arange(half)
    t = base[:, None] + base[None, :]  # (half, half) = t + tau
    # Use take along each axis successively.
    gathered = jnp.take(r_table, t.reshape(-1), axis=0).reshape(
        half, half, r_table.shape[1], r_table.shape[2]
    )
    gathered = jnp.take(gathered, t.reshape(-1), axis=2).reshape(
        half, half, half, half, r_table.shape[2]
    )
    gathered = jnp.take(gathered, t.reshape(-1), axis=4).reshape(half, half, half, half, half, half)
    # Axes order: (t, tau, u, nu, v, phi) -> reorder to (t,u,v,tau,nu,phi).
    return gathered.transpose(0, 2, 4, 1, 3, 5)


__all__ = [
    "eri_primitive",
    "hermite_coulomb_table",
    "hermite_table",
    "kinetic_primitive",
    "nuclear_primitive",
    "overlap_primitive",
]

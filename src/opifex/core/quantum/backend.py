r"""Native JAX Gaussian-integral engine (McMurchie-Davidson).

This module computes the one- and two-electron molecular integrals over the
contracted Cartesian-Gaussian basis of
:class:`~opifex.core.quantum.basis.AtomicOrbitalBasis` using the
McMurchie-Davidson (MMD) Hermite-Gaussian expansion. Everything is written in
JAX (``jnp``) so the integral tensors are differentiable with respect to the
nuclear positions and the whole pipeline is ``jit``-compatible.

The :class:`QCBackend` Protocol is the swappable seam (dependency inversion):
:class:`JaxGaussianBackend` is the native in-tree implementation, and a PySCF
adapter exists only in the tests as a validation oracle.

Method and references
---------------------
The implementation follows the standard McMurchie-Davidson scheme:

* L. E. McMurchie, E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978).
* T. Helgaker, P. Jorgensen, J. Olsen, *Molecular Electronic-Structure Theory*,
  Wiley (2000), Ch. 9 -- the Hermite expansion coefficients :math:`E_t^{ij}`
  (eq. 9.5.6), the overlap (9.5.41), kinetic (9.3.40-9.3.43) and the
  Hermite-Coulomb integrals :math:`R_{tuv}` (9.9.18-9.9.20) used for the
  nuclear-attraction and electron-repulsion integrals.
* The Boys function :math:`F_n(x)=\int_0^1 t^{2n} e^{-x t^2}\,dt` is evaluated in
  closed form via the lower incomplete gamma function,
  :math:`F_n(x)=\gamma(n+\tfrac12, x)/(2\,x^{\,n+1/2})`
  (Helgaker eq. 9.8.39), using :func:`jax.scipy.special.gammainc`.

The reference implementation cross-checked for the recurrence indexing is the
Joshua Goings "Integrals" write-up of McMurchie-Davidson, which itself follows
Helgaker; every integral is validated against PySCF to ~1e-8 in the test suite.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gamma, gammainc

from opifex.core.quantum.basis import AtomicOrbitalBasis, GaussianShell  # noqa: TC001
from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001


@runtime_checkable
class QCBackend(Protocol):
    """Quantum-chemistry integral backend seam.

    Implementations provide the AO integral tensors and the nuclear-repulsion
    energy for a fixed :class:`MolecularSystem` / :class:`AtomicOrbitalBasis`.
    """

    def overlap(self) -> Array:
        """Return the AO overlap matrix ``S`` [Shape: (n_ao, n_ao)]."""
        ...

    def core_hamiltonian(self) -> Array:
        """Return the core Hamiltonian ``T + V`` [Shape: (n_ao, n_ao)]."""
        ...

    def electron_repulsion(self) -> Array:
        """Return the ERI tensor ``(ij|kl)`` [Shape: (n_ao,)*4] in chemist order."""
        ...

    def nuclear_repulsion(self) -> Array:
        """Return the scalar nuclear-repulsion energy ``E_nn``."""
        ...


def boys_function(order: int, argument: Array) -> Array:
    r"""Boys function :math:`F_n(x)` for a single order ``n``.

    Computed from the regularised lower incomplete gamma ``P(a, x)`` as

    .. math::
        F_n(x) = \frac{\gamma(n+\tfrac12, x)}{2\,x^{\,n+1/2}}
               = \frac{P(n+\tfrac12, x)\,\Gamma(n+\tfrac12)}{2\,x^{\,n+1/2}},

    with the removable singularity at ``x = 0`` handled by the analytic limit
    :math:`F_n(0) = 1/(2n+1)`.

    Args:
        order: Boys order ``n`` (non-negative integer).
        argument: Argument ``x`` [any shape], expected non-negative.

    Returns:
        ``F_n(x)`` with the same shape as ``argument``.
    """
    a = order + 0.5
    # Guard the x -> 0 branch so the division and gammainc stay finite/AD-safe.
    safe_x = jnp.where(argument > 0.0, argument, 1.0)
    regular = gammainc(a, safe_x) * gamma(a) / (2.0 * safe_x**a)
    limit = jnp.asarray(1.0 / (2.0 * order + 1.0), dtype=regular.dtype)
    return jnp.where(argument > 0.0, regular, limit)


def boys_vector(max_order: int, argument: Array) -> Array:
    """Stacked Boys functions ``[F_0(x), ..., F_max_order(x)]`` for scalar ``x``.

    Args:
        max_order: Highest Boys order to return (inclusive).
        argument: Scalar argument ``x``.

    Returns:
        Array of shape ``(max_order + 1,)``.
    """
    return jnp.stack([boys_function(n, argument) for n in range(max_order + 1)])


def hermite_expansion(
    l_a: int,
    l_b: int,
    distance: Array,
    exp_a: Array,
    exp_b: Array,
) -> Array:
    r"""Hermite expansion coefficients :math:`E_t^{l_a l_b}` for one Cartesian axis.

    Implements the McMurchie-Davidson recurrence (Helgaker eq. 9.5.6-9.5.8)

    .. math::
        E_t^{i+1,j} = \frac{1}{2p} E_{t-1}^{ij}
                    + X_{PA} E_t^{ij} + (t+1) E_{t+1}^{ij},
        E_t^{i,j+1} = \frac{1}{2p} E_{t-1}^{ij}
                    + X_{PB} E_t^{ij} + (t+1) E_{t+1}^{ij},

    with :math:`E_0^{00} = \exp(-\mu X_{AB}^2)`, :math:`p=a+b`,
    :math:`\mu = ab/p`, :math:`X_{PA} = -b\,X_{AB}/p`,
    :math:`X_{PB} = a\,X_{AB}/p`.

    Args:
        l_a: Angular-momentum power on centre A for this axis.
        l_b: Angular-momentum power on centre B for this axis.
        distance: Signed separation ``X_AB = X_A - X_B`` along this axis.
        exp_a: Primitive exponent ``a`` on centre A.
        exp_b: Primitive exponent ``b`` on centre B.

    Returns:
        Array of length ``l_a + l_b + 1`` holding ``E_0, ..., E_{l_a+l_b}``.
    """
    total_p = exp_a + exp_b
    reduced = exp_a * exp_b / total_p
    pa = -exp_b * distance / total_p
    pb = exp_a * distance / total_p
    t_max = l_a + l_b

    # table[i, j, t] = E_t^{ij}; sized (l_a+1, l_b+1, t_max+1).
    table = jnp.zeros((l_a + 1, l_b + 1, t_max + 1), dtype=total_p.dtype)
    table = table.at[0, 0, 0].set(jnp.exp(-reduced * distance**2))

    half_inv_p = 1.0 / (2.0 * total_p)

    def shifted(tab: Array, i: int, j: int, offset: int) -> Array:
        """Return ``E_{t+offset}^{ij}`` aligned to index ``t``, zero-padded."""
        column = tab[i, j]
        if offset == 0:
            return column
        if offset == -1:
            return jnp.concatenate([jnp.zeros(1, column.dtype), column[:-1]])
        return jnp.concatenate([column[1:], jnp.zeros(1, column.dtype)])

    t_indices = jnp.arange(t_max + 1, dtype=total_p.dtype)

    # Increment i (centre A) with j = 0.
    for i in range(l_a):
        new_column = (
            half_inv_p * shifted(table, i, 0, -1)
            + pa * shifted(table, i, 0, 0)
            + (t_indices + 1.0) * shifted(table, i, 0, +1)
        )
        table = table.at[i + 1, 0].set(new_column)

    # Increment j (centre B) for every i.
    for i in range(l_a + 1):
        for j in range(l_b):
            new_column = (
                half_inv_p * shifted(table, i, j, -1)
                + pb * shifted(table, i, j, 0)
                + (t_indices + 1.0) * shifted(table, i, j, +1)
            )
            table = table.at[i, j + 1].set(new_column)

    return table[l_a, l_b]


def hermite_coulomb(max_total: int, alpha: Array, separation: Array) -> Array:
    r"""Hermite-Coulomb auxiliary integrals :math:`R_{tuv}`.

    Implements Helgaker eq. 9.9.18-9.9.20:

    .. math::
        R_{tuv}^n = \begin{cases}
            (-2\alpha)^n F_n(\alpha R_{PC}^2) & t=u=v=0 \\
            t\,R_{t-1,u,v}^{n+1} + X_{PC} R_{t-1,u,v}^{n+1} & \dots
        \end{cases}

    Args:
        max_total: Maximum of ``t + u + v`` required.
        alpha: Combined exponent ``alpha`` (``p`` for V, ``pq/(p+q)`` for ERI).
        separation: ``P - C`` vector [Shape: (3,)].

    Returns:
        Array ``R[t, u, v]`` of shape ``(max_total+1,)*3`` (entries with
        ``t+u+v > max_total`` are left at zero).
    """
    dx, dy, dz = separation[0], separation[1], separation[2]
    r2 = dx * dx + dy * dy + dz * dz
    boys = boys_vector(max_total, alpha * r2)
    powers = (-2.0 * alpha) ** jnp.arange(max_total + 1, dtype=alpha.dtype)
    # aux[n] = R_000^n = (-2 alpha)^n F_n.
    aux = powers * boys

    size = max_total + 1
    # r_table[t, u, v, n] built up the standard MMD way.
    r_table = jnp.zeros((size, size, size, size), dtype=alpha.dtype)
    r_table = r_table.at[0, 0, 0, :].set(aux)

    def lower(tab: Array, t: int, u: int, v: int) -> Array:
        """``R_{t,u,v}^{n+1}`` aligned to index ``n`` (last entry padded)."""
        return jnp.concatenate([tab[t, u, v, 1:], jnp.zeros(1, tab.dtype)])

    for total in range(1, max_total + 1):
        for t in range(total + 1):
            for u in range(total + 1 - t):
                v = total - t - u
                if t > 0:
                    term = (t - 1) * lower(r_table, t - 2, u, v) if t >= 2 else 0.0
                    new = term + dx * lower(r_table, t - 1, u, v)
                elif u > 0:
                    term = (u - 1) * lower(r_table, t, u - 2, v) if u >= 2 else 0.0
                    new = term + dy * lower(r_table, t, u - 1, v)
                else:
                    term = (v - 1) * lower(r_table, t, u, v - 2) if v >= 2 else 0.0
                    new = term + dz * lower(r_table, t, u, v - 1)
                r_table = r_table.at[t, u, v, :].set(new)

    return r_table[:, :, :, 0]


def _primitive_pair_grid(
    shell_a: GaussianShell, shell_b: GaussianShell
) -> tuple[Array, Array, Array]:
    """Flattened ``(exp_a, exp_b, coeff)`` grids over a shell-pair's primitives.

    Returns:
        Three 1-D arrays of length ``n_prim_a * n_prim_b`` holding the broadcast
        primitive exponents of each centre and their contraction-coefficient
        products.
    """
    exp_a = jnp.repeat(shell_a.exponents, shell_b.n_primitives)
    exp_b = jnp.tile(shell_b.exponents, shell_a.n_primitives)
    coeff = jnp.repeat(shell_a.coefficients, shell_b.n_primitives) * jnp.tile(
        shell_b.coefficients, shell_a.n_primitives
    )
    return exp_a, exp_b, coeff


def _shell_pair_overlap_kinetic(
    shell_a: GaussianShell, shell_b: GaussianShell
) -> tuple[Array, Array]:
    """Overlap and kinetic blocks between two contracted shells.

    The primitive-pair sum is vectorised with :func:`jax.vmap`; the small static
    Cartesian-component loops are unrolled at trace time.

    Returns:
        ``(s_block, t_block)`` each of shape ``(n_cart_a, n_cart_b)``.
    """
    rab = shell_a.center - shell_b.center
    exp_a_grid, exp_b_grid, coeff_grid = _primitive_pair_grid(shell_a, shell_b)

    s_columns = []
    t_columns = []
    for ang_a in shell_a.cartesian_components:
        for ang_b in shell_b.cartesian_components:

            def primitive(exp_a: Array, exp_b: Array, _aa=ang_a, _bb=ang_b) -> Array:
                overlap, kinetic = _primitive_overlap_kinetic(_aa, _bb, rab, exp_a, exp_b)
                return jnp.stack([overlap, kinetic])

            values = jax.vmap(primitive)(exp_a_grid, exp_b_grid)
            contracted = jnp.sum(coeff_grid[:, None] * values, axis=0)
            s_columns.append(contracted[0])
            t_columns.append(contracted[1])

    s_block = jnp.stack(s_columns).reshape(shell_a.n_cartesian, shell_b.n_cartesian)
    t_block = jnp.stack(t_columns).reshape(shell_a.n_cartesian, shell_b.n_cartesian)
    return s_block, t_block


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


def _shell_pair_nuclear(
    shell_a: GaussianShell,
    shell_b: GaussianShell,
    nuclear_positions: Array,
    nuclear_charges: Array,
) -> Array:
    """Nuclear-attraction block between two shells, summed over all nuclei.

    The primitive-pair sum is vectorised with :func:`jax.vmap`.
    """
    rab = shell_a.center - shell_b.center
    max_total = shell_a.angular_momentum + shell_b.angular_momentum
    exp_a_grid, exp_b_grid, coeff_grid = _primitive_pair_grid(shell_a, shell_b)
    center_a = shell_a.center
    center_b = shell_b.center

    columns = []
    for ang_a in shell_a.cartesian_components:
        for ang_b in shell_b.cartesian_components:

            def primitive(exp_a: Array, exp_b: Array, _aa=ang_a, _bb=ang_b) -> Array:
                gaussian_center = (exp_a * center_a + exp_b * center_b) / (exp_a + exp_b)
                return _primitive_nuclear(
                    _aa,
                    _bb,
                    rab,
                    exp_a,
                    exp_b,
                    gaussian_center,
                    nuclear_positions,
                    nuclear_charges,
                    max_total,
                )

            values = jax.vmap(primitive)(exp_a_grid, exp_b_grid)
            columns.append(jnp.sum(coeff_grid * values))

    return jnp.stack(columns).reshape(shell_a.n_cartesian, shell_b.n_cartesian)


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


def _shell_quartet_eri(
    shell_a: GaussianShell,
    shell_b: GaussianShell,
    shell_c: GaussianShell,
    shell_d: GaussianShell,
    rab: Array,
    rcd: Array,
) -> Array:
    """Vectorised ERI block for one shell quartet [Shape: (na, nb, nc, nd)].

    The four-index primitive sum is flattened to a single batch dimension and
    evaluated with :func:`jax.vmap`; the small static Cartesian-component loops
    are unrolled at trace time.
    """
    max_total = (
        shell_a.angular_momentum
        + shell_b.angular_momentum
        + shell_c.angular_momentum
        + shell_d.angular_momentum
    )
    n_a = shell_a.n_primitives
    n_b = shell_b.n_primitives
    n_c = shell_c.n_primitives
    n_d = shell_d.n_primitives
    inner = n_b * n_c * n_d

    # Flatten the four primitive axes into one batch of length n_a*n_b*n_c*n_d.
    exp_a = jnp.repeat(shell_a.exponents, inner)
    exp_b = jnp.tile(jnp.repeat(shell_b.exponents, n_c * n_d), n_a)
    exp_c = jnp.tile(jnp.repeat(shell_c.exponents, n_d), n_a * n_b)
    exp_d = jnp.tile(shell_d.exponents, n_a * n_b * n_c)
    coeff = (
        jnp.repeat(shell_a.coefficients, inner)
        * jnp.tile(jnp.repeat(shell_b.coefficients, n_c * n_d), n_a)
        * jnp.tile(jnp.repeat(shell_c.coefficients, n_d), n_a * n_b)
        * jnp.tile(shell_d.coefficients, n_a * n_b * n_c)
    )
    center_a, center_b = shell_a.center, shell_b.center
    center_c, center_d = shell_c.center, shell_d.center

    columns = []
    for ang_a in shell_a.cartesian_components:
        for ang_b in shell_b.cartesian_components:
            for ang_c in shell_c.cartesian_components:
                for ang_d in shell_d.cartesian_components:

                    def primitive(
                        ea: Array,
                        eb: Array,
                        ec: Array,
                        ed: Array,
                        _a=ang_a,
                        _b=ang_b,
                        _c=ang_c,
                        _d=ang_d,
                    ) -> Array:
                        center_p = (ea * center_a + eb * center_b) / (ea + eb)
                        center_q = (ec * center_c + ed * center_d) / (ec + ed)
                        return _primitive_eri(
                            _a,
                            _b,
                            _c,
                            _d,
                            rab,
                            rcd,
                            ea,
                            eb,
                            ec,
                            ed,
                            center_p,
                            center_q,
                            max_total,
                        )

                    values = jax.vmap(primitive)(exp_a, exp_b, exp_c, exp_d)
                    columns.append(jnp.sum(coeff * values))

    return jnp.stack(columns).reshape(
        shell_a.n_cartesian,
        shell_b.n_cartesian,
        shell_c.n_cartesian,
        shell_d.n_cartesian,
    )


class JaxGaussianBackend:
    """Native McMurchie-Davidson AO integral backend.

    Builds the overlap, kinetic, nuclear-attraction, electron-repulsion and
    nuclear-repulsion quantities for a fixed molecular system and basis. All
    tensors are JAX arrays differentiable with respect to nuclear positions.

    Args:
        system: The molecular system (nuclear positions and charges).
        basis: The contracted-Gaussian AO basis.
    """

    def __init__(self, system: MolecularSystem, basis: AtomicOrbitalBasis) -> None:
        """Store the system and basis."""
        self._system = system
        self._basis = basis
        self._nuclear_positions = jnp.asarray(system.positions)
        self._nuclear_charges = jnp.asarray(system.atomic_numbers).astype(
            self._nuclear_positions.dtype
        )

    @property
    def n_atomic_orbitals(self) -> int:
        """Number of atomic orbitals."""
        return self._basis.n_atomic_orbitals

    def _assemble_one_electron(self) -> tuple[Array, Array, Array]:
        """Assemble ``(S, T, V)`` by looping over shell pairs."""
        n_ao = self._basis.n_atomic_orbitals
        dtype = self._nuclear_positions.dtype
        overlap = jnp.zeros((n_ao, n_ao), dtype=dtype)
        kinetic = jnp.zeros((n_ao, n_ao), dtype=dtype)
        nuclear = jnp.zeros((n_ao, n_ao), dtype=dtype)

        for shell_a in self._basis.shells:
            for shell_b in self._basis.shells:
                s_block, t_block = _shell_pair_overlap_kinetic(shell_a, shell_b)
                v_block = _shell_pair_nuclear(
                    shell_a,
                    shell_b,
                    self._nuclear_positions,
                    self._nuclear_charges,
                )
                rows = slice(shell_a.ao_offset, shell_a.ao_offset + shell_a.n_cartesian)
                cols = slice(shell_b.ao_offset, shell_b.ao_offset + shell_b.n_cartesian)
                overlap = overlap.at[rows, cols].set(s_block)
                kinetic = kinetic.at[rows, cols].set(t_block)
                nuclear = nuclear.at[rows, cols].set(v_block)
        return overlap, kinetic, nuclear

    def overlap(self) -> Array:
        """Return the AO overlap matrix ``S``."""
        overlap, _, _ = self._assemble_one_electron()
        return overlap

    def kinetic(self) -> Array:
        """Return the AO kinetic-energy matrix ``T``."""
        _, kinetic, _ = self._assemble_one_electron()
        return kinetic

    def nuclear_attraction(self) -> Array:
        """Return the AO nuclear-attraction matrix ``V``."""
        _, _, nuclear = self._assemble_one_electron()
        return nuclear

    def core_hamiltonian(self) -> Array:
        """Return the core Hamiltonian ``T + V``."""
        _, kinetic, nuclear = self._assemble_one_electron()
        return kinetic + nuclear

    def electron_repulsion(self) -> Array:
        """Return the ERI tensor ``(ij|kl)`` in chemist notation.

        Each shell quartet (static structure) contributes a compact block built
        by vectorising the primitive-quartet sum with :func:`jax.vmap`.
        """
        n_ao = self._basis.n_atomic_orbitals
        dtype = self._nuclear_positions.dtype
        eri = jnp.zeros((n_ao, n_ao, n_ao, n_ao), dtype=dtype)
        shells = self._basis.shells

        for shell_a in shells:
            for shell_b in shells:
                rab = shell_a.center - shell_b.center
                for shell_c in shells:
                    for shell_d in shells:
                        rcd = shell_c.center - shell_d.center
                        block = _shell_quartet_eri(shell_a, shell_b, shell_c, shell_d, rab, rcd)
                        rows = slice(
                            shell_a.ao_offset,
                            shell_a.ao_offset + shell_a.n_cartesian,
                        )
                        cols = slice(
                            shell_b.ao_offset,
                            shell_b.ao_offset + shell_b.n_cartesian,
                        )
                        kets = slice(
                            shell_c.ao_offset,
                            shell_c.ao_offset + shell_c.n_cartesian,
                        )
                        lets = slice(
                            shell_d.ao_offset,
                            shell_d.ao_offset + shell_d.n_cartesian,
                        )
                        eri = eri.at[rows, cols, kets, lets].set(block)
        return eri

    def nuclear_repulsion(self) -> Array:
        """Return the nuclear-repulsion energy ``sum_{A<B} Z_A Z_B / R_AB``.

        The squared inter-nuclear distance is computed only on the strict upper
        triangle (``i < j``); the diagonal, where ``R_AB = 0``, is replaced with
        a finite value *before* the square root so the gradient with respect to
        the nuclear positions stays finite (the standard ``jnp.where`` masked
        ``sqrt`` is gradient-unsafe at zero -- needed for analytic forces).
        """
        positions = self._nuclear_positions
        charges = self._nuclear_charges
        diff = positions[:, None, :] - positions[None, :, :]
        squared = jnp.sum(diff**2, axis=-1)
        charge_products = charges[:, None] * charges[None, :]
        n_atoms = positions.shape[0]
        upper = jnp.triu(jnp.ones((n_atoms, n_atoms), dtype=bool), k=1)
        # Guard the squared distance before the sqrt so no NaN gradient leaks in
        # from the (masked-out) diagonal / lower triangle.
        safe_squared = jnp.where(upper, squared, 1.0)
        distances = jnp.sqrt(safe_squared)
        contributions = jnp.where(upper, charge_products / distances, 0.0)
        return jnp.sum(contributions)


__all__ = [
    "JaxGaussianBackend",
    "QCBackend",
    "boys_function",
    "boys_vector",
    "hermite_coulomb",
    "hermite_expansion",
]

"""Pressure solve for incompressible flow projection.

Implements the Helmholtz-Hodge decomposition on a collocated grid:

    v* = v - grad(p)   where   div(grad(p)) = div(v)

The Poisson operator is the exact composition ``divergence(gradient(.))`` of the two
operators the projection itself applies, never a separately discretised Laplacian. On a
collocated grid those are not the same operator: ``gradient`` and ``divergence`` are
two-point central differences, so their composition reaches ``i +/- 2`` and differs from the
compact three-point ``laplacian``. Inverting the compact stencil and correcting with the
wide gradient leaves a field that is not divergence free under the divergence that measured
it, so the result is not a projection at all.

Composing the operator instead of rediscretising it makes the solve an exact discrete
Helmholtz decomposition, at the price of the odd-even decoupling inherent to a collocated
arrangement: the composed operator annihilates the constant and the checkerboard modes.
Under periodic boundaries the gradient annihilates them too, so the pressure is undetermined
along them and the projected velocity does not depend on which representative is chosen.
Under the other boundaries it does not, which is why the iterative solve below takes the
minimum-norm solution rather than any least-squares solution. Avoiding the decoupling
altogether is a change of grid, not of solver: a staggered arrangement (Harlow, Welch 1965)
makes the composition collapse to the compact stencil, and momentum interpolation
(Rhie, Chow 1983) restores the coupling on a collocated one.

References:
    * Chorin 1968 -- *Numerical solution of the Navier-Stokes equations*,
      Math. Comp. 22(104), 745.
    * Harlow, Welch 1965 -- *Numerical calculation of time-dependent viscous
      incompressible flow of fluid with free surface*, Phys. Fluids 8(12), 2182.
    * Rhie, Chow 1983 -- *Numerical study of the turbulent flow past an airfoil with
      trailing edge separation*, AIAA J. 21(11), 1525.
    * Fong, Saunders 2011 -- *LSMR: An iterative algorithm for sparse least-squares
      problems*, SIAM J. Sci. Comput. 33(5), 2950.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — imported at runtime by design

import jax
import jax.numpy as jnp

from opifex.fields.field import CenteredGrid, Extrapolation
from opifex.fields.operations import divergence, gradient
from opifex.uncertainty.linalg.lstsq import lsmr


_DEFAULT_MATVECS = 500


def _poisson_symbol(resolution: tuple[int, ...], dx: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The eigenvalues of ``divergence(gradient(.))`` on a periodic grid, and its null modes.

    A two-point central difference multiplies the Fourier mode of index ``j`` along an axis
    of ``n`` cells by ``i sin(theta) / h`` with ``theta = 2 pi j / n``, so the composition
    multiplies it by ``-sum_d sin^2(theta_d) / h_d^2``. That vanishes exactly when every
    ``2 j_d`` is a multiple of ``n_d``, which is read off the indices rather than off
    ``sin(theta)``: at the Nyquist mode ``sin(pi)`` evaluates to ``-8.7e-08`` rather than to
    zero, and dividing by it costs every digit of the modes that carry the solution.

    Args:
        resolution: Number of cells along each axis.
        dx: Cell size along each axis.

    Returns:
        Tuple of (negated eigenvalues, mask of the modes the operator annihilates).
    """
    indices = jnp.meshgrid(*(jnp.arange(n) for n in resolution), indexing="ij")
    axes = list(enumerate(zip(indices, resolution, strict=True)))
    eigenvalues = [(jnp.sin(2.0 * jnp.pi * index / n) / dx[axis]) ** 2 for axis, (index, n) in axes]
    annihilated = [(2 * index) % n == 0 for _, (index, n) in axes]
    return jnp.sum(jnp.stack(eigenvalues), axis=0), jnp.all(jnp.stack(annihilated), axis=0)


def _poisson_operator(
    velocity: CenteredGrid, shape: tuple[int, ...]
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    """``divergence(gradient(.))`` on flattened pressure values, and its transpose.

    Args:
        velocity: The field whose box and boundary the pressure shares.
        shape: Spatial shape of the pressure field.

    Returns:
        Tuple of (operator, transposed operator), both on flattened values.
    """

    def operator(values: jax.Array) -> jax.Array:
        candidate = CenteredGrid(values.reshape(shape), velocity.box, velocity.extrapolation)
        return divergence(gradient(candidate)).values.ravel()

    transpose = jax.linear_transpose(operator, jnp.zeros(shape).ravel())
    return operator, lambda cotangent: transpose(cotangent)[0]


def pressure_solve_spectral(
    velocity: CenteredGrid,
) -> tuple[CenteredGrid, CenteredGrid]:
    """Project a periodic velocity field with an exact FFT solve.

    Divides by the operator's own symbol, so the pressure solves
    ``divergence(gradient(p)) = divergence(v)`` to rounding rather than approximately.

    Args:
        velocity: Vector velocity field, shape ``(*resolution, ndim)``.

    Returns:
        Tuple of (divergence-free velocity, pressure).

    Raises:
        ValueError: If boundary conditions are not periodic.
    """
    if velocity.extrapolation != Extrapolation.PERIODIC:
        raise ValueError("Spectral pressure solve requires periodic boundaries")

    target = divergence(velocity)
    symbol, null = _poisson_symbol(target.resolution, target.dx)

    target_hat = jnp.fft.fftn(target.values)
    pressure_hat = jnp.where(null, 0.0, -target_hat / jnp.where(null, 1.0, symbol))
    pressure = CenteredGrid(
        jnp.real(jnp.fft.ifftn(pressure_hat)), velocity.box, velocity.extrapolation
    )

    return velocity - gradient(pressure), pressure


def pressure_solve_lsmr(
    velocity: CenteredGrid,
    num_matvecs: int = _DEFAULT_MATVECS,
) -> tuple[CenteredGrid, CenteredGrid]:
    """Project a velocity field with a matrix-free least-squares solve.

    Carries any boundary ``gradient`` and ``divergence`` support, and costs two applications
    of the operator per iteration where ``pressure_solve_spectral`` costs one transform pair
    in total: at 128x128 the default budget runs about a hundred times longer than the
    spectral solve, so a periodic field belongs in that one. The operator is applied by
    composing them and its transpose comes from ``jax.linear_transpose``, so the boundary
    treatment is the field's own and is the same on both sides of the solve. LSMR is what
    the composed operator asks for rather than a symmetric solver: under a zero-gradient
    boundary the composition is neither symmetric -- half its entries differ from their
    transpose -- nor invertible, and a conjugate-gradient iterate drifts into the null space
    and diverges. LSMR returns the minimum-norm least-squares solution in both cases.

    **The budget does not scale, and the caller must set it.** A least-squares solve works
    against the normal-equation condition number, which for this operator grows like the
    fourth power of the grid, so the iterations needed grow like its square. Measured
    iterations to reach the float32 round-off floor:

    ================  ======  ======  ==================
    boundary          n=16    n=32    n=64
    ================  ======  ======  ==================
    periodic            50      50      50
    zero                50     200     800
    zero-gradient      100     400     not reached by 3200
    ================  ======  ======  ==================

    The default is calibrated for 32 cells and **under-converges above it**: at 64 cells a
    zero-gradient boundary leaves 3.0e-02 of the incoming divergence rather than the 7e-06
    it leaves at 32. A periodic field belongs in ``pressure_solve_spectral``, which is exact
    and about a hundred times faster at 128x128. Iterating far past convergence costs
    accuracy rather than gaining it, as the bidiagonalisation loses orthogonality in float32.

    The zero-gradient row is worse than the operator should make it: that boundary is the one
    where the composition measures asymmetric, which traces to the boundary treatment of
    derived fields rather than to anything about Neumann conditions.

    Args:
        velocity: Vector velocity field, shape ``(*resolution, ndim)``.
        num_matvecs: Number of LSMR iterations.

    Returns:
        Tuple of (divergence-free velocity, pressure).
    """
    target = divergence(velocity)
    shape = target.values.shape
    operator, transpose = _poisson_operator(velocity, shape)

    pressure_values = lsmr(
        matvec=operator,
        matvec_transpose=transpose,
        rhs=target.values.ravel(),
        dim_cols=target.values.size,
        num_matvecs=num_matvecs,
    )
    pressure = CenteredGrid(pressure_values.reshape(shape), velocity.box, velocity.extrapolation)

    return velocity - gradient(pressure), pressure


__all__ = ["pressure_solve_lsmr", "pressure_solve_spectral"]

r"""Kinetic-energy Laplacian operators for neural wavefunctions.

The local kinetic energy of a VMC wavefunction is

.. math::

    -\tfrac12 \frac{\nabla^2 \psi}{\psi}
    = -\tfrac12 \Big( \nabla^2 \log|\psi| + \|\nabla \log|\psi|\|^2 \Big),

so the Hamiltonian needs both the Laplacian and the squared gradient of
``log|psi|`` with respect to the electron coordinates. This module provides two
interchangeable ways to obtain them:

#. :func:`jvp_grad_laplacian` -- a reference *oracle*: linearise ``grad`` once
   and read the Hessian diagonal one coordinate at a time (DeepMind FermiNet's
   ``hamiltonian.local_kinetic_energy`` ``default`` method, ``jvp``-over-
   ``grad`` with an eye loop). Simple and obviously correct; used to gate the
   fast path in tests.

#. :func:`forward_laplacian` -- a native *forward-Laplacian*: propagate the
   value, the full Jacobian and the Laplacian accumulator through a **single**
   forward pass using the two-stacked-JVP identity

   .. math::

       \partial_v^2 f(x) = \mathrm{JVP}_v\big[\, y \mapsto \mathrm{JVP}_v[f](y)\,\big](x),

   summed over an orthonormal basis ``v`` of the input space. This is the
   LapNet / ``fwdlap`` (Chen, 2023) speed-up: the Hessian diagonal is obtained
   with ``O(1)`` forward passes (vectorised over the basis) rather than
   ``O(n)`` reverse-then-forward passes, and JAX's symbolic-zero tangent
   handling gives the same sparsity benefit on the value/Jacobian carry. No
   ``folx``/``fwdlap`` dependency -- it is expressed purely with
   :func:`jax.jvp` and :func:`jax.vmap`.

Both functions take a scalar-valued ``fn: positions -> log|psi|`` and a single
walker's positions, and return ``(value, laplacian, gradient)``. They are pure
JAX and therefore ``jit`` / ``vmap`` clean.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # noqa: TC002


def jvp_grad_laplacian(
    fn: Callable[[Array], Array],
    positions: Float[Array, "nelectron ndim"],
) -> tuple[Array, Array, Array]:
    r"""Reference Laplacian via ``jvp``-over-``grad`` with an eye loop.

    Linearises ``grad fn`` at ``positions`` once, then accumulates the Hessian
    diagonal by pushing each Cartesian unit vector through the linearised map.
    This is the obviously-correct oracle the fast path is validated against.

    Args:
        fn: A scalar function of the electron coordinates, e.g. ``log|psi|``.
        positions: Electron coordinates of shape ``(nelectron, ndim)``.

    Returns:
        A ``(value, laplacian, gradient)`` tuple. ``value`` and ``laplacian``
        are scalars; ``gradient`` has the shape of ``positions``.
    """
    flat = positions.reshape(-1)
    n = flat.shape[0]

    def flat_fn(x: Array) -> Array:
        return fn(x.reshape(positions.shape))

    value = flat_fn(flat)
    grad_fn = jax.grad(flat_fn)
    gradient, hvp = jax.linearize(grad_fn, flat)
    eye = jnp.eye(n, dtype=flat.dtype)

    def diagonal_entry(i: int, total: Array) -> Array:
        return total + hvp(eye[i])[i]

    laplacian = jax.lax.fori_loop(0, n, diagonal_entry, jnp.zeros((), dtype=flat.dtype))
    return value, laplacian, gradient.reshape(positions.shape)


def forward_laplacian(
    fn: Callable[[Array], Array],
    positions: Float[Array, "nelectron ndim"],
) -> tuple[Array, Array, Array]:
    r"""Native forward-Laplacian via two stacked JVPs over an orthonormal basis.

    For a scalar ``fn`` the Laplacian is the sum of second directional
    derivatives along an orthonormal basis ``{v_i}`` of the input space:

    .. math::

        \nabla^2 f = \sum_i \partial_{v_i}^2 f, \qquad
        \partial_v^2 f(x) = \tfrac{d^2}{dt^2} f(x + t v)\big|_{t=0}.

    Each ``\partial_{v_i}^2 f`` is computed by differentiating the first
    directional derivative again *in the same direction* -- two stacked
    :func:`jax.jvp` calls -- and the basis is vectorised with :func:`jax.vmap`,
    so value, Jacobian and Laplacian propagate together through one forward pass
    (the LapNet / ``fwdlap`` scheme). The gradient is recovered for free from
    the first-order tangents.

    Args:
        fn: A scalar function of the electron coordinates, e.g. ``log|psi|``.
        positions: Electron coordinates of shape ``(nelectron, ndim)``.

    Returns:
        A ``(value, laplacian, gradient)`` tuple. ``value`` and ``laplacian``
        are scalars; ``gradient`` has the shape of ``positions``.
    """
    flat = positions.reshape(-1)
    n = flat.shape[0]

    def flat_fn(x: Array) -> Array:
        return fn(x.reshape(positions.shape))

    def first_directional(x: Array, direction: Array) -> tuple[Array, Array]:
        """Return ``(fn(x), d/dt fn(x + t direction))``."""
        return jax.jvp(flat_fn, (x,), (direction,))

    def second_directional(direction: Array) -> tuple[Array, Array]:
        """Return the first and second directional derivatives along ``direction``."""
        # Differentiate ``t -> first_directional(x, direction)[1]`` again along
        # the same direction: the primal is the first derivative, the tangent is
        # the second derivative (the per-direction Hessian quadratic form).
        (_, first), (_, second) = jax.jvp(
            lambda x: first_directional(x, direction),
            (flat,),
            (direction,),
        )
        return first, second

    eye = jnp.eye(n, dtype=flat.dtype)
    firsts, seconds = jax.vmap(second_directional)(eye)
    value = flat_fn(flat)
    gradient = firsts.reshape(positions.shape)
    laplacian = jnp.sum(seconds)
    return value, laplacian, gradient


__all__ = ["forward_laplacian", "jvp_grad_laplacian"]

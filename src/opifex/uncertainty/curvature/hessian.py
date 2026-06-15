"""Hessian-vector product primitives.

Computes ``Hv = d/dε ∇f(x + ε v) |_{ε=0}`` via forward-over-reverse mode
without materialising the dense Hessian.

Canonical reference (line-by-line port):
* ``../jax/jax/_src/api.py`` — ``jax.jvp`` / ``jax.grad`` building
  blocks; the HVP recipe is the standard pattern from
  Pearlmutter (1994) *Fast Exact Multiplication by the Hessian*.

References
----------
* Pearlmutter, B. A. 1994 — *Fast Exact Multiplication by the Hessian*,
  Neural Computation 6(1).
* Martens, J. & Sutskever, I. 2012 — *Training Deep and Recurrent
  Networks with Hessian-Free Optimization*.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax


def hessian_vector_product(
    scalar_function: Callable[[jax.Array], jax.Array],
    point: jax.Array,
    vector: jax.Array,
) -> jax.Array:
    """Compute ``H v`` for ``H = ∇² scalar_function(point)`` via forward-over-reverse.

    Args:
        scalar_function: Maps a parameter array ``x`` to a scalar loss
            ``f(x)``.
        point: Parameter point ``x`` at which to evaluate the Hessian.
        vector: Direction ``v``; ``Hv`` returned has the same shape as
            ``point``.

    Returns:
        ``H v`` — same shape as ``point`` and ``vector``.
    """
    gradient = jax.grad(scalar_function)
    _, hvp = jax.jvp(gradient, (point,), (vector,))
    return hvp

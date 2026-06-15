"""Cornell / Hasofer-Lind reliability index ``beta = -Phi^{-1}(p_f)``.

The reliability index reports the limit-state safety margin in units
of standard deviation of the equivalent standard-normal random
variable. ``p_f = 0.5 → beta = 0`` (worst case), and ``beta`` grows
monotonically as ``p_f`` shrinks.

Reference: Hasofer, A. M. & Lind, N. C. (1974), "Exact and invariant
second-moment code format", J. Eng. Mech. Div. ASCE 100(1), 111–121.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def reliability_index(failure_probability: jax.Array) -> jax.Array:
    """Return the Cornell / Hasofer-Lind reliability index ``beta``.

    Args:
        failure_probability: Scalar (or batched) failure probability
            in ``[0, 1]``. Values at the open endpoints map to
            ``±inf`` via the inverse standard-normal CDF.

    Returns:
        ``beta = -Phi^{-1}(p_f)`` with the same shape as the input.
    """
    # ``jsp.stats.norm.ppf`` is JIT- and grad-compatible end-to-end;
    # we negate to follow the engineering convention that bigger beta
    # means safer.
    return -jsp.stats.norm.ppf(jnp.asarray(failure_probability))


__all__ = ["reliability_index"]

"""Higher-moment Hutchinson trace estimator.

Given a matrix-free symmetric operator ``A`` and a tuple of integer
powers ``(p_1, ..., p_M)``, ``trace_moments`` returns the empirical
average of ``(v.T A v)**p_m`` over Rademacher probes for each ``p_m``.

The first moment (``p_m = 1``) is the standard Hutchinson trace
estimate; higher moments enable variance, skewness, and confidence-
interval reconstruction in a single multi-output pass without re-running
matvecs. Sibling reference (line-by-line port):
``matfree/matfree/stochtrace.py::integrand_wrap_moments`` plus the outer
Hutchinson aggregator.

References
----------
* Hutchinson 1990 — stochastic trace estimator with Rademacher probes.
* matfree — Krämer arXiv:2405.17277, ``stochtrace.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from collections.abc import Callable


def trace_moments(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    dim: int,
    num_samples: int,
    powers: tuple[int, ...],
    key: jax.Array,
) -> tuple[jax.Array, ...]:
    """Estimate raw moments of the per-probe quadratic form ``v.T A v``.

    Args:
        matvec: symmetric matvec callable.
        dim: matrix dimension ``n`` (static).
        num_samples: number of Rademacher probes (static).
        powers: tuple of integer powers — one entry per moment to
            compute. ``(1,)`` returns the Hutchinson trace estimate.
            ``(1, 2)`` adds the second raw moment for variance recovery
            via ``Var = E[X^2] - E[X]^2``.
        key: PRNG key.

    Returns:
        Tuple of scalar arrays, one per power in ``powers``. The ``m``-th
        entry equals ``(1/K) * sum_k (v_k.T A v_k)**powers[m]`` over
        Rademacher probes ``v_k``.
    """
    bernoulli = jax.random.bernoulli(key, shape=(num_samples, dim))
    probes = 2.0 * bernoulli.astype(jnp.float32) - 1.0
    quadratic_forms = jax.vmap(lambda probe: probe @ matvec(probe))(probes)
    return tuple(jnp.mean(quadratic_forms**power) for power in powers)


__all__ = ["trace_moments"]

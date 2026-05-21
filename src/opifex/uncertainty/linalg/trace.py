"""Matrix-free stochastic trace estimators.

These estimators consume an opaque ``matvec`` callable and a PRNG key and
return an estimate of ``trace(A)``. They are pure JAX functions and pass
``jax.jit`` / ``jax.vmap`` / ``jax.grad`` transforms.

Algorithms
----------
* ``hutchinson_trace`` — Hutchinson 1990. Rademacher probes; unbiased with
  variance ``2 * (||A||_F^2 - ||diag(A)||^2) / num_samples`` for symmetric
  ``A``. Sibling reference: ``matfree/stochtrace.py`` (``estimator`` +
  ``sampler_rademacher`` + ``integrand_trace``).
* ``hutch_plus_plus_trace`` — Meyer et al. arXiv:2010.09649. Extracts the
  leading subspace via a random sketch and applies Hutchinson on the
  residual. Matches the matrix trace exactly when the spectrum is supported
  on a subspace of dimension at most ``num_samples // 3``. Sibling
  reference: ``traceax/src/traceax/_estimators.py`` (``HutchPlusPlusEstimator``).

References
----------
* Hutchinson 1990 — *A stochastic estimator of the trace of the influence
  matrix*.
* Meyer, Musco, Musco, Woodruff arXiv:2010.09649 — *Hutch++: Optimal
  stochastic trace estimation*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from collections.abc import Callable


def hutchinson_trace(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    dim: int,
    num_samples: int,
    key: jax.Array,
) -> jax.Array:
    """Estimate ``trace(A)`` from a matrix-free ``matvec`` using Rademacher probes.

    Args:
        matvec: callable mapping a ``(dim,)`` vector to a ``(dim,)`` vector.
            Conceptually computes ``A @ v`` for the matrix whose trace is
            being estimated.
        dim: the dimension ``n`` of the operator. Static — used to allocate
            the probe array.
        num_samples: number of Rademacher probes. Static — controls variance.
        key: a PRNG key.

    Returns:
        A scalar JAX array — the empirical mean of ``v^T A v`` over the
        Rademacher probes. Unbiased for symmetric ``A``; variance scales as
        ``O(1 / num_samples)``.
    """
    bernoulli_probes = jax.random.bernoulli(key, shape=(num_samples, dim))
    rademacher_probes = 2.0 * bernoulli_probes.astype(jnp.float32) - 1.0
    quadratic_forms = jax.vmap(lambda probe: probe @ matvec(probe))(rademacher_probes)
    return jnp.mean(quadratic_forms)


def hutch_plus_plus_trace(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    dim: int,
    num_samples: int,
    key: jax.Array,
) -> jax.Array:
    """Estimate ``trace(A)`` via Hutch++ (Meyer et al. arXiv:2010.09649).

    The total query budget ``num_samples`` is split evenly across three
    stages — sketch, exact reduction, and Hutchinson residual — so the
    estimator uses ``2 * (num_samples // 3)`` matvecs to build a leading
    subspace and ``num_samples // 3`` Rademacher probes for the orthogonal
    residual.

    Args:
        matvec: callable mapping a ``(dim,)`` vector to a ``(dim,)`` vector.
        dim: the dimension ``n`` of the operator (static).
        num_samples: total matvec budget. Static; must be divisible by 3 to
            allocate equal sub-budgets cleanly.
        key: PRNG key; split into ``sketch`` and ``hutchinson`` substreams.

    Returns:
        A scalar JAX array — the Hutch++ trace estimate. Captures the
        contribution of the top ``num_samples // 3`` directions exactly and
        adds an unbiased Hutchinson term for the orthogonal residual.
    """
    sub_budget = num_samples // 3
    sketch_key, hutch_key = jax.random.split(key)

    sketch = jax.random.normal(sketch_key, (dim, sub_budget))
    sketched = jax.vmap(matvec, in_axes=1, out_axes=1)(sketch)
    basis, _ = jnp.linalg.qr(sketched)
    basis_image = jax.vmap(matvec, in_axes=1, out_axes=1)(basis)
    exact_trace_component = jnp.trace(basis.T @ basis_image)

    bernoulli_probes = jax.random.bernoulli(hutch_key, shape=(sub_budget, dim))
    probes = 2.0 * bernoulli_probes.astype(jnp.float32) - 1.0
    orthogonal_probes = probes - (probes @ basis) @ basis.T
    orthogonal_images = jax.vmap(matvec)(orthogonal_probes)
    quadratic_forms = jnp.einsum("ij,ij->i", orthogonal_probes, orthogonal_images)
    residual_component = jnp.mean(quadratic_forms)

    return exact_trace_component + residual_component


__all__ = ["hutch_plus_plus_trace", "hutchinson_trace"]

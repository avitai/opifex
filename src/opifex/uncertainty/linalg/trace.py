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
* ``xtrace`` — Epperly, Tropp, Webber arXiv:2301.07825. Exchangeable
  estimator that reuses each random vector in both the sketch and the
  residual stages via leave-one-out structure. Sibling reference:
  ``traceax/src/traceax/_estimators.py:170 XTraceEstimator``.
* ``xnys_trace`` — Epperly+ arXiv:2301.07825 §5. PSD-specialised variant
  using Nyström approximation instead of randomized SVD. Lower variance
  than XTrace for symmetric positive-definite operators (curvature /
  Fisher information matrices). Sibling reference:
  ``traceax/src/traceax/_estimators.py:241 XNysTraceEstimator``.

References
----------
* Hutchinson 1990 — *A stochastic estimator of the trace of the influence
  matrix*.
* Meyer, Musco, Musco, Woodruff arXiv:2010.09649 — *Hutch++: Optimal
  stochastic trace estimation*.
* Epperly, Tropp, Webber arXiv:2301.07825 — *XTrace: Making the most of
  every sample in stochastic trace estimation*.
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


def _sphere_sample(key: jax.Array, dim: int, num_samples: int) -> jax.Array:
    """Sample ``num_samples`` vectors uniformly on the sphere of radius ``sqrt(dim)``.

    Sibling reference: ``traceax/src/traceax/_samplers.py:93 SphereSampler``.
    Used by XTrace / XNysTrace for the theoretical-analysis isotropy property.
    """
    raw = jax.random.normal(key, (dim, num_samples))
    column_norms = jnp.linalg.norm(raw, axis=0, keepdims=True)
    return jnp.sqrt(jnp.asarray(dim, dtype=raw.dtype)) * raw / column_norms


def xtrace(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    dim: int,
    num_samples: int,
    key: jax.Array,
) -> jax.Array:
    """Estimate ``trace(A)`` via XTrace (Epperly+ arXiv:2301.07825).

    XTrace enforces exchangeability across the ``m = num_samples // 2``
    random probes by reusing each probe in both the sketch and the
    residual stages — implemented efficiently through leave-one-out
    algebra so the total cost is ``num_samples`` matvecs (``m`` for the
    sketch, ``m`` for the basis image ``Z = A Q``). The improved scaling
    uses ``traceax``'s ``_get_scale`` factor that orthogonalises probes
    against the low-rank approximation.

    Sibling reference (line-by-line port):
    ``traceax/src/traceax/_estimators.py:170 XTraceEstimator``.

    Args:
        matvec: callable mapping ``(dim,)`` to ``(dim,)``.
        dim: matrix dimension ``n`` (static).
        num_samples: total matvec budget. Static. ``num_samples // 2``
            sketch matvecs plus the same number of basis-image matvecs.
        key: PRNG key.

    Returns:
        Scalar JAX array — the mean of the per-probe XTrace estimates.
    """
    half_samples = num_samples // 2
    samples = _sphere_sample(key, dim, half_samples)

    image = jax.vmap(matvec, in_axes=1, out_axes=1)(samples)
    basis, upper = jnp.linalg.qr(image)
    inv_upper_t = jnp.linalg.inv(upper).T
    inv_upper_col_norms = jnp.linalg.norm(inv_upper_t, axis=0)
    normalised_inv_upper_t = inv_upper_t / inv_upper_col_norms

    basis_image = jax.vmap(matvec, in_axes=1, out_axes=1)(basis)
    h_matrix = basis.T @ basis_image
    w_matrix = basis.T @ samples
    t_matrix = basis_image.T @ samples
    hw_matrix = h_matrix @ w_matrix

    sw_diag = jnp.sum(normalised_inv_upper_t * w_matrix, axis=0)
    tw_diag = jnp.sum(t_matrix * w_matrix, axis=0)
    shs_diag = jnp.sum(normalised_inv_upper_t * (h_matrix @ normalised_inv_upper_t), axis=0)
    whw_diag = jnp.sum(w_matrix * hw_matrix, axis=0)

    term1 = sw_diag * jnp.sum((t_matrix - h_matrix.T @ w_matrix) * normalised_inv_upper_t, axis=0)
    term2 = (sw_diag**2) * shs_diag
    term3 = sw_diag * jnp.sum(normalised_inv_upper_t * (upper - hw_matrix), axis=0)

    scale = (dim - half_samples + 1) / (dim - jnp.linalg.norm(w_matrix, axis=0) ** 2 + sw_diag**2)

    estimates = (
        jnp.trace(h_matrix) * jnp.ones(half_samples)
        - shs_diag
        + (whw_diag - tw_diag + term1 + term2 + term3) * scale
    )
    return jnp.mean(estimates)


def xnys_trace(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    dim: int,
    num_samples: int,
    key: jax.Array,
) -> jax.Array:
    """Estimate ``trace(A)`` via XNysTrace for a PSD operator ``A``.

    Builds a Nyström approximation from ``num_samples`` matvecs and adds
    an exchangeable residual correction. On a PSD matrix whose rank is at
    most ``num_samples``, XNysTrace recovers the trace exactly up to
    numerical roundoff.

    Sibling reference (line-by-line port):
    ``traceax/src/traceax/_estimators.py:241 XNysTraceEstimator``.

    Args:
        matvec: callable mapping ``(dim,)`` to ``(dim,)``. Must be a
            positive-semi-definite operator; the algorithm forms a
            symmetric Cholesky factor of ``Ω^T A Ω`` internally.
        dim: matrix dimension ``n`` (static).
        num_samples: total matvec budget. Static.
        key: PRNG key.

    Returns:
        Scalar JAX array — the mean of the per-probe XNysTrace estimates.
    """
    samples = _sphere_sample(key, dim, num_samples)
    image = jax.vmap(matvec, in_axes=1, out_axes=1)(samples)

    shift = (
        jnp.finfo(image.dtype).eps
        * jnp.linalg.norm(image, "fro")
        / jnp.sqrt(jnp.asarray(dim, dtype=image.dtype))
    )
    shifted_image = image + samples * shift
    basis, upper = jnp.linalg.qr(shifted_image)

    gram = samples.T @ shifted_image
    cholesky_factor = jnp.linalg.cholesky(0.5 * (gram + gram.T)).T
    sketch_factor = jax.scipy.linalg.solve_triangular(cholesky_factor.T, upper.T, lower=True).T

    sample_basis, sample_upper = jnp.linalg.qr(samples)
    sample_w_matrix = sample_basis.T @ samples
    sample_inv_upper_t = jnp.linalg.inv(sample_upper).T
    sample_inv_upper_col_norms = jnp.linalg.norm(sample_inv_upper_t, axis=0)
    normalised_sample_inv_upper_t = sample_inv_upper_t / sample_inv_upper_col_norms

    scale = (dim - num_samples + 1) / (
        dim
        - jnp.linalg.norm(sample_w_matrix, axis=0) ** 2
        + jnp.sum(normalised_sample_inv_upper_t * sample_w_matrix, axis=0) ** 2
    )

    w_matrix = basis.T @ samples
    residual_factor = jax.scipy.linalg.solve_triangular(
        cholesky_factor, sketch_factor.T, lower=False
    ).T / jnp.sqrt(jnp.diag(jnp.linalg.inv(gram)))
    dsw_diag = jnp.sum(residual_factor * w_matrix, axis=0)

    estimates = (
        jnp.linalg.norm(sketch_factor, "fro") ** 2
        - jnp.linalg.norm(residual_factor, axis=0) ** 2
        + (dsw_diag**2) * scale
        - shift * dim
    )
    return jnp.mean(estimates)


__all__ = ["hutch_plus_plus_trace", "hutchinson_trace", "xnys_trace", "xtrace"]

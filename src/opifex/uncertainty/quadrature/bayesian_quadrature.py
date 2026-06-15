r"""JAX-native Bayesian quadrature on an RBF kernel + Gaussian measure.

Two closed-form algorithms live together in this module per the design
notes' fix #189 (Vanilla BQ ↔ WSABI-L coexistence):

* :func:`vanilla_bayesian_quadrature` — standard Gaussian-process
  Bayesian quadrature. With an isotropic RBF kernel and a diagonal
  Gaussian integration measure, the kernel mean and double-kernel-mean
  integrals admit closed forms (Briol et al., *Probabilistic
  Integration*, Statistical Science 34(1), 2019, §2.4). The posterior
  integral mean and variance follow from the standard BQ identities.

  Sibling reference (READ-ONLY port — never imported at runtime):
  ``emukit/quadrature/methods/vanilla_bq.py`` (``integrate``) and
  ``emukit/quadrature/kernels/quadrature_rbf.py`` (``qK`` / ``qKq``
  in :class:`QuadratureRBFGaussianMeasure`).

* :func:`wsabi_l_bayesian_quadrature` — Warped Sequential Active
  Bayesian Integration with linear approximation (Gunter et al.,
  NeurIPS 2014). Models a non-negative integrand ``f(x) = α + 0.5
  g(x)²`` where ``g`` is a GP, and computes the closed-form integral
  via the pairwise double-kernel integral. The variance term is
  intentionally not returned — WSABI-L is paired with uncertainty
  sampling in the canonical loop and does not require integral
  variance for acquisition.

  Sibling reference (READ-ONLY port — never imported at runtime):
  ``emukit/quadrature/methods/bounded_bq_model.py`` (``integrate``),
  ``emukit/quadrature/methods/wsabi.py`` (warping + ``alpha`` offset).

Both routines take diagonal-Gaussian measure parameters as direct
arrays (rather than a measure object) so they compose cleanly with
:func:`jax.jit` / :func:`jax.grad` without requiring the
:class:`opifex.uncertainty.quadrature.measures.GaussianMeasure`
dataclass to be a registered pytree.

References
----------
* Briol, F.-X. et al. 2019 — *Probabilistic Integration*, Statistical
  Science 34(1).
* Gunter, T. et al. 2014 — *Sampling for Inference in Probabilistic
  Models with Fast Bayesian Quadrature*, NeurIPS.
* O'Hagan, A. 1991 — *Bayes-Hermite Quadrature*, JSPI 29(3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _rbf_kernel_matrix(
    points_left: jax.Array,
    points_right: jax.Array,
    lengthscales: jax.Array,
    amplitude: jax.Array,
) -> jax.Array:
    r"""RBF kernel matrix with per-dimension lengthscales.

    ``K_ij = amplitude · exp(-0.5 Σ_k ((x_i - x_j)_k / ℓ_k)²)``.
    """
    scaled_difference = (points_left[:, None, :] - points_right[None, :, :]) / lengthscales
    squared_distance = jnp.sum(scaled_difference**2, axis=-1)
    return amplitude * jnp.exp(-0.5 * squared_distance)


def _rbf_gaussian_measure_kernel_mean(
    points: jax.Array,
    lengthscales: jax.Array,
    amplitude: jax.Array,
    measure_mean: jax.Array,
    measure_variance: jax.Array,
) -> jax.Array:
    r"""Closed-form ``∫ k(x, ·) p(x) dx`` for RBF + diagonal Gaussian measure.

    For an isotropic RBF kernel with per-dim lengthscales ``ℓ`` and a
    diagonal Gaussian measure ``N(b, diag(s²))``:

    ``qK(x') = amplitude · ∏_i sqrt(ℓ_i² / (ℓ_i² + s²_i)) ·
              exp(-Σ_i (x'_i - b_i)² / (2(ℓ_i² + s²_i)))``.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_rbf.py``
    ``QuadratureRBFGaussianMeasure.qK`` (line 150).
    """
    combined_variance = lengthscales**2 + measure_variance
    determinant_factor = jnp.prod(jnp.sqrt(lengthscales**2 / combined_variance))
    scaled_squared = jnp.sum((points - measure_mean) ** 2 / combined_variance, axis=-1)
    return amplitude * determinant_factor * jnp.exp(-0.5 * scaled_squared)


def _rbf_gaussian_measure_kernel_double_mean(
    lengthscales: jax.Array,
    amplitude: jax.Array,
    measure_variance: jax.Array,
) -> jax.Array:
    r"""Closed-form ``∫∫ k(x, x') p(x) p(x') dx dx'`` for RBF + diagonal Gaussian.

    ``qKq = amplitude · ∏_i sqrt(ℓ_i² / (ℓ_i² + 2 s²_i))``.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_rbf.py``
    ``QuadratureRBFGaussianMeasure.qKq`` (line 158).
    """
    return amplitude * jnp.prod(
        jnp.sqrt(lengthscales**2 / (lengthscales**2 + 2.0 * measure_variance))
    )


def vanilla_bayesian_quadrature(
    *,
    points: jax.Array,
    values: jax.Array,
    measure_mean: jax.Array,
    measure_variance: jax.Array,
    kernel_lengthscales: jax.Array,
    kernel_amplitude: jax.Array,
    noise_variance: jax.Array = jnp.asarray(1e-8),
) -> tuple[jax.Array, jax.Array]:
    r"""Closed-form vanilla Bayesian quadrature posterior moments.

    Computes the posterior integral mean and variance of an integrand
    ``f`` under a Gaussian-process prior with RBF kernel and a diagonal
    Gaussian integration measure. The closed forms are Briol+ 2019
    §2.4 equations 2.4 and 2.5.

    .. math::

        \mathrm{mean} &= q_K^T (K_{XX} + \sigma_n^2 I)^{-1} y \\
        \mathrm{var}  &= q_{Kq} - q_K^T (K_{XX} + \sigma_n^2 I)^{-1} q_K

    Sibling reference (READ-ONLY port — no runtime import):
    ``emukit/quadrature/methods/vanilla_bq.py:integrate``.

    Args:
        points: Integrand evaluation locations, shape ``(n, d)``.
        values: Integrand values, shape ``(n,)``.
        measure_mean: Gaussian measure mean, shape ``(d,)``.
        measure_variance: Gaussian measure diagonal variance, shape
            ``(d,)``, strictly positive entry-wise.
        kernel_lengthscales: Per-dim RBF lengthscales, shape ``(d,)``.
        kernel_amplitude: Scalar RBF amplitude (``σ²``).
        noise_variance: Observation noise variance (jitter) added to
            the Gram diagonal for numerical stability.

    Returns:
        ``(integral_mean, integral_variance)`` — both scalar arrays.
    """
    num_points = points.shape[0]
    gram_matrix = _rbf_kernel_matrix(points, points, kernel_lengthscales, kernel_amplitude)
    gram_regularised = gram_matrix + noise_variance * jnp.eye(num_points)
    kernel_mean_vector = _rbf_gaussian_measure_kernel_mean(
        points,
        kernel_lengthscales,
        kernel_amplitude,
        measure_mean,
        measure_variance,
    )
    double_kernel_mean = _rbf_gaussian_measure_kernel_double_mean(
        kernel_lengthscales, kernel_amplitude, measure_variance
    )
    weights = jnp.linalg.solve(gram_regularised, values)
    integral_mean = kernel_mean_vector @ weights
    integral_variance = double_kernel_mean - kernel_mean_vector @ jnp.linalg.solve(
        gram_regularised, kernel_mean_vector
    )
    return integral_mean, integral_variance


def wsabi_l_bayesian_quadrature(
    *,
    points: jax.Array,
    values: jax.Array,
    offset: jax.Array,
    measure_mean: jax.Array,
    measure_variance: jax.Array,
    kernel_lengthscales: jax.Array,
    kernel_amplitude: jax.Array,
    noise_variance: jax.Array = jnp.asarray(1e-8),
) -> jax.Array:
    r"""WSABI-L (Gunter et al, 2014) bounded-integrand BQ mean estimate.

    Models the non-negative integrand as ``f(x) = α + 0.5 g(x)²`` with
    ``g`` a GP, then integrates the linearised Taylor approximation.
    The pairwise double-kernel integral factors as

    .. math::

        \int k(x, x_i)\,k(x, x_j)\,p(x)\,dx
        = \sigma^2 \exp\bigl(-\|x_i - x_j\|^2/(4\ell^2)\bigr)
        \cdot q_K\bigl(\tfrac{x_i + x_j}{2};\, \ell/\sqrt{2}\bigr).

    The final mean is ``α + 0.5 Σ_{ij} w_i w_j (qK_ij)``.

    Sibling reference (READ-ONLY port — no runtime import):
    ``emukit/quadrature/methods/bounded_bq_model.py:integrate``.

    Args:
        points: Integrand evaluation locations, shape ``(n, d)``.
        values: Integrand values (must satisfy ``values >= offset``),
            shape ``(n,)``.
        offset: Lower-bound offset ``α`` (scalar). The integrand is
            modelled as ``α + 0.5 g(x)²`` so ``values - offset >= 0``
            is required for the square-root warping to be real.
        measure_mean: Gaussian measure mean, shape ``(d,)``.
        measure_variance: Gaussian measure diagonal variance, shape
            ``(d,)``.
        kernel_lengthscales: Per-dim RBF lengthscales, shape ``(d,)``.
        kernel_amplitude: Scalar RBF amplitude.
        noise_variance: Observation noise variance for numerical
            stability.

    Returns:
        Scalar integral mean estimate.
    """
    num_points = points.shape[0]
    warped_values = jnp.sqrt(jnp.maximum(2.0 * (values - offset), 0.0))
    gram_matrix = _rbf_kernel_matrix(points, points, kernel_lengthscales, kernel_amplitude)
    gram_regularised = gram_matrix + noise_variance * jnp.eye(num_points)
    weights = jnp.linalg.solve(gram_regularised, warped_values)

    pair_squared_distance = jnp.sum(
        ((points[:, None, :] - points[None, :, :]) / kernel_lengthscales) ** 2,
        axis=-1,
    )
    pair_exp_factor = kernel_amplitude * jnp.exp(-0.25 * pair_squared_distance)

    midpoints = 0.5 * (points[:, None, :] + points[None, :, :])
    flat_midpoints = midpoints.reshape(-1, points.shape[-1])
    scaled_lengthscales = kernel_lengthscales / jnp.sqrt(2.0)
    flat_kernel_mean = _rbf_gaussian_measure_kernel_mean(
        flat_midpoints,
        scaled_lengthscales,
        kernel_amplitude,
        measure_mean,
        measure_variance,
    )
    pairwise_double_kernel = flat_kernel_mean.reshape(num_points, num_points) * pair_exp_factor

    integral_second_term = 0.5 * weights @ pairwise_double_kernel @ weights
    return offset + integral_second_term

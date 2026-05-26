r"""Closed-form kernel mean embeddings for Bayesian-quadrature crosses.

Six ``(qK, qKq)`` pairs covering the canonical kernel-measure crosses
catalogued in the design notes:

* :func:`qk_rbf_gaussian` / :func:`qkq_rbf_gaussian` — RBF × diagonal
  Gaussian measure.
* :func:`qk_rbf_lebesgue` / :func:`qkq_rbf_lebesgue` — RBF × product
  Lebesgue measure.
* :func:`qk_matern12_product_lebesgue` /
  :func:`qkq_matern12_product_lebesgue` — product Matern-1/2 ×
  product Lebesgue measure.
* :func:`qk_matern32_product_lebesgue` /
  :func:`qkq_matern32_product_lebesgue` — product Matern-3/2 ×
  product Lebesgue measure.
* :func:`qk_matern52_product_lebesgue` /
  :func:`qkq_matern52_product_lebesgue` — product Matern-5/2 ×
  product Lebesgue measure.
* :func:`qk_brownian_lebesgue` / :func:`qkq_brownian_lebesgue` —
  Brownian motion (1-D) × Lebesgue measure on a positive interval.

All formulas are line-by-line ports of the closed forms in emukit's
``quadrature/kernels/`` package:

* ``../emukit/emukit/quadrature/kernels/quadrature_rbf.py`` —
  ``QuadratureRBFGaussianMeasure.qK`` (line 150) /
  ``.qKq`` (line 158); ``QuadratureRBFLebesgueMeasure.qK`` (line 105)
  / ``.qKq`` (line 113).
* ``../emukit/emukit/quadrature/kernels/quadrature_matern12.py`` —
  ``QuadratureProductMatern12LebesgueMeasure._qK_1d`` (line 93) /
  ``._qKq_1d`` (line 103).
* ``../emukit/emukit/quadrature/kernels/quadrature_matern32.py`` —
  ``QuadratureProductMatern32LebesgueMeasure._qK_1d`` (line 93) /
  ``._qKq_1d`` (line 103).
* ``../emukit/emukit/quadrature/kernels/quadrature_matern52.py`` —
  ``QuadratureProductMatern52LebesgueMeasure._qK_1d`` (line 93) /
  ``._qKq_1d`` (line 111).
* ``../emukit/emukit/quadrature/kernels/quadrature_brownian.py`` —
  ``QuadratureBrownianLebesgueMeasure.qK`` (line 89) / ``.qKq``
  (line 95).

The "product" Matern crosses factor across input dimensions: the
multi-dim ``qK`` is the per-dim product of 1-D ``qK``'s, and likewise
for ``qKq``. The implementation here takes per-dim arrays
(``lengthscales``, ``lower``, ``upper``) of shape ``(d,)`` and folds
the dimensions automatically.

References
----------
* Briol, F.-X. et al. 2019 — *Probabilistic Integration*, Statistical
  Science 34(1). (Survey including closed-form integrals.)
* Stein, M. L. 1999 — *Interpolation of Spatial Data: Some Theory for
  Kriging*. (Matern kernel reference.)
* Karatzas, I. & Shreve, S. E. 1991 — *Brownian Motion and Stochastic
  Calculus*. (Brownian kernel reference.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import erf


# ---------------------------------------------------------------------------
# RBF x Gaussian
# ---------------------------------------------------------------------------


def qk_rbf_gaussian(
    *,
    points: jax.Array,
    measure_mean: jax.Array,
    measure_variance: jax.Array,
    lengthscales: jax.Array,
    amplitude: jax.Array,
) -> jax.Array:
    r"""Closed-form ``∫ k(x, ·) p(x) dx`` for RBF + diagonal Gaussian measure.

    ``qK(x') = amplitude · ∏_i √(ℓ_i²/(ℓ_i² + σ²_i)) ·
              exp(-½ Σ_i (x'_i - μ_i)² / (ℓ_i² + σ²_i))``.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_rbf.py``
    ``QuadratureRBFGaussianMeasure.qK`` (line 150).
    """
    combined_variance = lengthscales**2 + measure_variance
    determinant_factor = jnp.prod(jnp.sqrt(lengthscales**2 / combined_variance))
    scaled_squared = jnp.sum((points - measure_mean) ** 2 / combined_variance, axis=-1)
    return amplitude * determinant_factor * jnp.exp(-0.5 * scaled_squared)


def qkq_rbf_gaussian(
    *,
    measure_variance: jax.Array,
    lengthscales: jax.Array,
    amplitude: jax.Array,
) -> jax.Array:
    r"""Closed-form ``∫∫ k(x, x') p(x) p(x') dx dx'`` for RBF + Gaussian.

    ``qKq = amplitude · ∏_i √(ℓ_i² / (ℓ_i² + 2 σ²_i))``.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_rbf.py``
    ``QuadratureRBFGaussianMeasure.qKq`` (line 158).
    """
    return amplitude * jnp.prod(
        jnp.sqrt(lengthscales**2 / (lengthscales**2 + 2.0 * measure_variance))
    )


# ---------------------------------------------------------------------------
# RBF x Lebesgue (isotropic kernel + product Lebesgue measure)
# ---------------------------------------------------------------------------


def qk_rbf_lebesgue(
    *,
    points: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    lengthscales: jax.Array,
    amplitude: jax.Array,
    density: jax.Array,
) -> jax.Array:
    r"""Closed-form ``qK`` for RBF × product Lebesgue measure.

    For each dimension ``i``:
    ``qK_i(x') = √(π/2) · ℓ_i · [erf((b_i - x'_i)/(√2 ℓ_i))
                                 - erf((a_i - x'_i)/(√2 ℓ_i))]``.

    Multi-dim ``qK`` is the per-dim product times the kernel amplitude
    and the (constant) measure density.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_rbf.py``
    ``QuadratureRBFLebesgueMeasure.qK`` (line 105).
    """
    scaled_diff_upper = (upper - points) / (jnp.sqrt(2.0) * lengthscales)
    scaled_diff_lower = (lower - points) / (jnp.sqrt(2.0) * lengthscales)
    per_dim = (
        jnp.sqrt(jnp.pi / 2.0) * lengthscales * (erf(scaled_diff_upper) - erf(scaled_diff_lower))
    )
    factored = jnp.prod(per_dim, axis=-1)
    return amplitude * density * factored


def qkq_rbf_lebesgue(
    *,
    lower: jax.Array,
    upper: jax.Array,
    lengthscales: jax.Array,
    amplitude: jax.Array,
    density: jax.Array,
) -> jax.Array:
    r"""Closed-form ``qKq`` for RBF × product Lebesgue measure.

    For each dimension with ``d_i = (b_i - a_i)/(√2 ℓ_i)``:
    ``qKq_i = 2 √π ℓ_i² · [(exp(-d_i²) - 1)/√π + erf(d_i) · d_i]``.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_rbf.py``
    ``QuadratureRBFLebesgueMeasure.qKq`` (line 113).
    """
    diff_scaled = (upper - lower) / (jnp.sqrt(2.0) * lengthscales)
    exp_term = (jnp.exp(-(diff_scaled**2)) - 1.0) / jnp.sqrt(jnp.pi)
    erf_term = erf(diff_scaled) * diff_scaled
    per_dim = 2.0 * jnp.sqrt(jnp.pi) * lengthscales**2 * (exp_term + erf_term)
    factored = jnp.prod(per_dim)
    return amplitude * density**2 * factored


# ---------------------------------------------------------------------------
# Matern-1/2 x product Lebesgue
# ---------------------------------------------------------------------------


def _qk_matern12_per_dim(
    points_one_dim: jax.Array,
    lower_one_dim: jax.Array,
    upper_one_dim: jax.Array,
    lengthscale_one_dim: jax.Array,
) -> jax.Array:
    r"""Per-dim Matern-1/2 closed form.

    ``qK_1d(x') = ℓ · (2 - exp((a - x')/ℓ) - exp((x' - b)/ℓ))``.
    """
    return lengthscale_one_dim * (
        2.0
        - jnp.exp((lower_one_dim - points_one_dim) / lengthscale_one_dim)
        - jnp.exp((points_one_dim - upper_one_dim) / lengthscale_one_dim)
    )


def qk_matern12_product_lebesgue(
    *,
    points: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    lengthscales: jax.Array,
    amplitude: jax.Array,
) -> jax.Array:
    r"""``qK`` for product Matern-1/2 (exponential) × product Lebesgue.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_matern12.py``
    ``QuadratureProductMatern12LebesgueMeasure._qK_1d`` (line 93).
    """
    per_dim = _qk_matern12_per_dim(points, lower, upper, lengthscales)
    return amplitude * jnp.prod(per_dim, axis=-1)


def qkq_matern12_product_lebesgue(
    *,
    lower: jax.Array,
    upper: jax.Array,
    lengthscales: jax.Array,
    amplitude: jax.Array,
) -> jax.Array:
    r"""``qKq`` for product Matern-1/2 × product Lebesgue.

    Per dim: ``qKq_1d = 2 ℓ · ((b - a) + ℓ · (exp(-(b - a)/ℓ) - 1))``.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_matern12.py``
    ``QuadratureProductMatern12LebesgueMeasure._qKq_1d`` (line 103).
    """
    interval = upper - lower
    per_dim = (
        2.0 * lengthscales * (interval + lengthscales * (jnp.exp(-interval / lengthscales) - 1.0))
    )
    return amplitude * jnp.prod(per_dim)


# ---------------------------------------------------------------------------
# Matern-3/2 x product Lebesgue
# ---------------------------------------------------------------------------


def _qk_matern32_per_dim(
    points_one_dim: jax.Array,
    lower_one_dim: jax.Array,
    upper_one_dim: jax.Array,
    lengthscale_one_dim: jax.Array,
) -> jax.Array:
    r"""Per-dim Matern-3/2 closed form."""
    s3 = jnp.sqrt(3.0)
    first_term = 4.0 * lengthscale_one_dim / s3
    second_term = -jnp.exp(s3 * (points_one_dim - upper_one_dim) / lengthscale_one_dim) * (
        upper_one_dim + 2.0 * lengthscale_one_dim / s3 - points_one_dim
    )
    third_term = -jnp.exp(s3 * (lower_one_dim - points_one_dim) / lengthscale_one_dim) * (
        points_one_dim + 2.0 * lengthscale_one_dim / s3 - lower_one_dim
    )
    return first_term + second_term + third_term


def qk_matern32_product_lebesgue(
    *,
    points: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    lengthscales: jax.Array,
    amplitude: jax.Array,
) -> jax.Array:
    r"""``qK`` for product Matern-3/2 × product Lebesgue.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_matern32.py``
    ``QuadratureProductMatern32LebesgueMeasure._qK_1d`` (line 93).
    """
    per_dim = _qk_matern32_per_dim(points, lower, upper, lengthscales)
    return amplitude * jnp.prod(per_dim, axis=-1)


def qkq_matern32_product_lebesgue(
    *,
    lower: jax.Array,
    upper: jax.Array,
    lengthscales: jax.Array,
    amplitude: jax.Array,
) -> jax.Array:
    r"""``qKq`` for product Matern-3/2 × product Lebesgue.

    Per dim with ``c = √3 (b - a)``:
    ``qKq_1d = (2 ℓ / 3) · (2c - 3ℓ + exp(-c/ℓ) (c + 3ℓ))``.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_matern32.py``
    ``QuadratureProductMatern32LebesgueMeasure._qKq_1d`` (line 103).
    """
    c = jnp.sqrt(3.0) * (upper - lower)
    per_dim = (
        (2.0 / 3.0)
        * lengthscales
        * (2.0 * c - 3.0 * lengthscales + jnp.exp(-c / lengthscales) * (c + 3.0 * lengthscales))
    )
    return amplitude * jnp.prod(per_dim)


# ---------------------------------------------------------------------------
# Matern-5/2 x product Lebesgue
# ---------------------------------------------------------------------------


def _qk_matern52_per_dim(
    points_one_dim: jax.Array,
    lower_one_dim: jax.Array,
    upper_one_dim: jax.Array,
    lengthscale_one_dim: jax.Array,
) -> jax.Array:
    r"""Per-dim Matern-5/2 closed form."""
    s5 = jnp.sqrt(5.0)
    first_term = 16.0 * lengthscale_one_dim / (3.0 * s5)
    diff_upper = upper_one_dim - points_one_dim
    diff_lower = points_one_dim - lower_one_dim
    second_bracket = (
        8.0 * s5 * lengthscale_one_dim**2
        + 25.0 * lengthscale_one_dim * diff_upper
        + 5.0 * s5 * diff_upper**2
    )
    second_term = (
        -jnp.exp(s5 * (points_one_dim - upper_one_dim) / lengthscale_one_dim)
        / (15.0 * lengthscale_one_dim)
        * second_bracket
    )
    third_bracket = (
        8.0 * s5 * lengthscale_one_dim**2
        + 25.0 * lengthscale_one_dim * diff_lower
        + 5.0 * s5 * diff_lower**2
    )
    third_term = (
        -jnp.exp(s5 * (lower_one_dim - points_one_dim) / lengthscale_one_dim)
        / (15.0 * lengthscale_one_dim)
        * third_bracket
    )
    return first_term + second_term + third_term


def qk_matern52_product_lebesgue(
    *,
    points: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    lengthscales: jax.Array,
    amplitude: jax.Array,
) -> jax.Array:
    r"""``qK`` for product Matern-5/2 × product Lebesgue.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_matern52.py``
    ``QuadratureProductMatern52LebesgueMeasure._qK_1d`` (line 93).
    """
    per_dim = _qk_matern52_per_dim(points, lower, upper, lengthscales)
    return amplitude * jnp.prod(per_dim, axis=-1)


def qkq_matern52_product_lebesgue(
    *,
    lower: jax.Array,
    upper: jax.Array,
    lengthscales: jax.Array,
    amplitude: jax.Array,
) -> jax.Array:
    r"""``qKq`` for product Matern-5/2 × product Lebesgue.

    Per dim with ``c = √5 (b - a)``:
    ``qKq_1d = (2 ℓ (8c - 15 ℓ) + 2 exp(-c/ℓ) (5a² - 10ab + 5b² + 7cℓ + 15ℓ²)) / 15``.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_matern52.py``
    ``QuadratureProductMatern52LebesgueMeasure._qKq_1d`` (line 111).
    """
    c = jnp.sqrt(5.0) * (upper - lower)
    bracket = (
        5.0 * lower**2
        - 10.0 * lower * upper
        + 5.0 * upper**2
        + 7.0 * c * lengthscales
        + 15.0 * lengthscales**2
    )
    per_dim = (
        2.0 * lengthscales * (8.0 * c - 15.0 * lengthscales)
        + 2.0 * jnp.exp(-c / lengthscales) * bracket
    ) / 15.0
    return amplitude * jnp.prod(per_dim)


# ---------------------------------------------------------------------------
# Brownian motion x Lebesgue (intrinsically 1-D, positive support)
# ---------------------------------------------------------------------------


def qk_brownian_lebesgue(
    *,
    points: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    amplitude: jax.Array,
    density: jax.Array,
) -> jax.Array:
    r"""Closed-form ``qK`` for Brownian motion × Lebesgue on ``[a, b]`` with ``a ≥ 0``.

    The Brownian kernel ``k(x, x') = σ² min(x, x')`` integrates against
    a uniform measure on ``[a, b]`` to give

    ``qK(x') = σ² · density · (b x' - ½ x'² - ½ a²)``.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_brownian.py``
    ``QuadratureBrownianLebesgueMeasure.qK`` (line 89).
    """
    points_1d = jnp.squeeze(points, axis=-1)
    kernel_mean = upper * points_1d - 0.5 * points_1d**2 - 0.5 * lower**2
    return amplitude * density * kernel_mean


def qkq_brownian_lebesgue(
    *,
    lower: jax.Array,
    upper: jax.Array,
    amplitude: jax.Array,
    density: jax.Array,
) -> jax.Array:
    r"""Closed-form ``qKq`` for Brownian motion × Lebesgue on ``[a, b]``.

    ``qKq = σ² · density² · (½ b (b² - a²) - (b³ - a³)/6 - ½ a² (b - a))``.

    Sibling reference: ``emukit/quadrature/kernels/quadrature_brownian.py``
    ``QuadratureBrownianLebesgueMeasure.qKq`` (line 95).
    """
    expression = (
        0.5 * upper * (upper**2 - lower**2)
        - (upper**3 - lower**3) / 6.0
        - 0.5 * lower**2 * (upper - lower)
    )
    return amplitude * density**2 * expression

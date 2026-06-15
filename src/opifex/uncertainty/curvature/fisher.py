r"""Empirical Fisher information primitives.

The empirical Fisher matrix at parameters ``θ`` for a per-sample loss
``ℓ_i(θ)`` is

.. math::

    F(θ) = (1/N) \\sum_i \\nabla_θ ℓ_i(θ)\\, \\nabla_θ ℓ_i(θ)^\\top.

The diagonal of this matrix is the most popular cheap curvature proxy
for Laplace approximations and Adam-style optimisers.

Canonical reference:
* Daxberger Laplace and bayesian-torch use this same per-sample outer
  product. opifex computes it via the canonical
  ``jax.vmap(jax.grad(per_sample_loss))`` recipe.

References
----------
* Daxberger, E. et al. 2021 — *Laplace Redux — Effortless Bayesian Deep
  Learning*, arXiv:2106.14806.
* Kunstner, F., Hennig, P., Balles, L. 2019 — *Limitations of the
  empirical Fisher approximation for natural gradient descent*,
  arXiv:1905.12558.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax
import jax.numpy as jnp


def empirical_fisher_diagonal(
    per_sample_loss: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    parameters: jax.Array,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """Per-parameter diagonal of the empirical Fisher matrix.

    Args:
        per_sample_loss: Maps ``(parameters, input, target) -> scalar``.
            Must accept a single sample (the leading batch axis is added
            by ``vmap``).
        parameters: Model parameters of arbitrary shape.
        inputs: Batched inputs ``(batch, ...)``.
        targets: Batched targets ``(batch, ...)``.

    Returns:
        Per-parameter diagonal estimate of the empirical Fisher, same
        shape as ``parameters``.
    """
    per_sample_gradient = jax.grad(per_sample_loss)
    batched_gradients = jax.vmap(per_sample_gradient, in_axes=(None, 0, 0))(
        parameters, inputs, targets
    )
    return jnp.mean(batched_gradients**2, axis=0)

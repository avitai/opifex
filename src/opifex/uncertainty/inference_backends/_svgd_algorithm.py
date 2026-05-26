r"""Stein Variational Gradient Descent (Liu+Wang 2016) JAX-native primitives.

Line-by-line port of the SVGD reference at
``../blackjax/blackjax/vi/svgd.py``. The algorithm evolves a finite
set of particles ``{x_i}`` according to the kernelised Stein gradient

.. math::

    \phi^*(x_j) = \frac{1}{n} \sum_i \bigl[
        k(x_i, x_j) \nabla \log p(x_i) + \nabla_{x_i} k(x_i, x_j)
    \bigr]

so that the empirical distribution of the particles converges to the
target ``p`` in MMD. The default RBF kernel uses the median heuristic
for its bandwidth, as in Liu+Wang 2016 §4.2 and the blackjax reference
(``update_median_heuristic`` at ``blackjax/vi/svgd.py:163``).

Sibling reference (READ-ONLY port — never imported at runtime):

* ``../blackjax/blackjax/vi/svgd.py`` — ``init`` (line 25),
  ``build_kernel`` (line 49), ``rbf_kernel`` (line 117),
  ``median_heuristic`` (line 138), ``update_median_heuristic``
  (line 163).

References
----------
* Liu, Q. & Wang, D. 2016 — *Stein Variational Gradient Descent: A
  General Purpose Bayesian Inference Algorithm*, NeurIPS 29.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager per opifex convention

import jax
import jax.numpy as jnp
import optax


def rbf_kernel(x: jax.Array, y: jax.Array, length_scale: jax.Array | float) -> jax.Array:
    r"""RBF kernel ``k(x, y) = exp(-||x - y||² / ℓ)`` for two particles.

    Sibling reference: ``blackjax/vi/svgd.py:rbf_kernel`` (line 117).

    Args:
        x: First particle, shape ``(d,)``.
        y: Second particle, shape ``(d,)``.
        length_scale: Bandwidth ``ℓ``. Note the blackjax convention
            ``1/ℓ`` not ``1/(2ℓ²)`` — the heuristic returns
            ``median² / log(n)``, which already includes the factor.

    Returns:
        Scalar kernel value.
    """
    squared_distance = jnp.sum((x - y) ** 2)
    return jnp.exp(-squared_distance / length_scale)


def median_heuristic_bandwidth(particles: jax.Array) -> jax.Array:
    r"""Median-heuristic RBF bandwidth.

    ``length_scale = median(pairwise distances)² / log(n_particles)``,
    matching ``blackjax/vi/svgd.py:median_heuristic`` (line 138).

    Args:
        particles: Particle array of shape ``(n_particles, d)``.

    Returns:
        Scalar bandwidth.
    """

    def _pairwise_distance(left: jax.Array, right: jax.Array) -> jax.Array:
        return jnp.linalg.norm(left - right)

    pairwise = jax.vmap(jax.vmap(_pairwise_distance, (None, 0)), (0, None))(particles, particles)
    lower_triangular = pairwise[jnp.tril_indices(pairwise.shape[0], k=-1)]
    median_distance = jnp.median(lower_triangular)
    return median_distance**2 / jnp.log(particles.shape[0])


def _phi_star(
    particles: jax.Array,
    target_log_prob_fn: Callable[[jax.Array], jax.Array],
    length_scale: jax.Array,
) -> jax.Array:
    r"""Compute the Stein-gradient particle update.

    ``φ*(x_j) = (1/n) Σ_i [k(x_i, x_j) ∇log p(x_i) + ∇_{x_i} k(x_i, x_j)]``.

    Sibling reference: ``blackjax/vi/svgd.py:build_kernel`` inner
    ``phi_star_summand`` (line 96).
    """
    grad_log_prob_fn = jax.grad(target_log_prob_fn)

    def summand(left_particle: jax.Array, right_particle: jax.Array) -> jax.Array:
        gradient = grad_log_prob_fn(left_particle)
        kernel_value, grad_kernel = jax.value_and_grad(rbf_kernel, argnums=0)(
            left_particle, right_particle, length_scale
        )
        return -(kernel_value * gradient) - grad_kernel

    def phi_star_at(right_particle: jax.Array) -> jax.Array:
        return jax.vmap(lambda left: summand(left, right_particle))(particles).mean(axis=0)

    return jax.vmap(phi_star_at)(particles)


def svgd_fit(
    *,
    initial_particles: jax.Array,
    target_log_prob_fn: Callable[[jax.Array], jax.Array],
    num_iterations: int,
    learning_rate: jax.Array | float,
    length_scale: jax.Array | float | None = None,
) -> jax.Array:
    r"""Run SVGD for ``num_iterations`` Adam steps over particle positions.

    If ``length_scale`` is ``None`` the median heuristic computes the
    RBF bandwidth from the current particle cloud (recomputed at each
    iteration, matching the blackjax ``update_median_heuristic``
    pattern). Pass a fixed scalar to skip the heuristic.

    Sibling reference: ``blackjax/vi/svgd.py:build_kernel`` (line 49)
    composed with ``update_median_heuristic`` (line 163) inside a
    fixed-step Adam optimisation.

    Args:
        initial_particles: Starting particle cloud, shape
            ``(n_particles, d)``.
        target_log_prob_fn: ``log p(x)`` mapping ``(d,) -> scalar``.
        num_iterations: Number of Adam steps. Static under
            :func:`jax.jit`.
        learning_rate: Adam learning rate.
        length_scale: RBF bandwidth ``ℓ``. ``None`` enables the median
            heuristic (recomputed each step from current particles).

    Returns:
        Final particle cloud, shape ``(n_particles, d)``.
    """
    optimizer = optax.adam(learning_rate)
    initial_opt_state = optimizer.init(initial_particles)

    def step(
        carry: tuple[jax.Array, optax.OptState], _: None
    ) -> tuple[tuple[jax.Array, optax.OptState], None]:
        particles, opt_state = carry
        bandwidth = (
            median_heuristic_bandwidth(particles)
            if length_scale is None
            else jnp.asarray(length_scale)
        )
        functional_gradient = _phi_star(particles, target_log_prob_fn, bandwidth)
        updates, opt_state = optimizer.update(functional_gradient, opt_state, particles)
        new_particles = jnp.asarray(optax.apply_updates(particles, updates))
        return (new_particles, opt_state), None

    (final_particles, _), _ = jax.lax.scan(
        step, (initial_particles, initial_opt_state), None, length=num_iterations
    )
    return jnp.asarray(final_particles)

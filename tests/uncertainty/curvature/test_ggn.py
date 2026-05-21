"""Tests for the generalized Gauss-Newton (GGN) vector product.

The GGN matrix at parameters ``θ`` for a model ``y = f(θ, x)`` and a
convex-in-output loss ``L(y, t)`` is

    G(θ) = J(θ)^T H_y L(f(θ, x), t) J(θ),

where ``J = ∂f/∂θ`` is the model Jacobian and ``H_y L = ∂²L/∂y²`` is the
output-space Hessian of the loss. It is a positive-semidefinite
approximation of the true Hessian that drops the second-order
parameter-dependence of the model — a standard pre-conditioner for
natural-gradient methods and the linearised Laplace posterior covariance.

Canonical reference (line-by-line port):
* ``../kfac-jax/kfac_jax/_src/loss_functions.py`` ``multiply_ggn``
  (line 206) for the GGN-vp signature; opifex uses the functional
  forward-over-reverse recipe ``vjp ∘ (H_y L) ∘ jvp`` from Martens 2014.

References
----------
* Schraudolph, N. N. 2002 — *Fast Curvature Matrix-Vector Products for
  Second-Order Gradient Descent*, Neural Computation 14(7).
* Martens, J. 2014 — *New Insights and Perspectives on the Natural
  Gradient Method*, arXiv:1412.1193.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.curvature import ggn_vector_product


def test_ggn_vector_product_matches_jt_hy_j_on_linear_model_squared_loss() -> None:
    """On a linear model with squared loss, ``G = X^T X / N``."""
    rng = jax.random.PRNGKey(0)
    key_x, key_t, key_p, key_v = jax.random.split(rng, 4)
    batch_size = 5
    in_dim = 3
    inputs = jax.random.normal(key_x, (batch_size, in_dim))
    targets = jax.random.normal(key_t, (batch_size,))
    params = jax.random.normal(key_p, (in_dim,))
    vector = jax.random.normal(key_v, (in_dim,))

    def model(parameters: jax.Array, batch: jax.Array) -> jax.Array:
        return batch @ parameters

    def loss(outputs: jax.Array, ground_truth: jax.Array) -> jax.Array:
        return 0.5 * jnp.mean((outputs - ground_truth) ** 2)

    ggn_v = ggn_vector_product(model, loss, params, inputs, targets, vector)
    expected = (inputs.T @ inputs / batch_size) @ vector
    assert jnp.allclose(ggn_v, expected, atol=1e-5)


def test_ggn_vector_product_is_linear_in_v() -> None:
    """``GGN (a v + b w) = a GGN v + b GGN w``."""
    rng = jax.random.PRNGKey(1)
    key_x, key_p = jax.random.split(rng)
    inputs = jax.random.normal(key_x, (4, 2))
    targets = jnp.zeros(4)
    params = jax.random.normal(key_p, (2,))

    def model(parameters: jax.Array, batch: jax.Array) -> jax.Array:
        return batch @ parameters

    def loss(outputs: jax.Array, ground_truth: jax.Array) -> jax.Array:
        return 0.5 * jnp.mean((outputs - ground_truth) ** 2)

    vector_v = jnp.asarray([1.0, 0.0])
    vector_w = jnp.asarray([0.0, 1.0])
    combined = 0.7 * vector_v + 1.3 * vector_w
    actual = ggn_vector_product(model, loss, params, inputs, targets, combined)
    expected = 0.7 * ggn_vector_product(
        model, loss, params, inputs, targets, vector_v
    ) + 1.3 * ggn_vector_product(model, loss, params, inputs, targets, vector_w)
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_ggn_vector_product_is_positive_semidefinite() -> None:
    """``v^T G v >= 0`` for any ``v`` (GGN is PSD by construction)."""
    rng = jax.random.PRNGKey(2)
    key_x, key_p, key_v = jax.random.split(rng, 3)
    inputs = jax.random.normal(key_x, (8, 3))
    targets = jnp.zeros(8)
    params = jax.random.normal(key_p, (3,))

    def model(parameters: jax.Array, batch: jax.Array) -> jax.Array:
        return batch @ parameters

    def loss(outputs: jax.Array, ground_truth: jax.Array) -> jax.Array:
        return 0.5 * jnp.mean((outputs - ground_truth) ** 2)

    for index in range(5):
        vector = jax.random.normal(jax.random.fold_in(key_v, index), (3,))
        quadratic_form = vector @ ggn_vector_product(model, loss, params, inputs, targets, vector)
        assert quadratic_form >= -1e-6


def test_ggn_vector_product_is_jit_compatible() -> None:
    """``ggn_vector_product`` works under ``jax.jit``."""
    inputs = jnp.eye(3)
    targets = jnp.zeros(3)
    params = jnp.asarray([1.0, 2.0, 3.0])

    def model(parameters: jax.Array, batch: jax.Array) -> jax.Array:
        return batch @ parameters

    def loss(outputs: jax.Array, ground_truth: jax.Array) -> jax.Array:
        return 0.5 * jnp.mean((outputs - ground_truth) ** 2)

    @jax.jit
    def call(vector: jax.Array) -> jax.Array:
        return ggn_vector_product(model, loss, params, inputs, targets, vector)

    result = call(jnp.asarray([1.0, 0.0, 0.0]))
    assert jnp.all(jnp.isfinite(result))


def test_ggn_vector_product_handles_multivariate_output() -> None:
    """A two-output linear regression: ``G = (1/N) X^T X ⊗ I`` per output dim."""
    batch_size = 4
    in_dim = 2
    out_dim = 2
    rng = jax.random.PRNGKey(3)
    key_x, key_w = jax.random.split(rng)
    inputs = jax.random.normal(key_x, (batch_size, in_dim))
    targets = jnp.zeros((batch_size, out_dim))
    params = jax.random.normal(key_w, (in_dim, out_dim))

    def model(parameters: jax.Array, batch: jax.Array) -> jax.Array:
        return batch @ parameters

    def loss(outputs: jax.Array, ground_truth: jax.Array) -> jax.Array:
        return 0.5 * jnp.mean(jnp.sum((outputs - ground_truth) ** 2, axis=-1))

    vector = jnp.ones_like(params)
    ggn_v = ggn_vector_product(model, loss, params, inputs, targets, vector)
    assert ggn_v.shape == params.shape
    assert jnp.all(jnp.isfinite(ggn_v))

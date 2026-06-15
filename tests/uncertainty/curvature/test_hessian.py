"""Tests for Hessian-vector product primitives.

The Hessian-vector product ``Hv`` of a scalar function ``f`` at ``x`` is
the directional derivative of ``grad f`` along ``v``:
``Hv = d/dε ∇f(x + ε v) |_{ε=0}``. Computed as
``jax.jvp(jax.grad(f), (x,), (v,))[1]`` — a single-pass forward-over-
reverse mode pipeline that avoids materialising the dense Hessian.

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

import jax
import jax.numpy as jnp

from opifex.uncertainty.curvature import hessian_vector_product


def test_hessian_vector_product_matches_dense_hessian_on_quadratic() -> None:
    """For ``f(x) = 0.5 x^T A x``, ``Hv = A v``."""
    matrix_a = jnp.asarray([[2.0, 0.3], [0.3, 1.5]])

    def quadratic(x: jax.Array) -> jax.Array:
        return 0.5 * x @ matrix_a @ x

    vector = jnp.asarray([1.0, -1.0])
    point = jnp.asarray([0.5, 1.0])
    hvp = hessian_vector_product(quadratic, point, vector)
    assert jnp.allclose(hvp, matrix_a @ vector, atol=1e-6)


def test_hessian_vector_product_matches_dense_hessian_on_nonquadratic() -> None:
    """For ``f(x) = sum sin(x)``, ``Hv = -sin(x) ⊙ v``."""

    def loss(x: jax.Array) -> jax.Array:
        return jnp.sum(jnp.sin(x))

    point = jnp.asarray([0.1, 0.5, -0.2])
    vector = jnp.asarray([1.0, 0.0, -1.0])
    expected = -jnp.sin(point) * vector
    hvp = hessian_vector_product(loss, point, vector)
    assert jnp.allclose(hvp, expected, atol=1e-6)


def test_hessian_vector_product_is_linear_in_v() -> None:
    """``H (a v + b w) = a H v + b H w``."""
    matrix_a = jnp.asarray([[1.0, 0.5], [0.5, 2.0]])

    def quadratic(x: jax.Array) -> jax.Array:
        return 0.5 * x @ matrix_a @ x

    point = jnp.zeros(2)
    vector_v = jnp.asarray([1.0, 0.0])
    vector_w = jnp.asarray([0.0, 1.0])
    combined = 2.5 * vector_v - 1.7 * vector_w
    actual = hessian_vector_product(quadratic, point, combined)
    expected = 2.5 * hessian_vector_product(
        quadratic, point, vector_v
    ) - 1.7 * hessian_vector_product(quadratic, point, vector_w)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_hessian_vector_product_is_jit_compatible() -> None:
    """``hessian_vector_product`` works inside ``jax.jit``."""

    def loss(x: jax.Array) -> jax.Array:
        return jnp.sum(x**4) / 4.0

    @jax.jit
    def call(point: jax.Array, vector: jax.Array) -> jax.Array:
        return hessian_vector_product(loss, point, vector)

    point = jnp.asarray([0.5, -0.3])
    vector = jnp.asarray([1.0, 1.0])
    expected = 3.0 * point**2 * vector
    assert jnp.allclose(call(point, vector), expected, atol=1e-5)


def test_hessian_vector_product_is_vmap_compatible() -> None:
    """``vmap`` over batched ``v`` extracts a batch of HVPs."""
    matrix_a = jnp.asarray([[1.0, 0.0], [0.0, 4.0]])

    def quadratic(x: jax.Array) -> jax.Array:
        return 0.5 * x @ matrix_a @ x

    point = jnp.zeros(2)
    vectors = jnp.eye(2)  # batch of 2 basis vectors
    hvps = jax.vmap(lambda v: hessian_vector_product(quadratic, point, v))(vectors)
    assert jnp.allclose(hvps, matrix_a, atol=1e-6)

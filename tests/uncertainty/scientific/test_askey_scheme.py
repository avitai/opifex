"""Tests for the full Askey-scheme polynomial-chaos bases (Feature F16).

Extends the Legendre / Hermite bases (Task 6.6) with the remaining
canonical Wiener-Askey correspondences (Xiu & Karniadakis 2002, Table 4.1):

* **Laguerre** (generalized) <-> Gamma / exponential inputs on ``[0, inf)``.
* **Jacobi** <-> Beta inputs on ``[-1, 1]``.

The decisive correctness criterion is *orthonormality under the matching
Gauss quadrature*: for each family the Gram matrix
``G_ij = <p_i, p_j> = sum_k w_k p_i(node_k) p_j(node_k)`` must equal the
identity to ~1e-6.  Closed-form low-degree values cross-check the
recurrence against Abramowitz & Stegun.

Quadrature nodes / weights are constructed via the Golub-Welsch
eigenvalue method; the construction is run under ``jax.enable_x64`` so the
orthonormality residual reaches ~1e-12 (float32 only reaches ~1e-4 for the
Laguerre weight, which has a heavy tail).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.scientific.polynomial_chaos import (
    _scalar_basis_one,
    evaluate_basis,
    gauss_quadrature,
)


# ---------------------------------------------------------------------------
# Orthonormality (the binding correctness criterion).
# ---------------------------------------------------------------------------


def _gram_matrix(family: str, max_degree: int, **params: float) -> jax.Array:
    """Quadrature Gram matrix ``G_ij = <p_i, p_j>`` for degrees ``0..max_degree``."""
    quad_order = max_degree + 4  # exact for products up to degree 2*max_degree
    nodes, weights = gauss_quadrature(family=family, order=quad_order, **params)
    degrees = jnp.arange(max_degree + 1)
    basis = evaluate_basis(family=family, degrees=degrees, x=nodes, **params)  # (K, P)
    return jnp.einsum("k,ki,kj->ij", weights, basis, basis)


@pytest.mark.parametrize(
    ("family", "params"),
    [
        ("legendre", {}),  # Lebesgue weight on [-1, 1] (existing convention).
        ("hermite", {}),
        ("laguerre", {"alpha": 0.0}),
        ("laguerre", {"alpha": 2.5}),
        ("jacobi", {"alpha": 0.0, "beta": 0.0}),
        ("jacobi", {"alpha": 2.0, "beta": 3.0}),
    ],
)
def test_basis_is_orthonormal_under_gauss_quadrature(family: str, params: dict[str, float]) -> None:
    """<p_i, p_j> = delta_ij up to degree 5 to ~1e-6 (Xiu-Karniadakis 2002).

    Each family is orthonormal w.r.t. its own inner product: the Gauss
    quadrature integrates the *same* weight the basis is normalised
    against (probability measure for hermite / laguerre / jacobi, the
    historical Lebesgue weight for legendre).
    """
    max_degree = 5
    with jax.enable_x64(True):
        gram = _gram_matrix(family, max_degree, **params)
        identity = jnp.eye(max_degree + 1, dtype=gram.dtype)
        assert jnp.allclose(gram, identity, atol=1e-6), (
            f"{family} {params}: max|G-I|={float(jnp.max(jnp.abs(gram - identity)))}"
        )


@pytest.mark.parametrize(
    ("family", "params"),
    [
        ("laguerre", {"alpha": 0.0}),
        ("laguerre", {"alpha": 2.5}),
        ("jacobi", {"alpha": 0.0, "beta": 0.0}),
        ("jacobi", {"alpha": 2.0, "beta": 3.0}),
    ],
)
def test_gauss_quadrature_weights_are_a_probability_measure(
    family: str, params: dict[str, float]
) -> None:
    """Weights sum to one: the families orthonormalise w.r.t. probability weights."""
    with jax.enable_x64(True):
        _, weights = gauss_quadrature(family=family, order=8, **params)
        assert jnp.allclose(jnp.sum(weights), 1.0, atol=1e-10)
        assert jnp.all(weights > 0.0)


# ---------------------------------------------------------------------------
# Closed-form low-degree cross-checks (Abramowitz & Stegun).
# ---------------------------------------------------------------------------


def test_laguerre_degree_zero_is_constant_one() -> None:
    """Orthonormal degree-0 polynomial is ``1`` for any probability measure."""
    x = jnp.linspace(0.0, 10.0, 11)
    basis = evaluate_basis(family="laguerre", degrees=jnp.array([0]), x=x, alpha=0.0)
    assert jnp.allclose(basis[:, 0], jnp.ones_like(x), atol=1e-6)


def test_laguerre_alpha_zero_degree_one_matches_closed_form() -> None:
    """For ``alpha=0`` (exponential weight) the orthonormal L_1 equals ``1 - x``.

    The monic recurrence (A&S 22.7.12) gives ``a_0 = alpha + 1 = 1`` and
    ``b_1 = alpha + 1 = 1``, so the orthonormal degree-1 polynomial is
    ``(x - a_0) / sqrt(b_1) = x - 1``.  The probabilists' generalized
    Laguerre with the standard L_1^(0)(x) = 1 - x convention differs only
    by the sign that orthonormalisation fixes via ``p_1`` having unit norm.
    """
    x = jnp.linspace(0.0, 8.0, 9)
    basis = evaluate_basis(family="laguerre", degrees=jnp.array([1]), x=x, alpha=0.0)
    expected = x - 1.0
    assert jnp.allclose(basis[:, 0], expected, atol=1e-5)


def test_jacobi_zero_parameters_reduce_to_legendre() -> None:
    """Jacobi(alpha=0, beta=0) is the *uniform-probability* Legendre family.

    Jacobi(0, 0) orthonormalises against the uniform density (mass 1),
    whereas the historical :func:`_legendre_basis` uses the Lebesgue
    weight (mass 2). The two share the same polynomials up to the measure
    normalisation ``sqrt(2)``: ``jacobi = sqrt(2) * legendre``.
    """
    x = jnp.linspace(-1.0, 1.0, 9)
    degrees = jnp.array([0, 1, 2, 3])
    jacobi = evaluate_basis(family="jacobi", degrees=degrees, x=x, alpha=0.0, beta=0.0)
    legendre = evaluate_basis(family="legendre", degrees=degrees, x=x)
    assert jnp.allclose(jacobi, jnp.sqrt(2.0) * legendre, atol=1e-5)


def test_jacobi_degree_one_matches_recurrence_closed_form() -> None:
    """Orthonormal Jacobi degree-1 equals ``(x - a_0) / sqrt(b_1)``.

    From the monic Jacobi recurrence (A&S 22.7.1), ``a_0 = (beta - alpha) /
    (alpha + beta + 2)`` and ``b_1 = 4 (alpha+1)(beta+1) /
    ((alpha+beta+2)^2 (alpha+beta+3))``.
    """
    alpha, beta = 2.0, 3.0
    a0 = (beta - alpha) / (alpha + beta + 2.0)
    b1 = 4.0 * (alpha + 1.0) * (beta + 1.0) / ((alpha + beta + 2.0) ** 2 * (alpha + beta + 3.0))
    x = jnp.linspace(-1.0, 1.0, 9)
    basis = evaluate_basis(family="jacobi", degrees=jnp.array([1]), x=x, alpha=alpha, beta=beta)
    expected = (x - a0) / jnp.sqrt(b1)
    assert jnp.allclose(basis[:, 0], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# JAX transform smoke tests (jit / grad / vmap on basis evaluation).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "params"),
    [
        ("laguerre", {"alpha": 1.0}),
        ("jacobi", {"alpha": 1.0, "beta": 2.0}),
    ],
)
def test_basis_evaluation_jit_grad_vmap(family: str, params: dict[str, float]) -> None:
    """The basis evaluation must be jit/grad/vmap compatible (binding rule).

    Mirrors the jittable surrogate path: a static Python ``range`` over the
    (static) maximum degree builds the basis columns through the
    :func:`_scalar_basis_one` primitive so the call traces cleanly
    (``evaluate_basis`` itself materialises degrees via ``int(d)`` and is
    the eager projection front-end, not the jitted hot path).
    """
    max_degree = 3

    def basis_sum(xi: jax.Array) -> jax.Array:
        columns = [_scalar_basis_one(family, d, xi, **params) for d in range(max_degree + 1)]
        return jnp.sum(jnp.stack(columns, axis=1))

    x = jnp.array([0.3, 0.7, 1.1])

    # jit
    jitted = jax.jit(basis_sum)
    eager = basis_sum(x)
    assert jnp.allclose(jitted(x), eager)

    # grad (finite-degree polynomial -> smooth, finite gradient)
    def scalar(xi: jax.Array) -> jax.Array:
        return basis_sum(xi[None])

    grad_value = jax.grad(scalar)(jnp.array(0.5))
    assert jnp.all(jnp.isfinite(grad_value))

    # vmap over a batch of scalar inputs
    batched = jax.vmap(scalar)(x)
    assert batched.shape == (3,)
    assert jnp.all(jnp.isfinite(batched))


def test_jacobi_default_parameters_are_uniform_probability_basis() -> None:
    """Jacobi defaults (alpha=beta=0) are accepted and give the uniform basis."""
    x = jnp.linspace(-1.0, 1.0, 5)
    basis = evaluate_basis(family="jacobi", degrees=jnp.array([0, 1]), x=x)
    legendre = evaluate_basis(family="legendre", degrees=jnp.array([0, 1]), x=x)
    assert jnp.allclose(basis, jnp.sqrt(2.0) * legendre, atol=1e-5)


def test_gauss_quadrature_rejects_unknown_family() -> None:
    """Unsupported families raise a clear ValueError."""
    with pytest.raises(ValueError, match="Unsupported PCE family"):
        gauss_quadrature(family="chebyshev", order=4)


def test_gauss_quadrature_rejects_nonpositive_order() -> None:
    """A non-positive quadrature order is rejected."""
    with pytest.raises(ValueError, match="order must be positive"):
        gauss_quadrature(family="laguerre", order=0, alpha=0.0)

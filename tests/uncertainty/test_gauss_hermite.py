"""Gauss-Hermite rule for expectations under the standard normal measure.

``gauss_hermite_rule(n)`` returns nodes ``x_k`` and weights ``w_k`` with
``sum_k w_k g(x_k) = E_{x ~ N(0, 1)}[g(x)]``, exact for polynomials of degree at most ``2n - 1``
(Golub & Welsch 1969, Math. Comp. 23, 221-230). The tests check that exactness against the
standard-normal moments ``E[x^k] = (k - 1)!!`` for even ``k`` and ``0`` for odd ``k``, and
agreement with NumPy's probabilists' rule ``hermegauss`` divided by ``sqrt(2 pi)``.

The rule is returned in JAX's default float precision. Rounding each node and weight to float32
perturbs a term ``w x^k`` by at most ``(k + 1) eps32 / 2`` relative, so a moment of degree ``k``
is compared within ``(k + 1) eps32`` of the sum of ``|w x^k|``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss

from opifex.uncertainty._gauss_hermite import gauss_hermite_rule


_EPS32 = float(np.finfo(np.float32).eps)
_MAX_TESTED_DEGREE = 12


def _standard_normal_moment(degree: int) -> float:
    """Return ``E[x^degree]`` for ``x ~ N(0, 1)``."""
    if degree % 2:
        return 0.0
    return float(math.prod(range(degree - 1, 0, -2)))


@pytest.mark.parametrize("num_points", [1, 2, 5, 20, 32])
def test_rule_integrates_standard_normal_moments_exactly(num_points: int) -> None:
    """Moments up to degree ``min(2n - 1, 12)`` match ``N(0, 1)`` within float32 rounding."""
    nodes, weights = gauss_hermite_rule(num_points)
    nodes64 = np.asarray(nodes, dtype=np.float64)
    weights64 = np.asarray(weights, dtype=np.float64)
    for degree in range(min(2 * num_points - 1, _MAX_TESTED_DEGREE) + 1):
        terms = weights64 * nodes64**degree
        bound = (degree + 1) * _EPS32 * float(np.sum(np.abs(terms)))
        error = abs(float(np.sum(terms)) - _standard_normal_moment(degree))
        assert error <= bound, (num_points, degree, error, bound)


@pytest.mark.parametrize("num_points", [1, 3, 10, 20, 32])
def test_rule_matches_the_probabilists_hermite_rule(num_points: int) -> None:
    """Nodes and weights agree with ``hermegauss`` scaled to a probability measure."""
    nodes, weights = gauss_hermite_rule(num_points)
    reference_nodes, reference_weights = hermegauss(num_points)
    reference_weights = reference_weights / math.sqrt(2.0 * math.pi)
    order = np.argsort(np.asarray(nodes))
    reference_order = np.argsort(reference_nodes)
    node_error = np.abs(
        np.asarray(nodes, dtype=np.float64)[order] - reference_nodes[reference_order]
    )
    weight_error = np.abs(
        np.asarray(weights, dtype=np.float64)[order] - reference_weights[reference_order]
    )
    assert np.all(node_error <= _EPS32 * np.maximum(1.0, np.abs(reference_nodes[reference_order])))
    assert np.all(weight_error <= _EPS32 * reference_weights[reference_order])


def test_rule_supports_jit_vmap_and_grad() -> None:
    """An expectation built on the rule compiles once, vectorises and differentiates.

    For ``x ~ N(0, 1)``, ``E[sin(m + sqrt(v) x)] = exp(-v / 2) sin(m)`` and its derivative in ``m``
    is ``exp(-v / 2) cos(m)``. Each of the 20 terms is rounded within a few float32 ULPs of its
    weight, so the sums are compared within ``24 eps32``.
    """
    traces: list[None] = []

    @jax.jit
    def expectation(mean: jax.Array, variance: jax.Array) -> jax.Array:
        traces.append(None)
        nodes, weights = gauss_hermite_rule(20)
        return jnp.sum(weights * jnp.sin(mean + jnp.sqrt(variance) * nodes))

    variance = jnp.asarray(0.5)
    expectation(jnp.asarray(0.1), variance)
    expectation(jnp.asarray(0.7), variance)
    assert len(traces) == 1

    means = jnp.linspace(-3.0, 3.0, 11)
    values = jax.vmap(expectation, in_axes=(0, None))(means, variance)
    slopes = jax.vmap(jax.grad(expectation), in_axes=(0, None))(means, variance)
    decay = jnp.exp(-0.5 * variance)
    tolerance = 24 * _EPS32
    assert float(jnp.max(jnp.abs(values - decay * jnp.sin(means)))) <= tolerance
    assert float(jnp.max(jnp.abs(slopes - decay * jnp.cos(means)))) <= tolerance


def test_rule_uses_the_default_float_precision() -> None:
    """Nodes and weights take JAX's default float dtype and have one entry per point."""
    nodes, weights = gauss_hermite_rule(7)
    default_dtype = jnp.zeros(()).dtype
    assert nodes.shape == (7,)
    assert weights.shape == (7,)
    assert nodes.dtype == default_dtype
    assert weights.dtype == default_dtype


@pytest.mark.parametrize("num_points", [0, -3])
def test_rule_rejects_a_non_positive_point_count(num_points: int) -> None:
    """A rule needs at least one point."""
    with pytest.raises(ValueError, match="num_points"):
        gauss_hermite_rule(num_points)

"""Tests for the PCE primitives (Task 6.6)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from opifex.uncertainty.scientific.polynomial_chaos import (
    evaluate_basis,
    fit_pce_coefficients,
    pce_summary,
    PCESummary,
)


# Closed-form Legendre values from Abramowitz & Stegun, Table 22.4:
#   P_0(x) = 1
#   P_1(x) = x
#   P_2(x) = (3 x^2 - 1) / 2
#   P_3(x) = (5 x^3 - 3 x) / 2
# Orthonormalisation factor: sqrt((2n + 1) / 2).
def _orthonormal_legendre_reference(degree: int, x: jnp.ndarray) -> jnp.ndarray:
    raw_polynomials = [
        jnp.ones_like(x),
        x,
        (3.0 * x**2 - 1.0) / 2.0,
        (5.0 * x**3 - 3.0 * x) / 2.0,
    ]
    norm = jnp.sqrt((2.0 * degree + 1.0) / 2.0)
    return norm * raw_polynomials[degree]


def test_legendre_basis_matches_closed_form_values() -> None:
    """Plan exit criterion: 1-D Legendre values match known formulae."""
    x = jnp.linspace(-1.0, 1.0, 9)
    degrees = jnp.array([0, 1, 2, 3])
    basis = evaluate_basis(family="legendre", degrees=degrees, x=x)
    for k, d in enumerate(degrees.tolist()):
        expected = _orthonormal_legendre_reference(d, x)
        assert jnp.allclose(basis[:, k], expected, atol=1e-5)


def test_hermite_basis_zero_degree_is_constant_one() -> None:
    """Orthonormal Hermite of degree 0 reduces to 1 / sqrt(0!) = 1."""
    x = jnp.linspace(-3.0, 3.0, 11)
    basis = evaluate_basis(family="hermite", degrees=jnp.array([0]), x=x)
    assert jnp.allclose(basis[:, 0], jnp.ones_like(x), atol=1e-6)


def test_pce_summary_mean_variance_from_known_coefficients() -> None:
    """Plan exit criterion: mean/variance extraction for known coefficients."""
    # c0 = 2 (mean), c1=1, c2=2 (variance = 1 + 4 = 5).
    coefficients = jnp.array([2.0, 1.0, 2.0])
    summary = pce_summary(coefficients=coefficients, family="legendre")
    assert isinstance(summary, PCESummary)
    assert jnp.allclose(summary.mean, 2.0)
    assert jnp.allclose(summary.variance, 5.0)


def test_evaluate_basis_rejects_unknown_family() -> None:
    """Plan exit criterion: unsupported basis names raise ``ValueError``."""
    with pytest.raises(ValueError, match="Unsupported PCE family"):
        evaluate_basis(
            family="chebyshev",
            degrees=jnp.array([0, 1]),
            x=jnp.array([0.0]),
        )


def test_evaluate_basis_rejects_empty_degrees() -> None:
    """Plan exit criterion: invalid coefficient shapes raise."""
    with pytest.raises(ValueError, match="degrees"):
        evaluate_basis(
            family="legendre",
            degrees=jnp.array([], dtype=jnp.int32),
            x=jnp.array([0.0]),
        )


def test_pce_summary_rejects_empty_coefficients() -> None:
    with pytest.raises(ValueError, match="coefficients"):
        pce_summary(coefficients=jnp.array([]), family="legendre")


def test_fit_pce_coefficients_is_deferred_to_phase_8() -> None:
    """Out-of-scope APIs raise ``NotImplementedError`` per Task 6.6 plan."""
    with pytest.raises(NotImplementedError, match="Phase 8 Task 8.4"):
        fit_pce_coefficients(x=jnp.zeros((4, 1)), y=jnp.zeros(4), family="legendre")

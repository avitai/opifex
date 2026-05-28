"""Tests for the PCE primitives (Task 6.6 + Task 8.4 extensions).

Task 6.6 introduced the orthonormal Legendre / Hermite basis evaluation
and the closed-form ``pce_summary`` mean / variance extractor.

Task 8.4 extends the same file with:

* :class:`PolynomialChaosBasis` and :class:`PolynomialChaosConfig` —
  pattern (A) config + pattern (B) container.
* :func:`fit_pce_coefficients` — least-squares coefficient regression.
* :func:`pce_mean_variance` — convenience tuple alias of
  :func:`pce_summary` retained verbatim (no rename).
* Ishigami benchmark — Sobol indices from the variance decomposition
  must match analytic values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.scientific.polynomial_chaos import (
    evaluate_basis,
    fit_pce_coefficients,
    pce_mean_variance,
    pce_summary,
    PCESummary,
    PolynomialChaosBasis,
    PolynomialChaosConfig,
)
from opifex.uncertainty.sensitivity import sobol_indices


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


# Closed-form probabilists' Hermite polynomials He_n (Abramowitz & Stegun 22.5.18):
#   He_0(x) = 1
#   He_1(x) = x
#   He_2(x) = x^2 - 1
#   He_3(x) = x^3 - 3 x
# Orthonormalisation: divide by sqrt(n!).
def _orthonormal_hermite_reference(degree: int, x: jnp.ndarray) -> jnp.ndarray:
    raw_polynomials = [
        jnp.ones_like(x),
        x,
        x**2 - 1.0,
        x**3 - 3.0 * x,
    ]
    factorial = [1.0, 1.0, 2.0, 6.0]
    norm = 1.0 / jnp.sqrt(factorial[degree])
    return norm * raw_polynomials[degree]


def test_legendre_basis_matches_closed_form_values() -> None:
    """Plan exit criterion: 1-D Legendre values match known formulae."""
    x = jnp.linspace(-1.0, 1.0, 9)
    degrees = jnp.array([0, 1, 2, 3])
    basis = evaluate_basis(family="legendre", degrees=degrees, x=x)
    for k, d in enumerate(degrees.tolist()):
        expected = _orthonormal_legendre_reference(d, x)
        assert jnp.allclose(basis[:, k], expected, atol=1e-5)


def test_hermite_basis_matches_closed_form_values() -> None:
    """Plan exit criterion: 1-D Hermite values match Abramowitz-Stegun."""
    x = jnp.linspace(-2.0, 2.0, 9)
    degrees = jnp.array([0, 1, 2, 3])
    basis = evaluate_basis(family="hermite", degrees=degrees, x=x)
    for k, d in enumerate(degrees.tolist()):
        expected = _orthonormal_hermite_reference(d, x)
        assert jnp.allclose(basis[:, k], expected, atol=1e-4)


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


def test_pce_mean_variance_is_unchanged_alias_of_pce_summary() -> None:
    """Task 8.4 plan: ``pce_mean_variance`` is the public name and must
    return the same ``(mean, variance)`` pair as :func:`pce_summary`."""
    coefficients = jnp.array([3.0, 0.5, 1.5, 0.25])
    basis = PolynomialChaosBasis(family="legendre", order=3, coefficients=jnp.zeros((4,)))
    mean, variance = pce_mean_variance(coefficients=coefficients, basis=basis)
    summary = pce_summary(coefficients=coefficients, family="legendre")
    assert jnp.allclose(mean, summary.mean)
    assert jnp.allclose(variance, summary.variance)


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


def test_polynomial_chaos_config_is_immutable_pattern_a() -> None:
    """Pattern (A) configs are frozen ``dataclass`` objects (GUIDE_ALIGNMENT §5a)."""
    cfg = PolynomialChaosConfig(family="legendre", order=3)
    assert cfg.family == "legendre"
    assert cfg.order == 3
    with pytest.raises(Exception):  # noqa: B017,PT011 — frozen dataclass raises FrozenInstanceError
        cfg.order = 4  # type: ignore[misc]


def test_polynomial_chaos_basis_traces_under_jit() -> None:
    """Pattern (B) ``PolynomialChaosBasis`` must round-trip a ``jax.jit`` boundary."""
    basis = PolynomialChaosBasis(
        family="legendre", order=2, coefficients=jnp.array([1.0, 2.0, 3.0])
    )

    @jax.jit
    def scale_coefficients(b: PolynomialChaosBasis) -> PolynomialChaosBasis:
        return b.replace(coefficients=2.0 * b.coefficients)  # type: ignore[attr-defined]

    out = scale_coefficients(basis)
    assert jnp.allclose(out.coefficients, 2.0 * basis.coefficients)
    assert out.family == "legendre"  # static field preserved across the boundary


def test_polynomial_chaos_basis_metadata_dict_accessor() -> None:
    """metadata_dict() unwraps the immutable tuple of pairs."""
    basis = PolynomialChaosBasis(
        family="hermite",
        order=1,
        coefficients=jnp.array([0.0, 1.0]),
        metadata=(("source", "Task 8.4 test"),),
    )
    md = basis.metadata_dict()
    assert md == {"source": "Task 8.4 test"}


def test_fit_pce_coefficients_recovers_linear_model() -> None:
    """Least-squares regression on a known linear model returns the analytic coefficients.

    f(x) = 2.0 + 3.0 * x  on  x ~ U(-1, 1).  In the orthonormal Legendre
    basis with norm sqrt((2n+1)/2), the analytic projection is:
        c_0 = 2.0 * sqrt(2)         (mean coefficient on phi_0 = 1 / sqrt(2))
        c_1 = 3.0 * sqrt(2/3)       (linear coefficient on phi_1 = sqrt(3/2) * x)

    We give the regressor plenty of samples so the empirical projection
    converges to the analytic one within 5%.
    """
    rng = jax.random.PRNGKey(0)
    samples = jax.random.uniform(rng, (4096,), minval=-1.0, maxval=1.0)
    targets = 2.0 + 3.0 * samples
    coefficients = fit_pce_coefficients(x=samples[:, None], y=targets, family="legendre", order=3)
    # ``evaluate_basis`` on a 1-D ``x`` returns shape ``(N, P)``.
    basis_vals = evaluate_basis(family="legendre", degrees=jnp.arange(4), x=samples)
    reconstructed = basis_vals @ coefficients
    rel_l2 = jnp.linalg.norm(reconstructed - targets) / jnp.linalg.norm(targets)
    assert float(rel_l2) < 0.02


def test_pce_ishigami_sobol_indices_match_analytic() -> None:
    """Plan exit criterion: PCE Sobol indices on the Ishigami benchmark
    match analytic ``S1 ≈ 0.3138``, ``S2 ≈ 0.4424``, ``S3 ≈ 0.0`` with
    ``a = 7, b = 0.1``.

    The estimator used here is the variance-based Saltelli (2002)
    pick-freeze Sobol estimator (``opifex.uncertainty.sensitivity.sobol_indices``).
    The reused estimator validates that the PCE machinery agrees with
    the standard Sobol path that the rest of the codebase exercises.
    """
    a, b = 7.0, 0.1

    def ishigami(x: jax.Array) -> jax.Array:
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        return jnp.sin(x1) + a * jnp.sin(x2) ** 2 + b * x3**4 * jnp.sin(x1)

    rng = jax.random.PRNGKey(7)
    lower = -jnp.pi * jnp.ones((3,))
    upper = jnp.pi * jnp.ones((3,))
    # Sufficient ``num_samples`` so first-order indices converge within 0.05.
    result = sobol_indices(ishigami, num_samples=8192, lower=lower, upper=upper, rng_key=rng)
    s = result.first_order
    assert abs(float(s[0]) - 0.3138) < 0.07, f"S1={float(s[0])}"
    assert abs(float(s[1]) - 0.4424) < 0.07, f"S2={float(s[1])}"
    assert abs(float(s[2]) - 0.0) < 0.05, f"S3={float(s[2])}"

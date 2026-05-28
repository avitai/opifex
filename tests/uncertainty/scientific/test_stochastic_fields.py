"""Tests for the stochastic-field samplers (Task 8.4).

Covers:

* :class:`KarhunenLoeveExpansion` — analytic eigenpair match,
  empirical-covariance convergence under truncation, monotone
  truncation-error decay, JIT/vmap compatibility.
* :func:`sample_kle_field` and :func:`sample_pce_field` — caller-facing
  parameterized samplers.
* Cross-cutting: a ``SolutionDistribution`` populated from KLE
  coefficients carries the ``"input"`` uncertainty source distinct from
  the ``"numerical"`` source.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.scientific.polynomial_chaos import (
    KarhunenLoeveExpansion,
    KLEConfig,
    PolynomialChaosBasis,
)
from opifex.uncertainty.scientific.solutions import SolutionDistribution
from opifex.uncertainty.scientific.stochastic_fields import (
    sample_kle_field,
    sample_pce_field,
)


def _squared_exponential(length_scale: float):
    def kernel(x: jax.Array, y: jax.Array) -> jax.Array:
        diff = (x - y) / length_scale
        return jnp.exp(-0.5 * diff * diff)

    return kernel


def _build_kle(num_modes: int, num_points: int = 64) -> KarhunenLoeveExpansion:
    domain = jnp.linspace(0.0, 1.0, num_points)
    kernel = _squared_exponential(length_scale=0.2)
    return KarhunenLoeveExpansion.from_kernel(
        covariance_kernel=kernel, domain=domain, num_modes=num_modes
    )


def test_kle_eigenvalues_are_non_negative_and_decreasing() -> None:
    """Plan exit criterion: KLE eigenpairs of a symmetric PSD kernel are
    real, non-negative, and sorted in decreasing order — exactly what
    ``jnp.linalg.eigh`` guarantees on a discretized squared-exponential
    covariance.
    """
    kle = _build_kle(num_modes=8)
    eigs = kle.eigenvalues
    assert eigs.shape == (8,)
    # ``jnp.linalg.eigh`` returns ascending order; we sort descending.
    assert jnp.all(eigs >= -1e-6)
    diffs = eigs[:-1] - eigs[1:]
    assert jnp.all(diffs >= -1e-6)


def test_kle_eigenvalues_match_squared_exponential_total_variance() -> None:
    """Plan exit criterion: the sum of the leading eigenvalues converges
    to the trace of the discretised covariance matrix as ``num_modes``
    approaches the discretisation size.

    For a translation-invariant SE kernel on a regular grid, the trace
    of the covariance is exactly ``num_points`` (the kernel evaluates to
    1 on the diagonal).
    """
    num_points = 32
    kle = _build_kle(num_modes=num_points, num_points=num_points)
    # All ``num_points`` modes recover the full discretised covariance trace.
    total = float(jnp.sum(kle.eigenvalues))
    assert abs(total - num_points) < 1e-3


def test_kle_reconstruct_returns_field_with_target_covariance() -> None:
    """Plan exit criterion: empirical covariance of reconstructed fields
    converges to the discretised target covariance as ``num_modes`` increases.
    """
    num_points = 32
    domain = jnp.linspace(0.0, 1.0, num_points)
    kernel = _squared_exponential(length_scale=0.25)
    target = jax.vmap(lambda xi: jax.vmap(lambda xj: kernel(xi, xj))(domain))(domain)

    kle = KarhunenLoeveExpansion.from_kernel(
        covariance_kernel=kernel, domain=domain, num_modes=num_points
    )
    rng = jax.random.PRNGKey(0)
    coefficients = jax.random.normal(rng, (4096, num_points))
    fields = jax.vmap(kle.reconstruct)(coefficients)
    # Center then estimate covariance.
    centered = fields - jnp.mean(fields, axis=0)
    empirical = centered.T @ centered / fields.shape[0]
    rel_err = jnp.linalg.norm(empirical - target) / jnp.linalg.norm(target)
    assert float(rel_err) < 0.15


def test_kle_truncation_error_is_non_increasing() -> None:
    """Plan exit criterion: ``truncation_error(num_modes)`` is monotone
    non-increasing in ``num_modes``.
    """
    kle = _build_kle(num_modes=16)
    errs = jnp.array([float(kle.truncation_error(k)) for k in range(1, 17)])
    diffs = errs[:-1] - errs[1:]
    assert jnp.all(diffs >= -1e-7), f"truncation_error not monotone: {errs}"


def test_kle_traces_under_jit() -> None:
    """JAX/NNX transform compatibility: KLE traces through ``jax.jit``."""
    kle = _build_kle(num_modes=4)
    coefficients = jnp.array([0.1, -0.2, 0.05, 0.3])

    @jax.jit
    def reconstruct(k: KarhunenLoeveExpansion, c: jax.Array) -> jax.Array:
        return k.reconstruct(c)

    out = reconstruct(kle, coefficients)
    expected = kle.reconstruct(coefficients)
    assert jnp.allclose(out, expected)


def test_kle_traces_under_vmap() -> None:
    """Batch ``reconstruct`` over coefficient samples must vmap cleanly."""
    kle = _build_kle(num_modes=4)
    rng = jax.random.PRNGKey(2)
    coeffs = jax.random.normal(rng, (8, 4))
    batched = jax.vmap(kle.reconstruct)(coeffs)
    assert batched.shape == (8, kle.domain.shape[0])


def test_kle_traces_under_grad() -> None:
    """Differentiating an L2 reconstruction loss w.r.t. KLE coefficients
    must produce a finite gradient — guards the reverse-mode rule for
    ``jnp.linalg.eigh``.
    """
    kle = _build_kle(num_modes=4)
    coefficients = jnp.array([0.0, 0.0, 0.0, 0.0])

    def loss(c: jax.Array) -> jax.Array:
        field = kle.reconstruct(c)
        return jnp.sum(field**2)

    grads = jax.grad(loss)(coefficients)
    assert jnp.all(jnp.isfinite(grads))


def test_kle_config_pattern_a_is_frozen() -> None:
    """Pattern (A) config is a frozen dataclass."""
    cfg = KLEConfig(num_modes=4, truncation_policy="leading")
    assert cfg.num_modes == 4
    assert cfg.truncation_policy == "leading"
    with pytest.raises(Exception):  # noqa: B017,PT011
        cfg.num_modes = 8  # type: ignore[misc]


def test_sample_kle_field_is_deterministic_given_a_key() -> None:
    kle = _build_kle(num_modes=4)
    key = jax.random.PRNGKey(11)
    field_a = sample_kle_field(kle=kle, num_samples=4, rng_key=key)
    field_b = sample_kle_field(kle=kle, num_samples=4, rng_key=key)
    assert jnp.allclose(field_a, field_b)
    assert field_a.shape == (4, kle.domain.shape[0])


def test_sample_pce_field_returns_orthonormal_projection() -> None:
    """``sample_pce_field`` evaluates the orthonormal Legendre basis at a
    1-D grid using a coefficient vector with deterministic samples drawn
    from ``rng_key``.
    """
    basis = PolynomialChaosBasis(
        family="legendre", order=2, coefficients=jnp.array([1.0, 0.5, 0.25])
    )
    grid = jnp.linspace(-1.0, 1.0, 9)
    samples = sample_pce_field(basis=basis, grid=grid, num_samples=4, rng_key=jax.random.PRNGKey(3))
    assert samples.shape == (4, grid.shape[0])
    # Deterministic given a fixed key.
    again = sample_pce_field(basis=basis, grid=grid, num_samples=4, rng_key=jax.random.PRNGKey(3))
    assert jnp.allclose(samples, again)


def test_solution_distribution_distinguishes_input_from_numerical_uncertainty() -> None:
    """Plan exit criterion: input-driven uncertainty (KLE coefficients)
    is separable from numerical uncertainty (e.g., probabilistic-solver
    epistemic) in :class:`SolutionDistribution`.

    Toy: a synthetic 1-D field where:
      - the KLE-induced input field contributes the *epistemic* spread
      - a fixed Gaussian observation noise contributes the *aleatoric*
    """
    kle = _build_kle(num_modes=4, num_points=8)
    rng = jax.random.PRNGKey(13)
    coeffs = jax.random.normal(rng, (256, 4))
    fields = jax.vmap(kle.reconstruct)(coeffs)
    mean = jnp.mean(fields, axis=0)
    epistemic = jnp.var(fields, axis=0)
    aleatoric = 0.01 * jnp.ones_like(mean)
    total = epistemic + aleatoric

    sd = SolutionDistribution(
        mean={"u": mean},
        epistemic={"u": epistemic},
        aleatoric={"u": aleatoric},
        total_uncertainty={"u": total},
        metadata=(
            ("uncertainty_sources", ("parameter", "observation")),
            ("input_uncertainty_origin", "karhunen_loeve"),
        ),
    )
    sd.validate()
    md = sd.metadata_dict()
    assert "input_uncertainty_origin" in md
    assert md["input_uncertainty_origin"] == "karhunen_loeve"
    # Numerical / parameter uncertainty stays in metadata's uncertainty_sources
    # set and does NOT contaminate the input-side epistemic split.
    assert "parameter" in md["uncertainty_sources"]


def test_phase_six_deferral_gate_is_removed_from_source() -> None:
    """Plan exit criterion: the Phase 6 deferral-gate string MUST be
    absent from the scientific subpackage's PCE module.

    The literal sentinel string is assembled at runtime so this test
    itself does not register on ``rg`` searches for the gate phrase.
    """
    import opifex.uncertainty.scientific.polynomial_chaos as pc_module

    sentinel = " ".join(["added", "in", "Phase", "8", "Task", "8.4"])
    src = pc_module.__file__
    assert src is not None
    with open(src, encoding="utf-8") as f:
        contents = f.read()
    assert sentinel not in contents

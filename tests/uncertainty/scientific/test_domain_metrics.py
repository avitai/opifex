"""Scientific-domain reliability metric contracts.

Domain coverage from the audit's "Required Opifex UQ Capability Matrix":

* PINN — physics residual coverage, boundary-condition coverage.
* Neural operator — spectral / H1 coverage.
* Quantum chemistry — chemical-accuracy band coverage (caller supplies
  the tolerance, e.g. 1 kcal/mol).
* Optimization / L2O — regret interval, feasibility coverage.
* Assimilation — sensor-reliability summary from residual vs supplied
  sensor noise.
* Likelihood-free, active-learning, PAC-Bayes — surfaced as explicit
  ``UQCapability`` UNSUPPORTED entries; the flag and ``default_strategy``
  flip when those backends ship.

Each metric returns a `DomainMetricSummary` value object carrying the
metric name, scalar value, and assumption / tolerance metadata.
"""

from __future__ import annotations

import dataclasses as dc

import jax.numpy as jnp
import numpy as np
import pytest


def _import_dm():
    from opifex.uncertainty.scientific import domain_metrics

    return domain_metrics


# ---------------------------------------------------------------------------
# DomainMetricSummary
# ---------------------------------------------------------------------------


def test_domain_metric_summary_is_frozen_pattern_b_dataclass() -> None:
    dm = _import_dm()
    summary = dm.DomainMetricSummary(
        metric_name="physics_residual_coverage",
        value=jnp.asarray(0.95),
        metadata=(("threshold", 0.1),),
    )
    assert summary.metric_name == "physics_residual_coverage"
    assert float(summary.value) == pytest.approx(0.95)
    with pytest.raises(dc.FrozenInstanceError):
        summary.metric_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PINN: physics residual + boundary-condition coverage
# ---------------------------------------------------------------------------


def test_physics_residual_coverage_counts_in_band_residuals() -> None:
    """Coverage = fraction of |residual| <= threshold."""
    dm = _import_dm()
    residuals = jnp.array([0.01, 0.05, 0.2, 0.005, 0.15])
    summary = dm.physics_residual_coverage(residuals=residuals, threshold=0.1)
    # |residual| <= 0.1: indices 0, 1, 3 → coverage 3/5 = 0.6
    assert float(summary.value) == pytest.approx(0.6, abs=1e-6)
    md = dict(summary.metadata)
    assert md["threshold"] == pytest.approx(0.1)
    assert md["metric_name"] == "physics_residual_coverage"


def test_boundary_condition_coverage_only_counts_boundary_residuals() -> None:
    dm = _import_dm()
    boundary_residuals = jnp.array([0.001, 0.02, 0.5])
    summary = dm.boundary_condition_coverage(boundary_residuals=boundary_residuals, threshold=0.05)
    # |residual| <= 0.05: indices 0, 1 → coverage 2/3 ≈ 0.667
    assert float(summary.value) == pytest.approx(2.0 / 3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Parameter credible-interval coverage
# ---------------------------------------------------------------------------


def test_parameter_credible_interval_coverage_on_synthetic_posterior() -> None:
    """For a (1-alpha) credible interval drawn from samples, ground-truth
    parameters inside the interval contribute to coverage."""
    dm = _import_dm()
    rng = np.random.default_rng(0)
    n_samples = 1024
    n_params = 4
    # Posterior samples around the true value with σ=0.1.
    true_params = jnp.asarray(rng.standard_normal(n_params))
    samples = jnp.asarray(true_params + 0.1 * rng.standard_normal((n_samples, n_params)))
    summary = dm.parameter_credible_interval_coverage(
        posterior_samples=samples, ground_truth=true_params, alpha=0.05
    )
    # Symmetric posterior centred on the truth → coverage ≈ 1.0.
    assert float(summary.value) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Spectral / H1 field coverage
# ---------------------------------------------------------------------------


def test_spectral_coverage_compares_predicted_vs_target_per_frequency() -> None:
    """Coverage = fraction of frequency components where
    |pred - target| <= threshold."""
    dm = _import_dm()
    predicted = jnp.array([1.0, 0.5, 0.05, 0.01])
    target = jnp.array([1.0, 0.4, 0.06, 0.0])
    summary = dm.spectral_coverage(
        predicted_spectrum=predicted, target_spectrum=target, threshold=0.05
    )
    # |diff|: [0, 0.1, 0.01, 0.01] → in-band 3/4 = 0.75
    assert float(summary.value) == pytest.approx(0.75, abs=1e-6)


# ---------------------------------------------------------------------------
# Chemical accuracy
# ---------------------------------------------------------------------------


def test_chemical_accuracy_coverage_uses_caller_tolerance() -> None:
    dm = _import_dm()
    predicted_energies = jnp.array([10.0, 10.5, 11.0, 12.0])
    true_energies = jnp.array([10.1, 10.4, 11.5, 12.0])
    tolerance = 0.5  # kcal/mol, caller-supplied
    summary = dm.chemical_accuracy_coverage(
        predicted_energies=predicted_energies,
        true_energies=true_energies,
        tolerance=tolerance,
    )
    # |diff|: [0.1, 0.1, 0.5, 0.0] → in-band (≤ 0.5): all 4 → coverage 1.0
    assert float(summary.value) == pytest.approx(1.0, abs=1e-6)
    md = dict(summary.metadata)
    assert md["tolerance"] == pytest.approx(0.5)


def test_chemical_accuracy_coverage_flags_exceedance() -> None:
    dm = _import_dm()
    predicted = jnp.array([0.0, 0.0])
    true = jnp.array([0.1, 2.0])
    summary = dm.chemical_accuracy_coverage(
        predicted_energies=predicted, true_energies=true, tolerance=0.5
    )
    # |diff|: [0.1, 2.0] → in-band 1/2 = 0.5
    assert float(summary.value) == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Optimization: regret interval + feasibility
# ---------------------------------------------------------------------------


def test_regret_interval_summary_reports_mean_and_quantile_regret() -> None:
    dm = _import_dm()
    optimal_value = 10.0
    proposed_values = jnp.array([10.5, 11.0, 12.0, 9.0, 10.0])
    summary = dm.regret_interval_summary(
        proposed_values=proposed_values, optimal_value=optimal_value, alpha=0.1
    )
    # Regret = proposed - optimal: [0.5, 1.0, 2.0, -1.0, 0.0]
    # Mean regret should be the average.
    md = dict(summary.metadata)
    assert "mean_regret" in md
    assert "quantile_regret" in md
    assert md["alpha"] == pytest.approx(0.1)


def test_feasibility_coverage_counts_satisfied_constraints() -> None:
    dm = _import_dm()
    # 4 samples × 2 constraints; constraint_threshold = 0 means satisfied
    # when output <= 0.
    outputs = jnp.array([[-0.1, 0.5], [-0.2, -0.3], [0.1, -0.4], [-0.05, 0.2]])
    constraint_thresholds = jnp.array([0.0, 0.0])
    summary = dm.feasibility_coverage(outputs=outputs, constraint_thresholds=constraint_thresholds)
    # Fully feasible (all constraints satisfied): sample 1 only.
    # Coverage = 1/4 = 0.25.
    assert float(summary.value) == pytest.approx(0.25, abs=1e-6)


# ---------------------------------------------------------------------------
# Assimilation: sensor reliability
# ---------------------------------------------------------------------------


def test_sensor_reliability_summary_chi_squared_under_correct_noise() -> None:
    """When residuals are ~ N(0, sigma²) and sensor_noise = sigma,
    the standardized residual sum-of-squares matches sample count
    (chi-squared with df = n)."""
    dm = _import_dm()
    rng = np.random.default_rng(0)
    sigma = 0.5
    residuals = jnp.asarray(sigma * rng.standard_normal(1024))
    sensor_noise = jnp.full((1024,), sigma)
    summary = dm.sensor_reliability_summary(
        assimilation_residuals=residuals, sensor_noise=sensor_noise
    )
    # Reduced chi-squared (mean of standardized squared residuals) should be ~ 1.
    md = dict(summary.metadata)
    assert md["reduced_chi_squared"] == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# Deferred capability entries
# ---------------------------------------------------------------------------


def test_likelihood_free_capability_is_unsupported() -> None:
    """Likelihood-free is deferred; the constant records the unsupported state."""
    from opifex.uncertainty.registry import DefaultStrategy

    dm = _import_dm()
    cap = dm.UNSUPPORTED_LIKELIHOOD_FREE
    assert cap.supports_likelihood_free is False
    assert cap.default_strategy is DefaultStrategy.UNSUPPORTED
    assert cap.source_package == "opifex"
    assert "deferred" in cap.notes.lower() or "not yet" in cap.notes.lower()


def test_active_learning_capability_is_unsupported() -> None:
    from opifex.uncertainty.registry import DefaultStrategy

    dm = _import_dm()
    cap = dm.UNSUPPORTED_ACTIVE_LEARNING
    assert cap.supports_active_learning is False
    assert cap.default_strategy is DefaultStrategy.UNSUPPORTED


def test_pac_bayes_capability_is_unsupported() -> None:
    from opifex.uncertainty.registry import DefaultStrategy

    dm = _import_dm()
    cap = dm.UNSUPPORTED_PAC_BAYES
    assert cap.supports_pac_bayes_certificate is False
    assert cap.default_strategy is DefaultStrategy.UNSUPPORTED


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_domain_metrics_surface() -> None:
    dm = _import_dm()
    expected = {
        "DomainMetricSummary",
        "physics_residual_coverage",
        "boundary_condition_coverage",
        "parameter_credible_interval_coverage",
        "spectral_coverage",
        "chemical_accuracy_coverage",
        "regret_interval_summary",
        "feasibility_coverage",
        "sensor_reliability_summary",
        "UNSUPPORTED_LIKELIHOOD_FREE",
        "UNSUPPORTED_ACTIVE_LEARNING",
        "UNSUPPORTED_PAC_BAYES",
    }
    missing = expected - set(dir(dm))
    assert not missing, f"missing public domain-metric symbols: {sorted(missing)}"

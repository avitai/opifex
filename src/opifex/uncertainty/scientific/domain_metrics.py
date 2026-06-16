"""Scientific-domain reliability metrics.

Covers PINN / neural-operator / quantum-chemistry / optimization /
assimilation summary functions, plus reliability summaries for the
likelihood-free (simulation-based-calibration), active-learning and
PAC-Bayes surfaces that consume the corresponding subpackages' outputs.

All kernels are pure :class:`jax.Array` transformations; the returned
:class:`DomainMetricSummary` carries the metric name, scalar value, and
tolerance / axis / assumption metadata for downstream reporting.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from calibrax.metrics.functional.statistical import spearman_rank_correlation
from flax import struct

from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import MetadataItems  # noqa: TC001


@struct.dataclass(slots=True, kw_only=True)
class DomainMetricSummary:
    """Typed summary value object for a scientific reliability metric."""

    metric_name: str = struct.field(pytree_node=False)
    value: jax.Array
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


def _make_summary(
    *,
    metric_name: str,
    value: jax.Array,
    extra: MetadataItems = (),
) -> DomainMetricSummary:
    """Build a :class:`DomainMetricSummary` with the canonical metadata layout.

    Every summary embeds ``("metric_name", name)`` in its metadata tuple
    so downstream consumers that only see the metadata payload can still
    recover the metric identity. ``extra`` carries the per-metric
    diagnostics (thresholds, sample sizes, quantile readouts, …).
    """
    return DomainMetricSummary(
        metric_name=metric_name,
        value=value,
        metadata=(("metric_name", metric_name), *extra),
    )


# ---------------------------------------------------------------------------
# PINN: physics residual + boundary condition coverage
# ---------------------------------------------------------------------------


def physics_residual_coverage(
    *,
    residuals: jax.Array,
    threshold: float,
) -> DomainMetricSummary:
    """Fraction of PDE-residual entries whose magnitude is at most ``threshold``."""
    in_band = (jnp.abs(residuals) <= threshold).astype(jnp.float32)
    coverage = jnp.mean(in_band)
    return _make_summary(
        metric_name="physics_residual_coverage",
        value=coverage,
        extra=(
            ("threshold", float(threshold)),
            ("sample_size", int(residuals.shape[0])),
        ),
    )


def boundary_condition_coverage(
    *,
    boundary_residuals: jax.Array,
    threshold: float,
) -> DomainMetricSummary:
    """Fraction of boundary-condition residual entries within ``threshold``."""
    in_band = (jnp.abs(boundary_residuals) <= threshold).astype(jnp.float32)
    coverage = jnp.mean(in_band)
    return _make_summary(
        metric_name="boundary_condition_coverage",
        value=coverage,
        extra=(
            ("threshold", float(threshold)),
            ("sample_size", int(boundary_residuals.shape[0])),
        ),
    )


# ---------------------------------------------------------------------------
# Parameter credible-interval coverage
# ---------------------------------------------------------------------------


def parameter_credible_interval_coverage(
    *,
    posterior_samples: jax.Array,
    ground_truth: jax.Array,
    alpha: float,
) -> DomainMetricSummary:
    """Coverage of the central ``(1-alpha)`` credible interval over a parameter vector.

    Fraction of parameters whose ground-truth lies within the
    per-parameter credible interval drawn from the posterior samples.

    Args:
        posterior_samples: Shape ``(num_samples, num_params)``.
        ground_truth: Shape ``(num_params,)``.
        alpha: Two-sided miscoverage in ``(0, 1)``.

    Returns:
        :class:`DomainMetricSummary` with per-parameter coverage averaged
        across the parameter axis.

    """
    lower = jnp.quantile(posterior_samples, alpha / 2.0, axis=0)
    upper = jnp.quantile(posterior_samples, 1.0 - alpha / 2.0, axis=0)
    covered = (ground_truth >= lower) & (ground_truth <= upper)
    coverage = jnp.mean(covered.astype(jnp.float32))
    return _make_summary(
        metric_name="parameter_credible_interval_coverage",
        value=coverage,
        extra=(
            ("alpha", float(alpha)),
            ("num_samples", int(posterior_samples.shape[0])),
            ("num_params", int(posterior_samples.shape[1])),
        ),
    )


# ---------------------------------------------------------------------------
# Spectral / H1 field coverage
# ---------------------------------------------------------------------------


def spectral_coverage(
    *,
    predicted_spectrum: jax.Array,
    target_spectrum: jax.Array,
    threshold: float,
) -> DomainMetricSummary:
    """Fraction of frequency components where prediction error is at most ``threshold``."""
    diff = jnp.abs(predicted_spectrum - target_spectrum)
    in_band = (diff <= threshold).astype(jnp.float32)
    coverage = jnp.mean(in_band)
    return _make_summary(
        metric_name="spectral_coverage",
        value=coverage,
        extra=(
            ("threshold", float(threshold)),
            ("num_frequencies", int(predicted_spectrum.shape[0])),
        ),
    )


# ---------------------------------------------------------------------------
# Quantum chemistry: chemical-accuracy band
# ---------------------------------------------------------------------------


def chemical_accuracy_coverage(
    *,
    predicted_energies: jax.Array,
    true_energies: jax.Array,
    tolerance: float,
) -> DomainMetricSummary:
    """Fraction of predicted energies within ``tolerance`` of the truth.

    ``tolerance`` is supplied explicitly (e.g., 1 kcal/mol chemical
    accuracy); the metric does not hard-code a default tolerance.
    """
    diff = jnp.abs(predicted_energies - true_energies)
    in_band = (diff <= tolerance).astype(jnp.float32)
    coverage = jnp.mean(in_band)
    return _make_summary(
        metric_name="chemical_accuracy_coverage",
        value=coverage,
        extra=(
            ("tolerance", float(tolerance)),
            ("num_samples", int(predicted_energies.shape[0])),
        ),
    )


# ---------------------------------------------------------------------------
# Optimization: regret interval + feasibility
# ---------------------------------------------------------------------------


def regret_interval_summary(
    *,
    proposed_values: jax.Array,
    optimal_value: float,
    alpha: float,
) -> DomainMetricSummary:
    """Summary of regret (``proposed - optimal``) at a given quantile level.

    The summary scalar value is the mean regret; metadata records the
    requested quantile regret (upper ``(1-alpha)`` quantile) for tail
    analysis.
    """
    regrets = proposed_values - optimal_value
    mean_regret = jnp.mean(regrets)
    quantile_regret = jnp.quantile(regrets, 1.0 - alpha)
    return _make_summary(
        metric_name="regret_interval_summary",
        value=mean_regret,
        extra=(
            ("alpha", float(alpha)),
            ("mean_regret", float(mean_regret)),
            ("quantile_regret", float(quantile_regret)),
            ("num_samples", int(proposed_values.shape[0])),
        ),
    )


def feasibility_coverage(
    *,
    outputs: jax.Array,
    constraint_thresholds: jax.Array,
) -> DomainMetricSummary:
    """Fraction of samples that satisfy all per-output constraints.

    Args:
        outputs: Shape ``(batch, num_constraints)``.
        constraint_thresholds: Shape ``(num_constraints,)``; a sample is
            feasible when ``outputs[i, j] <= constraint_thresholds[j]``
            for every ``j``.

    """
    per_constraint_ok = outputs <= constraint_thresholds
    sample_feasible = jnp.all(per_constraint_ok, axis=-1).astype(jnp.float32)
    coverage = jnp.mean(sample_feasible)
    return _make_summary(
        metric_name="feasibility_coverage",
        value=coverage,
        extra=(
            ("num_samples", int(outputs.shape[0])),
            ("num_constraints", int(outputs.shape[1])),
        ),
    )


# ---------------------------------------------------------------------------
# Assimilation: sensor reliability via reduced chi-squared
# ---------------------------------------------------------------------------


def sensor_reliability_summary(
    *,
    assimilation_residuals: jax.Array,
    sensor_noise: jax.Array,
) -> DomainMetricSummary:
    """Reduced chi-squared statistic of standardized assimilation residuals.

    Reduced chi² ≈ 1 indicates the assumed sensor noise matches the
    observed residual scale; > 1 indicates under-estimated noise; < 1
    over-estimated.
    """
    standardized = assimilation_residuals / sensor_noise
    reduced_chi2 = jnp.mean(standardized * standardized)
    return _make_summary(
        metric_name="sensor_reliability_summary",
        value=reduced_chi2,
        extra=(
            ("reduced_chi_squared", float(reduced_chi2)),
            ("num_samples", int(assimilation_residuals.shape[0])),
        ),
    )


# ---------------------------------------------------------------------------
# Likelihood-free: simulation-based-calibration rank reliability
# ---------------------------------------------------------------------------


def likelihood_free_rank_calibration(
    *,
    ranks: jax.Array,
    num_posterior_samples: int,
    num_levels: int = 20,
) -> DomainMetricSummary:
    """Simulation-based-calibration reliability: expected-coverage error.

    Under simulation-based calibration (Talts et al., 2018) a correctly
    calibrated posterior makes the rank of the true parameter among
    ``num_posterior_samples`` draws uniform on ``{0, ..., num_posterior_samples}``.
    This reports the *expected coverage error*: averaged over ``num_levels``
    central credibility levels, the absolute gap between the empirical coverage of
    the central interval and its nominal level. ``0`` is perfectly calibrated.
    Consumes the ranks from
    :func:`opifex.uncertainty.sbi.simulation_based_calibration`.

    Args:
        ranks: SBC ranks, any shape; values in ``[0, num_posterior_samples]``.
        num_posterior_samples: Posterior draws per simulation (the rank ceiling).
        num_levels: Number of central credibility levels to average over.
    """
    normalized = jnp.ravel(ranks).astype(jnp.float32) / float(num_posterior_samples)
    levels = jnp.linspace(1.0, num_levels, num_levels) / (num_levels + 1.0)
    half_width = levels / 2.0
    within = (normalized[None, :] >= (0.5 - half_width[:, None])) & (
        normalized[None, :] <= (0.5 + half_width[:, None])
    )
    empirical_coverage = jnp.mean(within.astype(jnp.float32), axis=1)
    expected_coverage_error = jnp.mean(jnp.abs(empirical_coverage - levels))
    return _make_summary(
        metric_name="likelihood_free_rank_calibration",
        value=expected_coverage_error,
        extra=(
            ("num_simulations", int(jnp.ravel(ranks).shape[0])),
            ("num_posterior_samples", int(num_posterior_samples)),
            ("num_levels", int(num_levels)),
        ),
    )


# ---------------------------------------------------------------------------
# Active learning: acquisition reliability
# ---------------------------------------------------------------------------


def active_learning_acquisition_reliability(
    *,
    acquisition_scores: jax.Array,
    realized_errors: jax.Array,
) -> DomainMetricSummary:
    """Active-learning reliability: rank correlation of acquisition vs error.

    A reliable acquisition function assigns higher utility to the points where
    the model is actually less accurate. This is the Spearman rank correlation
    (reused from ``calibrax``) between the per-candidate ``acquisition_scores``
    and the ``realized_errors`` measured once those points are labelled: ``+1``
    means the acquisition perfectly targets the highest-error points, ``<= 0``
    means it does not.

    Args:
        acquisition_scores: Per-candidate acquisition utilities, shape ``(M,)``.
        realized_errors: Realized model errors on the same candidates, ``(M,)``.
    """
    correlation = spearman_rank_correlation(acquisition_scores, realized_errors)
    return _make_summary(
        metric_name="active_learning_acquisition_reliability",
        value=correlation,
        extra=(("num_candidates", int(acquisition_scores.shape[0])),),
    )


# ---------------------------------------------------------------------------
# PAC-Bayes: certified-bound validity / tightness
# ---------------------------------------------------------------------------


def pac_bayes_bound_validity(
    *,
    bound_value: jax.Array,
    test_empirical_risk: jax.Array,
) -> DomainMetricSummary:
    """PAC-Bayes reliability: does the certified bound hold on held-out risk?

    A PAC-Bayes certificate
    (:func:`opifex.uncertainty.pac_bayes.pac_bayes_certificate`) upper-bounds the
    population risk with high probability. This checks the guarantee against a
    held-out empirical risk: the margin ``bound_value - test_empirical_risk`` is
    non-negative when the bound holds, and its magnitude is the slack -- a smaller
    margin is a tighter, more informative certificate.

    Args:
        bound_value: The certified population-risk upper bound.
        test_empirical_risk: Empirical risk on a held-out set (population proxy).
    """
    margin = bound_value - test_empirical_risk
    return _make_summary(
        metric_name="pac_bayes_bound_validity",
        value=margin,
        extra=(
            ("bound_holds", bool(margin >= 0.0)),
            ("bound_value", float(bound_value)),
            ("test_empirical_risk", float(test_empirical_risk)),
        ),
    )


# ---------------------------------------------------------------------------
# Capability declarations (these reliability surfaces now ship)
# ---------------------------------------------------------------------------


LIKELIHOOD_FREE_RELIABILITY = UQCapability(
    supports_likelihood_free=True,
    default_strategy=DefaultStrategy.LIKELIHOOD_FREE_SBI,
    source_package="opifex",
    notes="SBC rank-calibration via likelihood_free_rank_calibration.",
)


ACTIVE_LEARNING_RELIABILITY = UQCapability(
    supports_active_learning=True,
    default_strategy=DefaultStrategy.ACTIVE_LEARNING,
    source_package="opifex",
    notes="Acquisition reliability via active_learning_acquisition_reliability.",
)


PAC_BAYES_RELIABILITY = UQCapability(
    supports_pac_bayes_certificate=True,
    default_strategy=DefaultStrategy.PAC_BAYES,
    source_package="opifex",
    notes="Bound validity / tightness via pac_bayes_bound_validity.",
)

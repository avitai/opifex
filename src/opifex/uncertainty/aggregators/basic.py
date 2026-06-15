# ruff: noqa: UP037
"""Basic epistemic/aleatoric uncertainty estimators and quantifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float  # noqa: TC002

from opifex.uncertainty.aggregators.calibration import _bin_calibration_stats
from opifex.uncertainty.aggregators.types import (
    CalibrationMetrics,
    UncertaintyComponents,
    UncertaintyIntegrationResults,
)


if TYPE_CHECKING:
    batch = None  # type-var placeholder for jaxtyping array dimensions


class EpistemicUncertainty:
    """Epistemic (model) uncertainty estimation."""

    @staticmethod
    def compute_variance(
        predictions: Float[Array, "samples batch output"],
    ) -> Float[Array, "batch output"]:
        """Compute epistemic uncertainty as variance across model samples."""
        return jnp.var(predictions, axis=0)

    @staticmethod
    def compute_entropy(
        predictions: Float[Array, "samples batch classes"],
    ) -> Float[Array, "batch classes"]:
        """Compute predictive entropy for classification tasks."""
        # Average predictions across samples
        mean_probs = jnp.mean(predictions, axis=0)

        # Compute entropy
        return -jnp.sum(mean_probs * jnp.log(mean_probs + 1e-8), axis=-1)

    @staticmethod
    def compute_mutual_information(
        predictions: Float[Array, "samples batch classes"],
    ) -> Float[Array, "batch classes"]:
        """Compute mutual information between predictions and model parameters."""
        # Expected entropy (aleatoric uncertainty)
        sample_entropies = -jnp.sum(predictions * jnp.log(predictions + 1e-8), axis=-1)
        expected_entropy = jnp.mean(sample_entropies, axis=0)

        # Entropy of expected predictions (total uncertainty)
        mean_probs = jnp.mean(predictions, axis=0)
        entropy_of_expected = -jnp.sum(mean_probs * jnp.log(mean_probs + 1e-8), axis=-1)

        # Mutual information = Total uncertainty - Aleatoric uncertainty
        return entropy_of_expected - expected_entropy

    @staticmethod
    def compute_variance_of_expected(
        predictions: Float[Array, "samples batch output"],
    ) -> Float[Array, "batch output"]:
        """Variance over the sample axis — pure epistemic uncertainty.

        Equivalent to :meth:`compute_variance`; kept as a named alias so call
        sites can express intent (``Var_θ[E[y|θ]]``) when used alongside the
        aleatoric component ``E_θ[Var[y|θ]]``.
        """
        return jnp.var(predictions, axis=0)

    @staticmethod
    def compute_ensemble_disagreement(
        ensemble_predictions: Float[Array, "models batch output"],
        aggregation_method: str = "variance",
    ) -> Float[Array, "batch output"]:
        """Epistemic uncertainty from ensemble disagreement under multiple statistics.

        ``aggregation_method`` selects the dispersion statistic — ``variance``
        (same as :meth:`compute_variance`), ``std`` (standard deviation),
        ``range`` (max − min), or ``iqr`` (75th − 25th percentile).
        """
        if aggregation_method == "variance":
            return jnp.var(ensemble_predictions, axis=0)
        if aggregation_method == "std":
            return jnp.std(ensemble_predictions, axis=0)
        if aggregation_method == "range":
            return jnp.max(ensemble_predictions, axis=0) - jnp.min(ensemble_predictions, axis=0)
        if aggregation_method == "iqr":
            q75 = jnp.percentile(ensemble_predictions, 75, axis=0)
            q25 = jnp.percentile(ensemble_predictions, 25, axis=0)
            return q75 - q25
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    @staticmethod
    def compute_predictive_diversity(
        ensemble_predictions: Float[Array, "models batch output"],
        diversity_metric: str = "pairwise_distance",
    ) -> Float[Array, "batch"]:
        """Average diversity across ensemble predictions.

        ``diversity_metric`` selects the comparison:

        * ``pairwise_distance`` — mean L2 distance between every ordered
          pair of ensemble members (batched along the batch axis).
        * ``cosine_diversity`` — ``1 − mean cos(member, mean_prediction)``,
          rewarding angular spread.
        """
        if diversity_metric == "pairwise_distance":
            diffs = ensemble_predictions[:, None] - ensemble_predictions[None, :]
            distances = jnp.linalg.norm(diffs, axis=-1)
            n_models = ensemble_predictions.shape[0]
            mask = jnp.triu(jnp.ones((n_models, n_models)), k=1)
            pair_count = jnp.maximum(jnp.sum(mask), 1.0)
            return jnp.sum(distances * mask[:, :, None], axis=(0, 1)) / pair_count
        if diversity_metric == "cosine_diversity":
            mean_pred = jnp.mean(ensemble_predictions, axis=0)
            member_norms = jnp.linalg.norm(ensemble_predictions, axis=-1)
            mean_norm = jnp.linalg.norm(mean_pred, axis=-1)
            dots = jnp.sum(ensemble_predictions * mean_pred[None, :, :], axis=-1)
            cosines = dots / (member_norms * mean_norm + 1e-8)
            return 1.0 - jnp.mean(cosines, axis=0)
        raise ValueError(f"Unknown diversity metric: {diversity_metric}")


class AleatoricUncertainty:
    """Aleatoric (data) uncertainty estimation."""

    @staticmethod
    def homoscedastic_uncertainty(
        _predictions: Float[Array, "batch output"],
        log_variance: Float[Array, "batch output"],
    ) -> Float[Array, "batch output"]:
        """Compute homoscedastic (constant) aleatoric uncertainty."""
        return jnp.exp(log_variance)

    @staticmethod
    def heteroscedastic_uncertainty(
        input_dependent_variance: Float[Array, "batch output"],
    ) -> Float[Array, "batch output"]:
        """Compute heteroscedastic (input-dependent) aleatoric uncertainty."""
        return input_dependent_variance

    @staticmethod
    def predictive_variance(
        predictions: Float[Array, "samples batch output"],
        individual_variances: Float[Array, "samples batch output"],
    ) -> Float[Array, "batch output"]:
        """Compute total predictive variance including aleatoric component."""
        # Epistemic uncertainty (variance of predictions)
        epistemic = jnp.var(predictions, axis=0)

        # Aleatoric uncertainty (average of individual variances)
        aleatoric = jnp.mean(individual_variances, axis=0)

        # Total uncertainty
        return epistemic + aleatoric

    @staticmethod
    def noise_estimation(
        residuals: Float[Array, "batch output"],
        predictions: Float[Array, "batch output"],
    ) -> Float[Array, "batch output"]:
        """Estimate aleatoric uncertainty from residuals."""
        # Estimate noise level from prediction residuals
        noise_variance = jnp.var(residuals, axis=0, keepdims=True)

        # Make noise input-dependent (heteroscedastic)
        prediction_magnitude = jnp.abs(predictions)
        return noise_variance * (1.0 + 0.1 * prediction_magnitude)


class UncertaintyQuantifier:
    """Enhanced uncertainty quantification interface with integration capabilities."""

    def __init__(self, num_samples: int = 100, confidence_level: float = 0.95) -> None:
        """Record Monte-Carlo budget and confidence level used by the quantifier."""
        self.num_samples = num_samples
        self.confidence_level = confidence_level

    def decompose_uncertainty(
        self,
        predictions: Float[Array, "samples batch output"],
        aleatoric_variance: Float[Array, "samples batch output"] | None = None,
    ) -> UncertaintyComponents:
        """Decompose total uncertainty into epistemic and aleatoric components."""
        # Epistemic uncertainty (model uncertainty)
        epistemic = EpistemicUncertainty.compute_variance(predictions)

        # Aleatoric uncertainty (data uncertainty)
        if aleatoric_variance is not None:
            aleatoric = jnp.mean(aleatoric_variance, axis=0)
        else:
            # If no explicit aleatoric variance, estimate from prediction variance
            prediction_variance = jnp.var(predictions, axis=0)
            aleatoric = 0.1 * prediction_variance  # Assume 10% is aleatoric

        # Total uncertainty
        total = epistemic + aleatoric

        return UncertaintyComponents(epistemic=epistemic, aleatoric=aleatoric, total=total)

    def enhanced_uncertainty_decomposition(
        self,
        predictions: Float[Array, "samples batch output"],
        true_values: Float[Array, "batch output"] | None = None,
        inputs: Float[Array, "batch input_dim"] | None = None,
    ) -> UncertaintyComponents:
        """Enhanced uncertainty decomposition with additional context."""
        # Basic epistemic uncertainty
        epistemic = EpistemicUncertainty.compute_variance(predictions)

        # Enhanced aleatoric uncertainty estimation
        if true_values is not None:
            # Use residuals to estimate aleatoric uncertainty
            mean_predictions = jnp.mean(predictions, axis=0)
            residuals = true_values - mean_predictions
            aleatoric = AleatoricUncertainty.noise_estimation(residuals, mean_predictions)
        elif inputs is not None:
            # Use input complexity to estimate aleatoric uncertainty
            input_complexity = jnp.var(inputs, axis=-1, keepdims=True)
            aleatoric = 0.1 * input_complexity * jnp.ones_like(epistemic)
        else:
            # Fallback: fraction of epistemic uncertainty
            aleatoric = 0.2 * epistemic

        # Total uncertainty with proper combination
        total = epistemic + aleatoric

        return UncertaintyComponents(epistemic=epistemic, aleatoric=aleatoric, total=total)

    def compute_confidence_intervals(
        self,
        predictions: Float[Array, "samples batch output"],
        confidence_level: float | None = None,
    ) -> tuple[Float[Array, "batch output"], Float[Array, "batch output"]]:
        """Compute confidence intervals from prediction samples."""
        if confidence_level is None:
            confidence_level = self.confidence_level

        alpha = 1.0 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        lower_bound = jnp.percentile(predictions, lower_percentile, axis=0)
        upper_bound = jnp.percentile(predictions, upper_percentile, axis=0)

        return lower_bound, upper_bound

    def compute_prediction_intervals(
        self,
        mean_predictions: Float[Array, "batch output"],
        total_variance: Float[Array, "batch output"],
        confidence_level: float | None = None,
    ) -> tuple[Float[Array, "batch output"], Float[Array, "batch output"]]:
        """Compute prediction intervals using Gaussian assumption."""
        if confidence_level is None:
            confidence_level = self.confidence_level

        # Exact two-sided Gaussian quantile (inverse normal CDF). jax.scipy's
        # ``norm.ppf`` is JIT- and grad-compatible, so this stays traceable for
        # any confidence level (matches the in-repo convention in
        # uncertainty/reliability/failure_probability.py).
        alpha = 1.0 - confidence_level
        z_score = jsp.stats.norm.ppf(1.0 - alpha / 2.0)

        std_dev = jnp.sqrt(total_variance)
        margin = z_score * std_dev

        lower_bound = mean_predictions - margin
        upper_bound = mean_predictions + margin

        return lower_bound, upper_bound

    def propagate_uncertainty(
        self,
        predictions: Float[Array, "samples batch output"],
        inputs: Float[Array, "batch input_dim"],
        true_values: Float[Array, "batch output"] | None = None,
    ) -> UncertaintyIntegrationResults:
        """Propagate uncertainty through the entire prediction pipeline."""
        # Compute mean predictions
        mean_predictions = jnp.mean(predictions, axis=0)

        # Enhanced uncertainty decomposition
        uncertainty_components = self.enhanced_uncertainty_decomposition(
            predictions, true_values, inputs
        )

        # Compute confidence intervals
        confidence_intervals = self.compute_confidence_intervals(predictions)

        # Compute prediction intervals
        prediction_intervals = self.compute_prediction_intervals(
            mean_predictions, uncertainty_components.total
        )

        # Assess calibration if true values available
        if true_values is not None:
            calibration_metrics = self._assess_uncertainty_calibration(
                mean_predictions, uncertainty_components.total, true_values
            )
        else:
            # No true_values supplied — emit a zero-filled placeholder so the
            # downstream UncertaintyIntegrationResults still type-checks; the
            # zeros are not calibration claims, they are sentinel values
            # downstream consumers must check before using.
            calibration_metrics = CalibrationMetrics(
                expected_calibration_error=0.0,
                maximum_calibration_error=0.0,
                reliability_diagram={},
                confidence_histogram=jnp.array([]),
                accuracy_histogram=jnp.array([]),
            )

        return UncertaintyIntegrationResults(
            predictions=mean_predictions,
            uncertainty_components=uncertainty_components,
            calibration_metrics=calibration_metrics,
            confidence_intervals=confidence_intervals,
            prediction_intervals=prediction_intervals,
        )

    def _assess_uncertainty_calibration(
        self,
        predictions: Float[Array, "batch output"],
        uncertainties: Float[Array, "batch output"],
        true_values: Float[Array, "batch output"],
    ) -> CalibrationMetrics:
        """Assess how well uncertainties predict actual errors."""
        # Compute prediction errors
        errors = jnp.abs(predictions - true_values)

        # Convert uncertainties to confidences
        confidences = 1.0 / (1.0 + uncertainties)

        # Create binary accuracy (whether error is below median)
        median_error = jnp.median(errors)
        accuracies = jnp.asarray(errors <= median_error)

        # Compute calibration metrics using binning approach
        n_bins = 10
        ece, mce, reliability_data = self._compute_calibration_bins(confidences, accuracies, n_bins)

        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            reliability_diagram=reliability_data,
            confidence_histogram=confidences,
            accuracy_histogram=accuracies,
        )

    def _compute_calibration_bins(
        self,
        confidences: Float[Array, "..."],
        accuracies: Float[Array, "..."],
        n_bins: int,
    ) -> tuple[float, float, dict[str, Array]]:
        """Compute calibration metrics using reliability binning.

        Pure ``jnp.where``-based masked accumulation — no Python branches on
        traced arrays, no boolean fancy-indexing; traces under ``jax.jit``.
        """
        bin_boundaries = jnp.linspace(0.0, 1.0, n_bins + 1)
        bin_confidences, bin_accuracies, bin_counts = _bin_calibration_stats(
            confidences=confidences, accuracies=accuracies, bin_boundaries=bin_boundaries
        )
        bin_weights = bin_counts / jnp.maximum(jnp.sum(bin_counts), 1.0)
        calibration_errors = jnp.abs(bin_confidences - bin_accuracies)
        ece = float(jnp.sum(bin_weights * calibration_errors))
        mce = float(jnp.max(calibration_errors))
        reliability_data = {
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
            "bin_boundaries": bin_boundaries,
        }
        return ece, mce, reliability_data

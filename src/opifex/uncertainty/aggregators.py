# ruff: noqa: UP037
"""Uncertainty quantification utilities for Bayesian neural networks."""

from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from flax import nnx
    from jaxtyping import Array, Float

    # Define common type variable names for array dimensions
    batch = None  # Type variable for batch dimension


@dataclasses.dataclass
class UncertaintyComponents:
    """Decomposed uncertainty components."""

    epistemic: Float[Array, ...]  # Model uncertainty
    aleatoric: Float[Array, ...]  # Data uncertainty
    total: Float[Array, ...]  # Combined uncertainty


@dataclasses.dataclass
class CalibrationMetrics:
    """Uncertainty calibration assessment metrics."""

    expected_calibration_error: float
    maximum_calibration_error: float
    reliability_diagram: dict[str, Array]
    confidence_histogram: Array
    accuracy_histogram: Array


@dataclasses.dataclass
class UncertaintyIntegrationResults:
    """Results from uncertainty propagation through model pipeline."""

    predictions: Float[Array, "batch output"]
    uncertainty_components: UncertaintyComponents
    calibration_metrics: CalibrationMetrics
    confidence_intervals: tuple[Float[Array, "batch output"], Float[Array, "batch output"]]
    prediction_intervals: tuple[Float[Array, "batch output"], Float[Array, "batch output"]]


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

        # Use JAX-compatible approach instead of scipy
        # alpha = 1.0 - confidence_level  # Computed but not used in simplified impl
        # Approximate z-score for common confidence levels
        if confidence_level >= 0.99:
            z_score = 2.576  # 99%
        elif confidence_level >= 0.95:
            z_score = 1.96  # 95%
        elif confidence_level >= 0.90:
            z_score = 1.645  # 90%
        else:
            z_score = 1.0  # Fallback

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
            # Create dummy calibration metrics
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


def _bin_calibration_stats(
    *,
    confidences: Float[Array, "n_samples"],  # noqa: F821
    accuracies: Float[Array, "n_samples"],  # noqa: F821
    bin_boundaries: Float[Array, "n_bins_plus_1"],  # noqa: F821
) -> tuple[Array, Array, Array]:  # type: ignore[reportUndefinedVariable]
    """Vectorised reliability-bin statistics.

    For each bin ``b`` covering ``[lo_b, hi_b)`` (last bin closed),
    returns the mean confidence, mean accuracy, and sample count using
    pure ``jnp.where`` masked accumulation. No Python branches on traced
    arrays, no boolean fancy-indexing — traces under ``jax.jit``.
    """
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    n_bins = bin_lowers.shape[0]
    # (n_samples, n_bins) bin-membership mask.
    in_bin = (confidences[:, None] >= bin_lowers[None, :]) & (
        confidences[:, None] < bin_uppers[None, :]
    )
    # Close the rightmost bin on the upper edge.
    last_bin = jax.nn.one_hot(n_bins - 1, n_bins, dtype=jnp.bool_)
    closed_right = (confidences[:, None] == bin_uppers[None, :]) & last_bin[None, :]
    in_bin = in_bin | closed_right
    in_bin_f = in_bin.astype(jnp.float32)
    counts = jnp.sum(in_bin_f, axis=0)
    safe_counts = jnp.maximum(counts, 1.0)
    bin_confidences = jnp.sum(in_bin_f * confidences[:, None], axis=0) / safe_counts
    bin_accuracies = jnp.sum(in_bin_f * accuracies[:, None], axis=0) / safe_counts
    # Zero-out bins that had no samples so callers can detect empty bins via counts.
    nonempty = counts > 0
    bin_confidences = jnp.where(nonempty, bin_confidences, 0.0)
    bin_accuracies = jnp.where(nonempty, bin_accuracies, 0.0)
    return bin_confidences, bin_accuracies, counts


class CalibrationAssessment:
    """Enhanced uncertainty calibration assessment tools."""

    @staticmethod
    def expected_calibration_error(
        confidences: Float[Array, "n_samples"],  # noqa: F821
        accuracies: Float[Array, "n_samples"],  # noqa: F821
        n_bins: int = 10,
    ) -> float:  # type: ignore[reportUndefinedVariable]
        """Compute Expected Calibration Error (ECE)."""
        bin_boundaries = jnp.linspace(0.0, 1.0, n_bins + 1)
        bin_confidences, bin_accuracies, counts = _bin_calibration_stats(
            confidences=confidences, accuracies=accuracies, bin_boundaries=bin_boundaries
        )
        total = jnp.maximum(jnp.sum(counts), 1.0)
        bin_weights = counts / total
        ece = jnp.sum(bin_weights * jnp.abs(bin_confidences - bin_accuracies))
        return float(ece)

    @staticmethod
    def maximum_calibration_error(
        confidences: Float[Array, "n_samples"],  # noqa: F821
        accuracies: Float[Array, "n_samples"],  # noqa: F821
        n_bins: int = 10,
    ) -> float:  # type: ignore[reportUndefinedVariable]
        """Compute Maximum Calibration Error (MCE)."""
        bin_boundaries = jnp.linspace(0.0, 1.0, n_bins + 1)
        bin_confidences, bin_accuracies, counts = _bin_calibration_stats(
            confidences=confidences, accuracies=accuracies, bin_boundaries=bin_boundaries
        )
        # Mask empty bins out of the max — set their error to -inf so they
        # never win the argmax / max reduction.
        errors = jnp.abs(bin_confidences - bin_accuracies)
        errors = jnp.where(counts > 0, errors, -jnp.inf)
        return float(jnp.max(errors))

    @staticmethod
    def reliability_diagram_data(
        confidences: Float[Array, "n_samples"],  # noqa: F821
        accuracies: Float[Array, "n_samples"],  # noqa: F821
        n_bins: int = 10,
    ) -> dict[str, Array]:  # type: ignore[reportUndefinedVariable]
        """Compute reliability diagram data for visualization."""
        bin_boundaries = jnp.linspace(0.0, 1.0, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        bin_centers = (bin_lowers + bin_uppers) / 2
        bin_confidences, bin_accuracies, counts = _bin_calibration_stats(
            confidences=confidences, accuracies=accuracies, bin_boundaries=bin_boundaries
        )
        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": counts,
        }

    def assess_calibration(
        self,
        confidences: Float[Array, "n_samples"],  # noqa: F821
        accuracies: Float[Array, "n_samples"],  # noqa: F821
        n_bins: int = 10,
    ) -> CalibrationMetrics:  # type: ignore[reportUndefinedVariable]
        """Assess overall calibration with multiple metrics."""
        ece = self.expected_calibration_error(confidences, accuracies, n_bins)
        mce = self.maximum_calibration_error(confidences, accuracies, n_bins)
        rel_data = self.reliability_diagram_data(confidences, accuracies, n_bins)

        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            reliability_diagram=rel_data,
            confidence_histogram=rel_data["bin_confidences"],
            accuracy_histogram=rel_data["bin_accuracies"],
        )


@dataclasses.dataclass
class EnhancedUncertaintyComponents:
    """Enhanced uncertainty components with multiple sources."""

    epistemic_ensemble: Float[Array, "batch output"]  # Ensemble-based epistemic uncertainty
    aleatoric_distributional: Float[Array, "batch output"]  # Distributional aleatoric uncertainty
    total_uncertainty: Float[Array, "batch output"]  # Combined uncertainty
    uncertainty_breakdown: dict[str, Float[Array, "batch output"]]  # Detailed breakdown
    epistemic_dropout: Float[Array, "batch output"] | None = (
        None  # Dropout-based epistemic uncertainty
    )


class EnsembleEpistemicUncertainty:
    """Ensemble-based epistemic uncertainty estimation."""

    def __init__(self, num_models: int) -> None:
        """Initialize ensemble uncertainty estimator.

        Args:
            num_models: Number of models in the ensemble

        """
        self.num_models = num_models
        self.models: list = []

    def add_model(self, model: Any) -> None:
        """Add a model to the ensemble.

        Args:
            model: Neural network model to add to ensemble

        """
        if len(self.models) >= self.num_models:
            raise ValueError(f"Ensemble already has {self.num_models} models")
        self.models.append(model)

    def aggregate_predictions(
        self,
        ensemble_predictions: Float[Array, "models batch output"],
        method: str = "mean",
    ) -> Float[Array, "batch output"]:
        """Aggregate predictions from ensemble models.

        Args:
            ensemble_predictions: Predictions from all ensemble models
            method: Aggregation method ("mean", "median", "weighted_mean")

        Returns:
            Aggregated predictions

        """
        if method == "mean":
            return jnp.mean(ensemble_predictions, axis=0)
        if method == "median":
            return jnp.median(ensemble_predictions, axis=0)
        if method == "weighted_mean":
            # Equal weights for now, could be learned
            weights = jnp.ones(ensemble_predictions.shape[0]) / ensemble_predictions.shape[0]
            return jnp.average(ensemble_predictions, axis=0, weights=weights)
        raise ValueError(f"Unknown aggregation method: {method}")

    def compute_epistemic_uncertainty(
        self, ensemble_predictions: Float[Array, "models batch output"]
    ) -> Float[Array, "batch output"]:
        """Compute epistemic uncertainty from ensemble predictions.

        Args:
            ensemble_predictions: Predictions from all ensemble models

        Returns:
            Epistemic uncertainty (variance across models)

        """
        return jnp.var(ensemble_predictions, axis=0)

    def compute_prediction_disagreement(
        self, ensemble_predictions: Float[Array, "models batch output"]
    ) -> Float[Array, "batch output"]:
        """Compute prediction disagreement metric.

        Args:
            ensemble_predictions: Predictions from all ensemble models

        Returns:
            Disagreement metric (pairwise prediction variance)

        """
        # Compute pairwise differences
        models, _, _ = ensemble_predictions.shape
        disagreements = []

        for i in range(models):
            for j in range(i + 1, models):
                diff = jnp.abs(ensemble_predictions[i] - ensemble_predictions[j])
                disagreements.append(diff)

        # Average pairwise disagreements
        return jnp.mean(jnp.stack(disagreements), axis=0)


class DistributionalAleatoricUncertainty:
    """Distributional modeling of aleatoric uncertainty."""

    def sample_gaussian(
        self,
        mean: Float[Array, "batch output"],
        log_std: Float[Array, "batch output"],
        num_samples: int,
        *,
        rngs: nnx.Rngs,
    ) -> Float[Array, "samples batch output"]:
        """Sample from Gaussian distributional output.

        Args:
            mean: Mean predictions
            log_std: Log standard deviation predictions
            num_samples: Number of samples to draw
            rngs: Caller-owned RNG bundle; advances its ``sample`` stream once
                per call to produce reproducible-given-seed Monte-Carlo draws.

        Returns:
            Samples from the distributional output

        """
        std = jnp.exp(log_std)
        eps = jax.random.normal(rngs.sample(), (num_samples, *mean.shape))
        return mean + std * eps

    def compute_gaussian_uncertainty(
        self,
        mean: Float[Array, "batch output"],
        log_std: Float[Array, "batch output"],
    ) -> Float[Array, "batch output"]:
        """Compute uncertainty from Gaussian distributional parameters.

        Args:
            mean: Mean predictions
            log_std: Log standard deviation predictions

        Returns:
            Aleatoric uncertainty (variance)

        """
        return jnp.exp(2 * log_std)

    def compute_laplace_uncertainty(
        self, scale: Float[Array, "batch output"]
    ) -> Float[Array, "batch output"]:
        """Std-equivalent uncertainty from a Laplace(0, ``scale``) likelihood.

        A Laplace distribution with scale parameter ``b`` has variance
        ``2 * b**2`` and std ``b * sqrt(2)``. Returning the std lets the
        Laplace branch line up with :meth:`compute_gaussian_uncertainty`
        for callers that switch on the noise model.
        """
        return scale * jnp.sqrt(2.0)

    def compute_mixture_uncertainty(
        self,
        mixture_weights: Float[Array, "batch components"],
        means: Float[Array, "batch components output"],
        log_stds: Float[Array, "batch components output"],
    ) -> Float[Array, "batch output"]:
        """Compute uncertainty from mixture of Gaussians.

        Args:
            mixture_weights: Mixture component weights
            means: Component means
            log_stds: Component log standard deviations

        Returns:
            Total uncertainty from mixture model

        """
        # Compute weighted mean
        weighted_mean = jnp.sum(mixture_weights[..., None] * means, axis=1)

        # Compute weighted variance
        variances = jnp.exp(2 * log_stds)
        weighted_variance = jnp.sum(mixture_weights[..., None] * variances, axis=1)

        # Compute variance of means (epistemic component within mixture)
        mean_variance = jnp.sum(
            mixture_weights[..., None] * (means - weighted_mean[:, None, :]) ** 2,
            axis=1,
        )

        # Total uncertainty = weighted variance + variance of means
        return weighted_variance + mean_variance


class MultiSourceUncertaintyAggregator:
    """Aggregation of uncertainty from multiple sources."""

    def aggregate_uncertainties(
        self,
        epistemic_sources: list[Float[Array, "batch output"]],
        aleatoric_sources: list[Float[Array, "batch output"]],
        method: str = "variance_sum",
        epistemic_weights: jax.Array | None = None,
        aleatoric_weights: jax.Array | None = None,
    ) -> Float[Array, "batch output"]:
        """Aggregate uncertainties from multiple sources.

        Args:
            epistemic_sources: List of epistemic uncertainty estimates
            aleatoric_sources: List of aleatoric uncertainty estimates
            method: Aggregation method ("variance_sum", "weighted_sum", "max")
            epistemic_weights: Weights for epistemic sources
            aleatoric_weights: Weights for aleatoric sources

        Returns:
            Total aggregated uncertainty

        """
        if method == "variance_sum":
            # Sum variances (assumes independence)
            total_epistemic = jnp.sum(jnp.stack(epistemic_sources), axis=0)
            total_aleatoric = jnp.sum(jnp.stack(aleatoric_sources), axis=0)
            return total_epistemic + total_aleatoric

        if method == "weighted_sum":
            # Weighted sum of uncertainties
            if epistemic_weights is None:
                epistemic_weights = jnp.ones(len(epistemic_sources)) / len(epistemic_sources)
            if aleatoric_weights is None:
                aleatoric_weights = jnp.ones(len(aleatoric_sources)) / len(aleatoric_sources)

            weighted_epistemic = jnp.sum(
                jnp.stack(
                    [w * u for w, u in zip(epistemic_weights, epistemic_sources, strict=False)]
                ),
                axis=0,
            )
            weighted_aleatoric = jnp.sum(
                jnp.stack(
                    [w * u for w, u in zip(aleatoric_weights, aleatoric_sources, strict=False)]
                ),
                axis=0,
            )
            return weighted_epistemic + weighted_aleatoric

        if method == "max":
            # Maximum uncertainty across sources
            max_epistemic = jnp.max(jnp.stack(epistemic_sources), axis=0)
            max_aleatoric = jnp.max(jnp.stack(aleatoric_sources), axis=0)
            return max_epistemic + max_aleatoric

        raise ValueError(f"Unknown aggregation method: {method}")

    def compute_uncertainty_breakdown(
        self,
        epistemic_sources: list[Float[Array, "batch output"]],
        aleatoric_sources: list[Float[Array, "batch output"]],
        source_names: list[str] | None = None,
    ) -> dict[str, Float[Array, "batch output"]]:
        """Compute detailed uncertainty breakdown by source.

        Args:
            epistemic_sources: List of epistemic uncertainty estimates
            aleatoric_sources: List of aleatoric uncertainty estimates
            source_names: Names for uncertainty sources

        Returns:
            Dictionary mapping source names to uncertainty values

        """
        breakdown = {}

        if source_names is None:
            source_names = [f"epistemic_{i}" for i in range(len(epistemic_sources))] + [
                f"aleatoric_{i}" for i in range(len(aleatoric_sources))
            ]

        # Add epistemic sources
        for i, uncertainty in enumerate(epistemic_sources):
            name = source_names[i] if i < len(source_names) else f"epistemic_{i}"
            breakdown[name] = uncertainty

        # Add aleatoric sources
        for i, uncertainty in enumerate(aleatoric_sources):
            name = (
                source_names[len(epistemic_sources) + i]
                if len(epistemic_sources) + i < len(source_names)
                else f"aleatoric_{i}"
            )
            breakdown[name] = uncertainty

        return breakdown

    @staticmethod
    def adaptive_weighting(
        uncertainty_sources: list[Float[Array, "batch output"]],
        reliability_scores: list[Float[Array, "batch"]] | None = None,
        adaptation_method: str = "reliability_based",
    ) -> Float[Array, "sources batch"]:
        """Compute per-source, per-sample weights for uncertainty fusion.

        ``adaptation_method``:

        * ``reliability_based`` — weights ∝ per-source reliability scores
          supplied by the caller (must be non-``None``).
        * ``inverse_variance`` — weights ∝ ``1 / Var(source)``.
        * ``entropy_based`` — weights ∝ predictive entropy of each source.
        * ``uniform`` — flat weights, primarily a baseline.
        """
        n_sources = len(uncertainty_sources)
        batch_size = uncertainty_sources[0].shape[0]

        if adaptation_method == "reliability_based" and reliability_scores is not None:
            reliability_array = jnp.stack(reliability_scores, axis=0)
            return reliability_array / (jnp.sum(reliability_array, axis=0, keepdims=True) + 1e-8)
        if adaptation_method == "inverse_variance":
            variances = jnp.stack([jnp.var(u, axis=-1) for u in uncertainty_sources], axis=0)
            inv_variances = 1.0 / (variances + 1e-8)
            return inv_variances / (jnp.sum(inv_variances, axis=0, keepdims=True) + 1e-8)
        if adaptation_method == "entropy_based":
            entropies = jnp.stack(
                [-jnp.sum(u * jnp.log(u + 1e-8), axis=-1) for u in uncertainty_sources],
                axis=0,
            )
            return entropies / (jnp.sum(entropies, axis=0, keepdims=True) + 1e-8)
        if adaptation_method == "uniform":
            return jnp.ones((n_sources, batch_size)) / n_sources
        raise ValueError(f"Unknown adaptation method: {adaptation_method}")

    @staticmethod
    def assess_uncertainty_quality(
        predictions: Float[Array, "batch output"],
        uncertainties: Float[Array, "batch output"],
        true_values: Float[Array, "batch output"] | None = None,
    ) -> dict[str, float]:
        """Diagnostic summary of an uncertainty estimate.

        Always returns the source-side statistics (``mean_uncertainty``,
        ``uncertainty_std``, ``uncertainty_range``, ``mean_confidence``).
        If ``true_values`` is supplied, also returns
        ``coverage_probability`` (under a 2σ band), ``mean_interval_width``,
        and ``calibration_error`` (mean ``|y - μ| / σ``).
        """
        quality: dict[str, float] = {}

        if true_values is not None:
            lower_bound = predictions - 2.0 * uncertainties
            upper_bound = predictions + 2.0 * uncertainties
            coverage = jnp.mean((true_values >= lower_bound) & (true_values <= upper_bound))
            quality["coverage_probability"] = float(coverage)
            quality["mean_interval_width"] = float(jnp.mean(upper_bound - lower_bound))
            errors = jnp.abs(predictions - true_values)
            quality["calibration_error"] = float(jnp.mean(errors / (uncertainties + 1e-8)))

        quality["mean_uncertainty"] = float(jnp.mean(uncertainties))
        quality["uncertainty_std"] = float(jnp.std(uncertainties))
        quality["uncertainty_range"] = float(jnp.max(uncertainties) - jnp.min(uncertainties))
        quality["mean_confidence"] = float(jnp.mean(1.0 / (1.0 + uncertainties)))
        return quality


class EnhancedUncertaintyQuantifier:
    """Enhanced uncertainty quantifier with multiple decomposition methods."""

    def __init__(
        self,
        ensemble_size: int = 5,
        distributional_output: bool = True,
        multi_source_aggregation: bool = True,
        confidence_level: float = 0.95,
    ) -> None:
        """Initialize enhanced uncertainty quantifier.

        Args:
            ensemble_size: Number of models in ensemble
            distributional_output: Whether to use distributional outputs
            multi_source_aggregation: Whether to aggregate multiple uncertainty sources
            confidence_level: Confidence level for intervals

        """
        self.ensemble_size = ensemble_size
        self.distributional_output = distributional_output
        self.multi_source_aggregation = multi_source_aggregation
        self.confidence_level = confidence_level

        # Initialize components
        self.ensemble_estimator = EnsembleEpistemicUncertainty(ensemble_size)
        self.distributional_estimator = DistributionalAleatoricUncertainty()
        self.aggregator = MultiSourceUncertaintyAggregator()

    def enhanced_decompose_uncertainty(
        self,
        ensemble_predictions: Float[Array, "models batch output"],
        distributional_std: Float[Array, "batch output"] | None = None,
        inputs: Float[Array, "batch input_dim"] | None = None,
        dropout_predictions: Float[Array, "samples batch output"] | None = None,
    ) -> EnhancedUncertaintyComponents:
        """Enhanced uncertainty decomposition with multiple sources.

        Args:
            ensemble_predictions: Predictions from ensemble models
            distributional_std: Standard deviation from distributional output
            inputs: Input data for context-dependent uncertainty
            dropout_predictions: Predictions with dropout for additional
                epistemic uncertainty

        Returns:
            Enhanced uncertainty components with detailed breakdown

        """
        # Compute ensemble epistemic uncertainty
        epistemic_ensemble = self.ensemble_estimator.compute_epistemic_uncertainty(
            ensemble_predictions
        )

        # Compute distributional aleatoric uncertainty
        if distributional_std is not None:
            mean_predictions = jnp.mean(ensemble_predictions, axis=0)
            log_std = jnp.log(distributional_std + 1e-8)
            aleatoric_distributional = self.distributional_estimator.compute_gaussian_uncertainty(
                mean_predictions, log_std
            )
        else:
            aleatoric_distributional = jnp.zeros_like(epistemic_ensemble)

        # Compute dropout epistemic uncertainty (if available)
        epistemic_dropout = None
        if dropout_predictions is not None:
            epistemic_dropout = jnp.var(dropout_predictions, axis=0)

        # Aggregate uncertainties
        epistemic_sources = [epistemic_ensemble]
        if epistemic_dropout is not None:
            epistemic_sources.append(epistemic_dropout)

        aleatoric_sources = [aleatoric_distributional]

        total_uncertainty = self.aggregator.aggregate_uncertainties(
            epistemic_sources, aleatoric_sources
        )

        # Compute detailed breakdown
        uncertainty_breakdown = self.aggregator.compute_uncertainty_breakdown(
            epistemic_sources,
            aleatoric_sources,
            source_names=["ensemble", "dropout", "distributional"][
                : len(epistemic_sources) + len(aleatoric_sources)
            ],
        )

        return EnhancedUncertaintyComponents(
            epistemic_ensemble=epistemic_ensemble,
            epistemic_dropout=epistemic_dropout,
            aleatoric_distributional=aleatoric_distributional,
            total_uncertainty=total_uncertainty,
            uncertainty_breakdown=uncertainty_breakdown,
        )

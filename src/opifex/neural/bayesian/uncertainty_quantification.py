# ruff: noqa: UP037
"""Uncertainty quantification utilities for Bayesian neural networks."""

from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
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
    confidence_intervals: tuple[
        Float[Array, "batch output"], Float[Array, "batch output"]
    ]
    prediction_intervals: tuple[
        Float[Array, "batch output"], Float[Array, "batch output"]
    ]


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
        """Compute variance of expected predictions (pure epistemic uncertainty)."""
        mean_predictions = jnp.mean(predictions, axis=0)
        return jnp.var(jnp.broadcast_to(mean_predictions, predictions.shape), axis=0)


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

    def __init__(self, num_samples: int = 100, confidence_level: float = 0.95):
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

        return UncertaintyComponents(
            epistemic=epistemic, aleatoric=aleatoric, total=total
        )

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
            aleatoric = AleatoricUncertainty.noise_estimation(
                residuals, mean_predictions
            )
        elif inputs is not None:
            # Use input complexity to estimate aleatoric uncertainty
            input_complexity = jnp.var(inputs, axis=-1, keepdims=True)
            aleatoric = 0.1 * input_complexity * jnp.ones_like(epistemic)
        else:
            # Fallback: fraction of epistemic uncertainty
            aleatoric = 0.2 * epistemic

        # Total uncertainty with proper combination
        total = epistemic + aleatoric

        return UncertaintyComponents(
            epistemic=epistemic, aleatoric=aleatoric, total=total
        )

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
        ece, mce, reliability_data = self._compute_calibration_bins(
            confidences, accuracies, n_bins
        )

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
        """Compute calibration metrics using reliability binning."""

        bin_boundaries = jnp.linspace(0.0, 1.0, n_bins + 1)
        bin_confidences = jnp.zeros(n_bins)
        bin_accuracies = jnp.zeros(n_bins)
        bin_counts = jnp.zeros(n_bins)

        # Compute statistics for each bin
        for i in range(n_bins):
            # Find samples in this bin
            in_bin = (confidences >= bin_boundaries[i]) & (
                confidences < bin_boundaries[i + 1]
            )

            # Handle last bin edge case
            if i == n_bins - 1:
                in_bin = in_bin | (confidences == bin_boundaries[i + 1])

            bin_count = jnp.sum(in_bin)

            if bin_count > 0:
                bin_confidences = bin_confidences.at[i].set(
                    jnp.mean(confidences[in_bin])
                )
                bin_accuracies = bin_accuracies.at[i].set(jnp.mean(accuracies[in_bin]))
                bin_counts = bin_counts.at[i].set(bin_count)

        # Expected Calibration Error
        bin_weights = bin_counts / jnp.sum(bin_counts)
        calibration_errors = jnp.abs(bin_confidences - bin_accuracies)
        ece = float(jnp.sum(bin_weights * calibration_errors))

        # Maximum Calibration Error
        mce = float(jnp.max(calibration_errors))

        # Reliability diagram data
        reliability_data = {
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
            "bin_boundaries": bin_boundaries,
        }

        return ece, mce, reliability_data


class CalibrationAssessment:
    """Enhanced uncertainty calibration assessment tools."""

    @staticmethod
    def expected_calibration_error(
        confidences: Float[Array, "n_samples"],  # noqa: F821
        accuracies: Float[Array, "n_samples"],  # noqa: F821
        n_bins: int = 10,
    ) -> float:  # type: ignore[reportUndefinedVariable]
        """Compute Expected Calibration Error (ECE)."""
        bin_boundaries = jnp.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            # Find points in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = jnp.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = jnp.mean(accuracies[in_bin])
                avg_confidence_in_bin = jnp.mean(confidences[in_bin])
                ece += jnp.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)

    @staticmethod
    def maximum_calibration_error(
        confidences: Float[Array, "n_samples"],  # noqa: F821
        accuracies: Float[Array, "n_samples"],  # noqa: F821
        n_bins: int = 10,
    ) -> float:  # type: ignore[reportUndefinedVariable]
        """Compute Maximum Calibration Error (MCE)."""
        bin_boundaries = jnp.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            # Find points in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = jnp.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = jnp.mean(accuracies[in_bin])
                avg_confidence_in_bin = jnp.mean(confidences[in_bin])
                mce = jnp.maximum(mce, jnp.abs(avg_confidence_in_bin - accuracy_in_bin))

        return float(mce)

    @staticmethod
    def reliability_diagram_data(
        confidences: Float[Array, "n_samples"],  # noqa: F821
        accuracies: Float[Array, "n_samples"],  # noqa: F821
        n_bins: int = 10,
    ) -> dict[str, Array]:  # type: ignore[reportUndefinedVariable]
        """Compute reliability diagram data for visualization."""
        bin_boundaries = jnp.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        bin_centers = (bin_lowers + bin_uppers) / 2

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = jnp.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = float(jnp.mean(accuracies[in_bin]))
                avg_confidence_in_bin = float(jnp.mean(confidences[in_bin]))
                count_in_bin = float(jnp.sum(in_bin))
            else:
                accuracy_in_bin = 0.0
                avg_confidence_in_bin = 0.0
                count_in_bin = 0.0

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(count_in_bin)

        return {
            "bin_centers": bin_centers,
            "bin_accuracies": jnp.array(bin_accuracies),
            "bin_confidences": jnp.array(bin_confidences),
            "bin_counts": jnp.array(bin_counts),
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

    epistemic_ensemble: Float[
        Array, "batch output"
    ]  # Ensemble-based epistemic uncertainty
    aleatoric_distributional: Float[
        Array, "batch output"
    ]  # Distributional aleatoric uncertainty
    total_uncertainty: Float[Array, "batch output"]  # Combined uncertainty
    uncertainty_breakdown: dict[str, Float[Array, "batch output"]]  # Detailed breakdown
    epistemic_dropout: Float[Array, "batch output"] | None = (
        None  # Dropout-based epistemic uncertainty
    )


class EnsembleEpistemicUncertainty:
    """Ensemble-based epistemic uncertainty estimation."""

    def __init__(self, num_models: int):
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
            weights = (
                jnp.ones(ensemble_predictions.shape[0]) / ensemble_predictions.shape[0]
            )
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
    ) -> Float[Array, "samples batch output"]:
        """Sample from Gaussian distributional output.

        Args:
            mean: Mean predictions
            log_std: Log standard deviation predictions
            num_samples: Number of samples to draw

        Returns:
            Samples from the distributional output
        """
        std = jnp.exp(log_std)
        eps = jax.random.normal(jax.random.PRNGKey(42), (num_samples, *mean.shape))
        return mean + std * eps

    def compute_gaussian_uncertainty(
        self, mean: Float[Array, "batch output"], log_std: Float[Array, "batch output"]
    ) -> Float[Array, "batch output"]:
        """Compute uncertainty from Gaussian distributional parameters.

        Args:
            mean: Mean predictions
            log_std: Log standard deviation predictions

        Returns:
            Aleatoric uncertainty (variance)
        """
        return jnp.exp(2 * log_std)

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
                epistemic_weights = jnp.ones(len(epistemic_sources)) / len(
                    epistemic_sources
                )
            if aleatoric_weights is None:
                aleatoric_weights = jnp.ones(len(aleatoric_sources)) / len(
                    aleatoric_sources
                )

            weighted_epistemic = jnp.sum(
                jnp.stack(
                    [
                        w * u
                        for w, u in zip(
                            epistemic_weights, epistemic_sources, strict=False
                        )
                    ]
                ),
                axis=0,
            )
            weighted_aleatoric = jnp.sum(
                jnp.stack(
                    [
                        w * u
                        for w, u in zip(
                            aleatoric_weights, aleatoric_sources, strict=False
                        )
                    ]
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


class EnhancedUncertaintyQuantifier:
    """Enhanced uncertainty quantifier with multiple decomposition methods."""

    def __init__(
        self,
        ensemble_size: int = 5,
        distributional_output: bool = True,
        multi_source_aggregation: bool = True,
        confidence_level: float = 0.95,
    ):
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
            aleatoric_distributional = (
                self.distributional_estimator.compute_gaussian_uncertainty(
                    mean_predictions, log_std
                )
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


class AdvancedUncertaintyAggregator:
    """Advanced uncertainty aggregation with multiple sources and weighting."""

    @staticmethod
    def weighted_uncertainty_aggregation(
        uncertainty_sources: list[Float[Array, "batch output"]],
        weights: Float[Array, "sources"] | None = None,  # noqa: F821
        aggregation_method: str = "weighted_variance",
    ) -> Float[Array, "batch output"]:
        """Aggregate uncertainties from multiple sources with optional weighting."""
        uncertainties = jnp.stack(uncertainty_sources, axis=0)

        if weights is None:
            weights = jnp.ones(len(uncertainty_sources)) / len(uncertainty_sources)

        weights = weights / jnp.sum(weights)  # Normalize weights

        if aggregation_method == "weighted_variance":
            # Aggregate as weighted sum of variances
            return jnp.sum(weights[:, None, None] * uncertainties**2, axis=0)
        if aggregation_method == "weighted_mean":
            # Simple weighted average
            return jnp.sum(weights[:, None, None] * uncertainties, axis=0)
        if aggregation_method == "max_weighted":
            # Maximum weighted uncertainty
            weighted_uncertainties = weights[:, None, None] * uncertainties
            return jnp.max(weighted_uncertainties, axis=0)
        if aggregation_method == "robust_weighted":
            # Robust aggregation using median
            weighted_uncertainties = weights[:, None, None] * uncertainties
            return jnp.median(weighted_uncertainties, axis=0)
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    @staticmethod
    def adaptive_weighting(
        uncertainty_sources: list[Float[Array, "batch output"]],
        reliability_scores: list[Float[Array, "batch"]] | None = None,
        adaptation_method: str = "reliability_based",
    ) -> Float[Array, "sources batch"]:
        """Compute adaptive weights for uncertainty sources based on reliability."""
        n_sources = len(uncertainty_sources)
        batch_size = uncertainty_sources[0].shape[0]

        if adaptation_method == "reliability_based" and reliability_scores is not None:
            # Weight by reliability scores
            reliability_array = jnp.stack(reliability_scores, axis=0)
            weights = reliability_array / (
                jnp.sum(reliability_array, axis=0, keepdims=True) + 1e-8
            )
        elif adaptation_method == "inverse_variance":
            # Weight inversely proportional to variance
            variances = jnp.stack(
                [jnp.var(u, axis=-1) for u in uncertainty_sources], axis=0
            )
            inv_variances = 1.0 / (variances + 1e-8)
            weights = inv_variances / (
                jnp.sum(inv_variances, axis=0, keepdims=True) + 1e-8
            )
        elif adaptation_method == "entropy_based":
            # Weight based on predictive entropy
            entropies = jnp.stack(
                [-jnp.sum(u * jnp.log(u + 1e-8), axis=-1) for u in uncertainty_sources],
                axis=0,
            )
            weights = entropies / (jnp.sum(entropies, axis=0, keepdims=True) + 1e-8)
        elif adaptation_method == "uniform":
            # Uniform weighting
            weights = jnp.ones((n_sources, batch_size)) / n_sources
        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")

        return weights

    @staticmethod
    def uncertainty_quality_assessment(
        predictions: Float[Array, "batch output"],
        uncertainties: Float[Array, "batch output"],
        true_values: Float[Array, "batch output"] | None = None,
    ) -> dict[str, float]:
        """Assess the quality of uncertainty estimates."""
        quality_metrics = {}

        # Coverage probability (if true values available)
        if true_values is not None:
            # Compute prediction intervals
            lower_bound = predictions - 2 * uncertainties
            upper_bound = predictions + 2 * uncertainties

            # Check coverage
            coverage = jnp.mean(
                (true_values >= lower_bound) & (true_values <= upper_bound)
            )
            quality_metrics["coverage_probability"] = float(coverage)

            # Interval width
            interval_width = jnp.mean(upper_bound - lower_bound)
            quality_metrics["mean_interval_width"] = float(interval_width)

            # Calibration error
            errors = jnp.abs(predictions - true_values)
            normalized_errors = errors / (uncertainties + 1e-8)
            quality_metrics["calibration_error"] = float(jnp.mean(normalized_errors))

        # Uncertainty statistics
        quality_metrics["mean_uncertainty"] = float(jnp.mean(uncertainties))
        quality_metrics["uncertainty_std"] = float(jnp.std(uncertainties))
        quality_metrics["uncertainty_range"] = float(
            jnp.max(uncertainties) - jnp.min(uncertainties)
        )

        # Prediction confidence
        prediction_confidence = 1.0 / (1.0 + uncertainties)
        quality_metrics["mean_confidence"] = float(jnp.mean(prediction_confidence))

        return quality_metrics


class AdvancedEpistemicUncertainty:
    """Advanced epistemic uncertainty estimation methods."""

    @staticmethod
    def compute_ensemble_disagreement(
        ensemble_predictions: Float[Array, "models batch output"],
        aggregation_method: str = "variance",
    ) -> Float[Array, "batch output"]:
        """Compute epistemic uncertainty from ensemble disagreement."""
        if aggregation_method == "variance":
            return jnp.var(ensemble_predictions, axis=0)
        if aggregation_method == "std":
            return jnp.std(ensemble_predictions, axis=0)
        if aggregation_method == "range":
            return jnp.max(ensemble_predictions, axis=0) - jnp.min(
                ensemble_predictions, axis=0
            )
        if aggregation_method == "iqr":
            q75 = jnp.percentile(ensemble_predictions, 75, axis=0)
            q25 = jnp.percentile(ensemble_predictions, 25, axis=0)
            return q75 - q25
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    @staticmethod
    def compute_predictive_diversity(
        ensemble_predictions: Float[Array, "models batch output"],
        diversity_metric: str = "pairwise_distance",
    ) -> Float[Array, "batch output"]:
        """Compute predictive diversity as a measure of epistemic uncertainty."""
        if diversity_metric == "pairwise_distance":
            # Compute average pairwise L2 distance
            n_models = ensemble_predictions.shape[0]
            total_distance = jnp.zeros_like(ensemble_predictions[0, :, 0])
            count = 0
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    distance = jnp.linalg.norm(
                        ensemble_predictions[i] - ensemble_predictions[j], axis=-1
                    )
                    total_distance += distance
                    count += 1
            if count > 0:
                return total_distance / count
            return jnp.zeros_like(ensemble_predictions[0, :, 0])
        if diversity_metric == "cosine_diversity":
            # Compute average cosine diversity
            mean_pred = jnp.mean(ensemble_predictions, axis=0)
            cosine_similarities = []
            for i in range(ensemble_predictions.shape[0]):
                cos_sim = jnp.sum(ensemble_predictions[i] * mean_pred, axis=-1) / (
                    jnp.linalg.norm(ensemble_predictions[i], axis=-1)
                    * jnp.linalg.norm(mean_pred, axis=-1)
                    + 1e-8
                )
                cosine_similarities.append(cos_sim)
            return 1.0 - jnp.mean(jnp.stack(cosine_similarities), axis=0)
        raise ValueError(f"Unknown diversity metric: {diversity_metric}")


class AdvancedAleatoricUncertainty:
    """Advanced aleatoric uncertainty estimation methods."""

    @staticmethod
    def distributional_uncertainty(
        distribution_params: dict[str, Float[Array, "batch ..."]],
        distribution_type: str = "gaussian",
    ) -> Float[Array, "batch output"]:
        """Compute aleatoric uncertainty from distributional outputs."""
        if distribution_type == "gaussian":
            if "log_std" in distribution_params:
                return jnp.exp(distribution_params["log_std"])
            if "std" in distribution_params:
                return distribution_params["std"]
            if "variance" in distribution_params:
                return jnp.sqrt(distribution_params["variance"])
            raise ValueError(
                "Gaussian distribution requires 'log_std', 'std', or 'variance'"
            )
        if distribution_type == "laplace":
            if "scale" in distribution_params:
                return distribution_params["scale"] * jnp.sqrt(
                    2.0
                )  # Convert to std equivalent
            raise ValueError("Laplace distribution requires 'scale' parameter")
        if distribution_type == "mixture":
            # For mixture distributions, compute weighted variance
            weights = distribution_params["weights"]
            means = distribution_params["means"]
            variances = distribution_params["variances"]

            # Mixture variance formula: E[Var] + Var[E]
            expected_variance = jnp.sum(weights * variances, axis=-2)
            variance_of_means = (
                jnp.sum(weights * means**2, axis=-2)
                - (jnp.sum(weights * means, axis=-2)) ** 2
            )

            return jnp.sqrt(expected_variance + variance_of_means)
        raise ValueError(f"Unknown distribution type: {distribution_type}")

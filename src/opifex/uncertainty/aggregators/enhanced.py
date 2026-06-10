# ruff: noqa: UP037
"""Enhanced ensemble / distributional / multi-source uncertainty quantifiers."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.uncertainty.aggregators.types import EnhancedUncertaintyComponents


if TYPE_CHECKING:
    from flax import nnx
    from jaxtyping import Array, Float

    # Define common type variable names for array dimensions
    batch = None  # Type variable for batch dimension


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
        mean: Float[Array, "batch output"],  # noqa: ARG002 - aggregation interface receives the predictive mean
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
        inputs: Float[Array, "batch input_dim"] | None = None,  # noqa: ARG002 - decomposition interface receives inputs
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

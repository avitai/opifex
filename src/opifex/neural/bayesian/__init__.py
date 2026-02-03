"""Bayesian neural network components with uncertainty quantification."""

# Import new probabilistic framework components
from opifex.neural.bayesian.blackjax_integration import BlackJAXIntegration
from opifex.neural.bayesian.calibration_tools import (
    CalibrationTools,
    ConformalPrediction,
    IsotonicRegression,
    PlattScaling,
    TemperatureScaling,
)
from opifex.neural.bayesian.conformal import ConformalConfig, ConformalPredictor
from opifex.neural.bayesian.physics_informed_priors import (
    ConservationLawPriors,
    DomainSpecificPriors,
    HierarchicalBayesianFramework,
    PhysicsAwareUncertaintyPropagation,
    PhysicsInformedPriors,
)
from opifex.neural.bayesian.uncertainty_quantification import (
    AdvancedAleatoricUncertainty,
    AdvancedEpistemicUncertainty,
    AdvancedUncertaintyAggregator,
    AleatoricUncertainty,
    CalibrationAssessment,
    CalibrationMetrics,
    DistributionalAleatoricUncertainty,
    EnhancedUncertaintyComponents,
    EnhancedUncertaintyQuantifier,
    EnsembleEpistemicUncertainty,
    EpistemicUncertainty,
    MultiSourceUncertaintyAggregator,
    UncertaintyComponents,
    UncertaintyIntegrationResults,
    UncertaintyQuantifier,
)
from opifex.neural.bayesian.variational_framework import (
    AmortizedVariationalFramework,
    MeanFieldGaussian,
    PriorConfig,
    UncertaintyEncoder,
    VariationalConfig,
)


__all__ = [
    "AdvancedAleatoricUncertainty",
    "AdvancedEpistemicUncertainty",
    "AdvancedUncertaintyAggregator",
    "AleatoricUncertainty",
    "AmortizedVariationalFramework",
    "BlackJAXIntegration",
    "CalibrationAssessment",
    "CalibrationMetrics",
    "CalibrationTools",
    "ConformalConfig",
    "ConformalPrediction",
    "ConformalPredictor",
    "ConservationLawPriors",
    "DistributionalAleatoricUncertainty",
    "DomainSpecificPriors",
    "EnhancedUncertaintyComponents",
    "EnhancedUncertaintyQuantifier",
    "EnsembleEpistemicUncertainty",
    "EpistemicUncertainty",
    "HierarchicalBayesianFramework",
    "IsotonicRegression",
    "MeanFieldGaussian",
    "MultiSourceUncertaintyAggregator",
    "PhysicsAwareUncertaintyPropagation",
    "PhysicsInformedPriors",
    "PlattScaling",
    "PriorConfig",
    "TemperatureScaling",
    "UncertaintyComponents",
    "UncertaintyEncoder",
    "UncertaintyIntegrationResults",
    "UncertaintyQuantifier",
    "VariationalConfig",
]

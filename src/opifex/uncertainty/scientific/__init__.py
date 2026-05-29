"""Scientific-domain UQ utilities: field metadata, field metrics, domain metrics."""

from __future__ import annotations

from opifex.uncertainty.registry import UQRegistry
from opifex.uncertainty.scientific._uq_capabilities import (
    SCIENTIFIC_FIELD_CAPABILITIES,
)
from opifex.uncertainty.scientific.domain_metrics import (
    boundary_condition_coverage,
    chemical_accuracy_coverage,
    DomainMetricSummary,
    feasibility_coverage,
    parameter_credible_interval_coverage,
    physics_residual_coverage,
    regret_interval_summary,
    sensor_reliability_summary,
    spectral_coverage,
    UNSUPPORTED_ACTIVE_LEARNING,
    UNSUPPORTED_LIKELIHOOD_FREE,
    UNSUPPORTED_PAC_BAYES,
)
from opifex.uncertainty.scientific.fields import (
    conservation_law_residual_summary,
    FieldMetadata,
    function_space_l2_coverage,
    residual_uncertainty_alignment,
    spatial_calibration_error,
)
from opifex.uncertainty.scientific.polynomial_chaos import (
    evaluate_basis,
    fit_pce_coefficients,
    KarhunenLoeveExpansion,
    KLEConfig,
    pce_mean_variance,
    pce_summary,
    PCESummary,
    PolynomialChaosBasis,
    PolynomialChaosConfig,
    smolyak_sparse_grid,
    SparseGrid,
    StochasticCollocationSurrogate,
    StochasticGalerkinSurrogate,
    tensor_grid_gauss_hermite,
)
from opifex.uncertainty.scientific.probabilistic_numerics import (
    CalibrationSpec,
    CorrectionSpec,
    CubatureRuleSpec,
    DaltonAdapterSpec,
    DataUpdateCallbackSpec,
    DenseOutputSamplingSpec,
    DiagonalEK1Spec,
    DiffeqzooAdapterSpec,
    DiffusionSpec,
    DynamicMVDiffusionSpec,
    ExpEKSpec,
    FenrirAdapterSpec,
    FixedMVDiffusionSpec,
    InitSchemeSpec,
    IOUPPriorSpec,
    IWPPriorSpec,
    ManifoldUpdateSpec,
    MaternPriorSpec,
    PerturbedStepSolverSpec,
    ProbdiffeqAdapterSpec,
    ProbfindiffAdapterSpec,
    ProbnumAdapterSpec,
    RosenbrockExpEKSpec,
    SsmFactSpec,
    StrategySpec,
    TornadoxAdapterSpec,
)
from opifex.uncertainty.scientific.solutions import (
    aggregate_solver_solutions,
    SolutionDistribution,
    summarize_stacked_sample_solution,
)
from opifex.uncertainty.scientific.stochastic_fields import (
    sample_kle_field,
    sample_pce_field,
)
from opifex.uncertainty.scientific.stochastic_galerkin import (
    evaluate_collocation_surrogate,
    evaluate_galerkin_surrogate,
    fit_collocation_surrogate,
    fit_galerkin_surrogate,
)


# Idempotent capability registration (Rule 13 — no mutable side effects
# beyond constants + idempotent registry seeding). The :class:`UQRegistry`
# is a singleton; guard against double-registration on repeat imports.
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in SCIENTIFIC_FIELD_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = [
    "SCIENTIFIC_FIELD_CAPABILITIES",
    "UNSUPPORTED_ACTIVE_LEARNING",
    "UNSUPPORTED_LIKELIHOOD_FREE",
    "UNSUPPORTED_PAC_BAYES",
    "CalibrationSpec",
    "CorrectionSpec",
    "CubatureRuleSpec",
    "DaltonAdapterSpec",
    "DataUpdateCallbackSpec",
    "DenseOutputSamplingSpec",
    "DiagonalEK1Spec",
    "DiffeqzooAdapterSpec",
    "DiffusionSpec",
    "DomainMetricSummary",
    "DynamicMVDiffusionSpec",
    "ExpEKSpec",
    "FenrirAdapterSpec",
    "FieldMetadata",
    "FixedMVDiffusionSpec",
    "IOUPPriorSpec",
    "IWPPriorSpec",
    "InitSchemeSpec",
    "KLEConfig",
    "KarhunenLoeveExpansion",
    "ManifoldUpdateSpec",
    "MaternPriorSpec",
    "PCESummary",
    "PerturbedStepSolverSpec",
    "PolynomialChaosBasis",
    "PolynomialChaosConfig",
    "ProbdiffeqAdapterSpec",
    "ProbfindiffAdapterSpec",
    "ProbnumAdapterSpec",
    "RosenbrockExpEKSpec",
    "SolutionDistribution",
    "SparseGrid",
    "SsmFactSpec",
    "StochasticCollocationSurrogate",
    "StochasticGalerkinSurrogate",
    "StrategySpec",
    "TornadoxAdapterSpec",
    "aggregate_solver_solutions",
    "boundary_condition_coverage",
    "chemical_accuracy_coverage",
    "conservation_law_residual_summary",
    "evaluate_basis",
    "evaluate_collocation_surrogate",
    "evaluate_galerkin_surrogate",
    "feasibility_coverage",
    "fit_collocation_surrogate",
    "fit_galerkin_surrogate",
    "fit_pce_coefficients",
    "function_space_l2_coverage",
    "parameter_credible_interval_coverage",
    "pce_mean_variance",
    "pce_summary",
    "physics_residual_coverage",
    "regret_interval_summary",
    "residual_uncertainty_alignment",
    "sample_kle_field",
    "sample_pce_field",
    "sensor_reliability_summary",
    "smolyak_sparse_grid",
    "spatial_calibration_error",
    "spectral_coverage",
    "summarize_stacked_sample_solution",
    "tensor_grid_gauss_hermite",
]

"""Neural Tangent Kernel (NTK) utilities for scientific machine learning.

This module provides tools for computing and analyzing the Neural Tangent Kernel,
which is fundamental for understanding neural network training dynamics.

Key Components:
    - NTKWrapper: Wrapper for NTK computation with NNX models
    - compute_empirical_ntk: Compute empirical NTK matrix
    - NTKConfig: Configuration for NTK computation
    - NTKTrainingDiagnostics: Track NTK evolution during training
    - NTKDiagnosticsCallback: Training callback for NTK monitoring

References:
    - Jacot et al. (2018): Neural Tangent Kernel
    - Survey Section 3: Neural Tangent Kernel Analysis
    - Google's neural-tangents library
"""

from opifex.core.physics.ntk.diagnostics import (
    compute_mode_coefficients,
    compute_mode_decay_factors,
    detect_spectral_bias,
    estimate_convergence_rate,
    estimate_epochs_to_convergence,
    identify_slow_modes,
    NTKDiagnosticsCallback,
    NTKTrainingDiagnostics,
    predict_mode_errors,
)
from opifex.core.physics.ntk.spectral_analysis import (
    compute_condition_number,
    compute_condition_number_from_ntk,
    compute_effective_rank,
    compute_mode_convergence_rates,
    compute_spectral_bias_indicator,
    estimate_pde_order,
    NTKDiagnostics,
    NTKSpectralAnalyzer,
)
from opifex.core.physics.ntk.wrapper import (
    compute_empirical_ntk,
    compute_jacobian,
    create_ntk_fn_from_nnx,
    flatten_jacobian,
    NTKConfig,
    NTKWrapper,
)


__all__ = [
    "NTKConfig",
    "NTKDiagnostics",
    "NTKDiagnosticsCallback",
    "NTKSpectralAnalyzer",
    "NTKTrainingDiagnostics",
    "NTKWrapper",
    "compute_condition_number",
    "compute_condition_number_from_ntk",
    "compute_effective_rank",
    "compute_empirical_ntk",
    "compute_jacobian",
    "compute_mode_coefficients",
    "compute_mode_convergence_rates",
    "compute_mode_decay_factors",
    "compute_spectral_bias_indicator",
    "create_ntk_fn_from_nnx",
    "detect_spectral_bias",
    "estimate_convergence_rate",
    "estimate_epochs_to_convergence",
    "estimate_pde_order",
    "flatten_jacobian",
    "identify_slow_modes",
    "predict_mode_errors",
]

"""UQ capability declarations for GP adapter specs + Phase 4 model adapters.

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.uncertainty.adapters.__init__`` which then registers each
declaration into the singleton :class:`UQRegistry`.

The GP entries (Task 6.3.4 spec inventory) live here rather than in a
dedicated ``opifex.uncertainty.gp/`` subpackage because Phase 6 Task
6.3.4 shipped specs only — the algorithms themselves are vendored into
:mod:`opifex.uncertainty.statespace` (state-space GPs) or live in
user-installed backends (GPJax / tinygp).

Phase 4 model adapter specs (Bayesian last-layer / Laplace / SNGP /
VBLL) are paired with capability declarations advertising
``supports_calibration`` so the registry can be queried for adapter
coverage of model-uncertainty surfaces.
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


# ---------------------------------------------------------------------------
# Gaussian-process adapter specs (Task 6.3.4 inventory).
#
# Each adapter is metadata-only (the algorithms are either in a
# user-installed backend or vendored under
# :mod:`opifex.uncertainty.statespace`); ``default_strategy`` is
# ``DefaultStrategy.UNSUPPORTED`` until a backend lands or the user
# installs the upstream package (plan §7.2).
# ---------------------------------------------------------------------------


_GPJAX_ADAPTER_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="gpjax",
    notes=(
        "GPJaxAdapterSpec — 9 family tags: exact_gp, conjugate_gaussian, "
        "svgp, non_conjugate, multi_output, deep_kernel, "
        "stochastic_variational, natural_gradient, rff_approximation. "
        "GPJax 0.14+ uses equinox.Module; users must supply an "
        "eqx<->nnx PyTree bridge before wiring into NNX-native surfaces."
    ),
)


_TINYGP_ADAPTER_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="tinygp",
    notes=(
        "TinygpAdapterSpec — 4 family tags: exact_gp, "
        "conjugate_gaussian, stationary_kernel, quasisep_1d_state_space. "
        "Recommended substrate for the future LUNO (Laplace UQ Neural "
        "Operator) task."
    ),
)


_MARKOVFLOW_ADAPTER_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="markovflow",
    notes=(
        "MarkovflowAdapterSpec — metadata-only. Specific algorithms "
        "(banded-precision Cholesky, SDE-linearize) are vendored into "
        "opifex.uncertainty.statespace. markovflow is TF-based and not "
        "directly importable from opifex."
    ),
)


_BAYESNEWTON_ADAPTER_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="bayesnewton",
    notes=(
        "BayesnewtonAdapterSpec — metadata-only. Sequential and "
        "parallel-scan Kalman primitives plus Matern / Cosine / "
        "Periodic state-space kernels are vendored into "
        "opifex.uncertainty.statespace. bayesnewton's pinned "
        "jax==0.4.14 + objax stack conflicts with the opifex JAX "
        "baseline."
    ),
)


_KALMAN_JAX_ADAPTER_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="kalman-jax",
    notes=(
        "KalmanJaxAdapterSpec — deprecated. kalman-jax/README.md:1 "
        "states bayesnewton is the official successor; the generic "
        "expm(F*dt) LTI-SDE discretization is vendored into "
        "opifex.uncertainty.statespace.discretize_lti_sde. Constructing "
        "the spec emits a DeprecationWarning."
    ),
)


# ---------------------------------------------------------------------------
# Phase 4 model adapter specs — declarative metadata only.
#
# These entries match the AdapterSpec classes exported from
# ``opifex.uncertainty.adapters.model``; ``source_package`` records the
# canonical implementation home so the registry can flag overclaiming.
# ---------------------------------------------------------------------------


_LAPLACE_ADAPTER_CAPABILITY = UQCapability(
    supports_calibration=True,
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.LAPLACE,
    source_package="opifex",
    notes=(
        "LaplaceAdapterSpec — diagonal Laplace posterior approximation "
        "around a MAP point (MacKay 1992; Daxberger et al. "
        "arXiv:2106.14806). Curvature kernels vendored in "
        "opifex.uncertainty.curvature."
    ),
)


_MC_DROPOUT_ADAPTER_CAPABILITY = UQCapability(
    supports_calibration=True,
    native_nnx_module=True,
    default_strategy=DefaultStrategy.MC_DROPOUT,
    source_package="opifex",
    notes=(
        "MCDropoutAdapter — Gal & Ghahramani ICML 2016 approximate "
        "Bayesian inference via test-time dropout sampling. NNX module "
        "wrapper around a deterministic backbone with dropout layers."
    ),
)


_BAYESIAN_LAST_LAYER_ADAPTER_CAPABILITY = UQCapability(
    supports_calibration=True,
    native_nnx_module=True,
    default_strategy=DefaultStrategy.BAYESIAN_LAST_LAYER,
    source_package="opifex",
    notes=(
        "BayesianLastLayerAdapterSpec — Bayesian-only final layer "
        "(BayesianLinear) over a deterministic backbone."
    ),
)


_SNGP_ADAPTER_CAPABILITY = UQCapability(
    supports_calibration=True,
    supports_ood_detection=True,
    native_nnx_module=True,
    default_strategy=DefaultStrategy.SNGP,
    source_package="opifex",
    notes=(
        "SNGPAdapterSpec — Spectral-Normalized Neural Gaussian Process "
        "last layer (Liu et al. NeurIPS 2020) for distance-aware "
        "uncertainty and OOD detection."
    ),
)


_VBLL_ADAPTER_CAPABILITY = UQCapability(
    supports_calibration=True,
    native_nnx_module=True,
    default_strategy=DefaultStrategy.VBLL,
    source_package="opifex",
    notes=(
        "VBLLAdapterSpec — Variational Bayesian Last Layer (Harrison "
        "et al. NeurIPS 2023). Probabilistic last layer with a "
        "variational objective."
    ),
)


# ---------------------------------------------------------------------------
# Phase 4 ensemble-adapter specs.
# ---------------------------------------------------------------------------


_DEEP_ENSEMBLE_ADAPTER_CAPABILITY = UQCapability(
    supports_ensemble=True,
    supports_calibration=True,
    native_nnx_module=True,
    default_strategy=DefaultStrategy.ENSEMBLE,
    source_package="opifex",
    notes=(
        "DeepEnsembleAdapter — Lakshminarayanan et al. NeurIPS 2017. "
        "Trains M independent NNX members; predict over the ensemble "
        "mean and variance."
    ),
)


_SNAPSHOT_ENSEMBLE_ADAPTER_CAPABILITY = UQCapability(
    supports_ensemble=True,
    supports_calibration=True,
    native_nnx_module=True,
    default_strategy=DefaultStrategy.SNAPSHOT_ENSEMBLE,
    source_package="opifex",
    notes=(
        "SnapshotEnsembleAdapter — Huang et al. ICLR 2017. Cyclic "
        "LR schedule produces M snapshots from one training run."
    ),
)


_SWAG_ADAPTER_CAPABILITY = UQCapability(
    supports_ensemble=True,
    supports_calibration=True,
    native_nnx_module=True,
    default_strategy=DefaultStrategy.SWAG,
    source_package="opifex",
    notes=(
        "SWAGAdapter — Maddox et al. NeurIPS 2019. Stochastic "
        "Weight Averaging Gaussian posterior over weights."
    ),
)


_BATCH_ENSEMBLE_ADAPTER_CAPABILITY = UQCapability(
    supports_ensemble=True,
    supports_calibration=True,
    native_nnx_module=True,
    default_strategy=DefaultStrategy.BATCH_ENSEMBLE,
    source_package="opifex",
    notes=(
        "BatchEnsembleAdapter — Wen et al. ICLR 2020. Rank-1 "
        "ensemble perturbations shared across a single base network."
    ),
)


_DUE_ADAPTER_CAPABILITY = UQCapability(
    supports_calibration=True,
    supports_ood_detection=True,
    native_nnx_module=True,
    default_strategy=DefaultStrategy.DUE,
    source_package="opifex",
    notes=(
        "DUEAdapterSpec — Deterministic Uncertainty Estimation (van "
        "Amersfoort et al. ICML 2021). Deep kernel + spectral "
        "normalization for distance-aware uncertainty."
    ),
)


_TTA_ADAPTER_CAPABILITY = UQCapability(
    supports_calibration=True,
    native_nnx_module=True,
    default_strategy=DefaultStrategy.TEST_TIME_AUGMENTATION,
    source_package="opifex",
    notes=(
        "TestTimeAugmentationAdapterSpec — Wang et al. Neurocomputing "
        "2019. Average predictions across input augmentations at "
        "evaluation time."
    ),
)


# ---------------------------------------------------------------------------
# Phase 4 calibration/conformal calibrators — concrete Pattern (A)
# calibrators that take a fitted model and return a fitted state.
#
# ``source_package`` is ``"calibrax"`` where the underlying metric /
# statistic is reused (calibration metrics, RCPS Hoeffding statistic);
# ``"opifex"`` for the calibrators themselves (the API contract +
# fit/predict surface live in opifex).
# ---------------------------------------------------------------------------


_TEMPERATURE_SCALING_CAPABILITY = UQCapability(
    supports_calibration=True,
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.CALIBRATION,
    source_package="opifex",
    notes=(
        "TemperatureScaling — Guo et al. ICML 2017 temperature "
        "scaling for multiclass logits. Pure JAX optax-driven fit "
        "with a fitted-state container."
    ),
)


_SPLIT_CONFORMAL_REGRESSOR_CAPABILITY = UQCapability(
    supports_conformal=True,
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.CONFORMAL,
    source_package="opifex",
    notes=(
        "SplitConformalRegressor — absolute-residual split conformal "
        "regression (Vovk et al. 2005). Pure JAX scalar-quantile fit."
    ),
)


_CONFORMALIZED_QUANTILE_REGRESSOR_CAPABILITY = UQCapability(
    supports_conformal=True,
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.CONFORMAL,
    source_package="opifex",
    notes=(
        "ConformalizedQuantileRegressor — Romano et al. NeurIPS 2019 "
        "CQR. Pure JAX over a fitted quantile regressor."
    ),
)


_GROUPED_SPLIT_CONFORMAL_REGRESSOR_CAPABILITY = UQCapability(
    supports_conformal=True,
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.CONFORMAL,
    source_package="opifex",
    notes=(
        "GroupedSplitConformalRegressor — group-conditional split "
        "conformal for fairness / heterogeneous coverage."
    ),
)


_LAC_CONFORMAL_CLASSIFIER_CAPABILITY = UQCapability(
    supports_conformal=True,
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.CONFORMAL,
    source_package="opifex",
    notes=(
        "LACConformalClassifier — Least Ambiguous Classifier conformal "
        "set predictor (Sadinle et al. JASA 2019)."
    ),
)


_FIELD_SPLIT_CONFORMAL_REGRESSOR_CAPABILITY = UQCapability(
    supports_conformal=True,
    supports_function_space=True,
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.CONFORMAL,
    source_package="opifex",
    notes=(
        "FieldSplitConformalRegressor — function-space conformal "
        "intervals via L2 / Linf / H1 field-residual scores."
    ),
)


_RISK_CONTROLLER_CAPABILITY = UQCapability(
    supports_conformal=True,
    supports_selective_risk=True,
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.CONFORMAL,
    source_package="calibrax",
    notes=(
        "RiskControllerState — RCPS threshold selection with the "
        "Hoeffding upper bound (Bates et al. JMLR 2021). Uses "
        "calibrax.statistics.analyzer for the inner statistical "
        "primitives."
    ),
)


ADAPTER_CAPABILITIES: dict[str, UQCapability] = {
    # GP adapter specs (Task 6.3.4 inventory).
    "adapter:GPJaxAdapterSpec": _GPJAX_ADAPTER_CAPABILITY,
    "adapter:TinygpAdapterSpec": _TINYGP_ADAPTER_CAPABILITY,
    "adapter:MarkovflowAdapterSpec": _MARKOVFLOW_ADAPTER_CAPABILITY,
    "adapter:BayesnewtonAdapterSpec": _BAYESNEWTON_ADAPTER_CAPABILITY,
    "adapter:KalmanJaxAdapterSpec": _KALMAN_JAX_ADAPTER_CAPABILITY,
    # Phase 4 model-uncertainty adapter specs.
    "adapter:LaplaceAdapterSpec": _LAPLACE_ADAPTER_CAPABILITY,
    "adapter:MCDropoutAdapter": _MC_DROPOUT_ADAPTER_CAPABILITY,
    "adapter:BayesianLastLayerAdapterSpec": _BAYESIAN_LAST_LAYER_ADAPTER_CAPABILITY,
    "adapter:SNGPAdapterSpec": _SNGP_ADAPTER_CAPABILITY,
    "adapter:VBLLAdapterSpec": _VBLL_ADAPTER_CAPABILITY,
    # Phase 4 ensemble adapter specs.
    "adapter:DeepEnsembleAdapter": _DEEP_ENSEMBLE_ADAPTER_CAPABILITY,
    "adapter:SnapshotEnsembleAdapter": _SNAPSHOT_ENSEMBLE_ADAPTER_CAPABILITY,
    "adapter:SWAGAdapter": _SWAG_ADAPTER_CAPABILITY,
    "adapter:BatchEnsembleAdapter": _BATCH_ENSEMBLE_ADAPTER_CAPABILITY,
    "adapter:DUEAdapterSpec": _DUE_ADAPTER_CAPABILITY,
    "adapter:TestTimeAugmentationAdapterSpec": _TTA_ADAPTER_CAPABILITY,
    # Phase 4 calibration / conformal concrete calibrators.
    "calibration:TemperatureScaling": _TEMPERATURE_SCALING_CAPABILITY,
    "conformal:SplitConformalRegressor": _SPLIT_CONFORMAL_REGRESSOR_CAPABILITY,
    "conformal:ConformalizedQuantileRegressor": (_CONFORMALIZED_QUANTILE_REGRESSOR_CAPABILITY),
    "conformal:GroupedSplitConformalRegressor": (_GROUPED_SPLIT_CONFORMAL_REGRESSOR_CAPABILITY),
    "conformal:LACConformalClassifier": _LAC_CONFORMAL_CLASSIFIER_CAPABILITY,
    "conformal:FieldSplitConformalRegressor": (_FIELD_SPLIT_CONFORMAL_REGRESSOR_CAPABILITY),
    "conformal:RiskControllerState": _RISK_CONTROLLER_CAPABILITY,
}


__all__ = ["ADAPTER_CAPABILITIES"]

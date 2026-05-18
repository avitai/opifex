"""Authoritative Opifex uncertainty-quantification platform.

Core contracts:

* :mod:`opifex.uncertainty.types` — predictive-distribution / prediction-set /
  prediction-interval value objects, plus the :class:`PredictiveMode` enum.
* :mod:`opifex.uncertainty.objectives` — :class:`ObjectiveConfig` and
  :class:`UQLossComponents`; ``scale_kl`` helper.
* :mod:`opifex.uncertainty.kernels` — pure JAX Bayesian-kernel helpers
  (``diagonal_gaussian_kl`` delegates to Artifex for N(0,1) prior;
  ``sample_diagonal_gaussian``).
* :mod:`opifex.uncertainty.protocols` — structural UQ-aware module / variational
  / calibrator / conformalizer / estimator protocols.
* :mod:`opifex.uncertainty.registry` — :class:`UQCapability`,
  :class:`UQRegistry`, :class:`DefaultStrategy` (singleton registry extending
  CalibraX ``SingletonRegistry``).
* :mod:`opifex.uncertainty.inference_backends` — backend protocol + base
  result / spec / diagnostics containers.
* :mod:`opifex.uncertainty.adapters` — distribution / model-uncertainty adapter
  protocols + capability specs.
* :mod:`opifex.uncertainty.likelihoods` — backend-neutral Gaussian /
  heteroscedastic-Gaussian / Laplace / Student-t / mixture log-likelihood
  helpers, plus :class:`LikelihoodSpec`.
* :mod:`opifex.uncertainty.priors` — diagonal-Gaussian prior log density,
  plus :class:`PriorSpec`.

Additional sub-packages cover ``calibration``, ``conformal``, ``ood``,
``selective``, ``forecasting_metrics``, ``scientific``, ``assimilation``,
``sensitivity``, ``reliability``, ``surrogate``, ``monitoring``, ``pac_bayes``,
``sbi``, and ``active``.
"""

from opifex.uncertainty.adapters import (
    DistributionAdapterProtocol,
    DistributionAdapterSpec,
    ModelUncertaintyAdapterProtocol,
)
from opifex.uncertainty.distributions import (
    ArtifexDistributionAdapter,
    DistrAxAdapter,
    from_distribution,
)
from opifex.uncertainty.inference_backends import (
    BackendDiagnostics,
    BackendResult,
    BlackJAXBackend,
    InferenceBackendProtocol,
    InferenceBackendSpec,
    UnsupportedBackendError,
)
from opifex.uncertainty.kernels.bayesian import (
    diagonal_gaussian_kl,
    sample_diagonal_gaussian,
)
from opifex.uncertainty.layers.bayesian import (
    BayesianLinear,
    BayesianSpectralConvolution,
)
from opifex.uncertainty.likelihoods import (
    gaussian_log_likelihood,
    heteroscedastic_gaussian_log_likelihood,
    laplace_log_likelihood,
    LikelihoodSpec,
    mixture_log_likelihood,
    student_t_log_likelihood,
)
from opifex.uncertainty.objectives import (
    ObjectiveConfig,
    scale_kl,
    UQLossComponents,
)
from opifex.uncertainty.priors import (
    diagonal_gaussian_log_prior,
    PriorSpec,
)
from opifex.uncertainty.protocols import (
    Calibrator,
    Conformalizer,
    UncertaintyAwareModule,
    UncertaintyEstimator,
    VariationalModule,
)
from opifex.uncertainty.registry import (
    DefaultStrategy,
    register_uq_capability,
    UQCapability,
    UQRegistry,
)
from opifex.uncertainty.types import (
    PredictionInterval,
    PredictionSet,
    PredictiveDistribution,
    PredictiveMode,
)


__all__ = [
    "ArtifexDistributionAdapter",
    "BackendDiagnostics",
    "BackendResult",
    "BayesianLinear",
    "BayesianSpectralConvolution",
    "BlackJAXBackend",
    "Calibrator",
    "Conformalizer",
    "DefaultStrategy",
    "DistrAxAdapter",
    "DistributionAdapterProtocol",
    "DistributionAdapterSpec",
    "InferenceBackendProtocol",
    "InferenceBackendSpec",
    "LikelihoodSpec",
    "ModelUncertaintyAdapterProtocol",
    "ObjectiveConfig",
    "PredictionInterval",
    "PredictionSet",
    "PredictiveDistribution",
    "PredictiveMode",
    "PriorSpec",
    "UQCapability",
    "UQLossComponents",
    "UQRegistry",
    "UncertaintyAwareModule",
    "UncertaintyEstimator",
    "UnsupportedBackendError",
    "VariationalModule",
    "diagonal_gaussian_kl",
    "diagonal_gaussian_log_prior",
    "from_distribution",
    "gaussian_log_likelihood",
    "heteroscedastic_gaussian_log_likelihood",
    "laplace_log_likelihood",
    "mixture_log_likelihood",
    "register_uq_capability",
    "sample_diagonal_gaussian",
    "scale_kl",
    "student_t_log_likelihood",
]

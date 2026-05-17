"""Authoritative Opifex uncertainty-quantification platform.

Sub-packages and modules in this package replace the fragmented
``opifex.neural.bayesian`` / ``opifex.neural.operators.specialized.uqno`` /
``opifex.solvers.wrappers`` UQ surfaces with shared contracts.

Phase 1 contracts (complete):

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

Subsequent phases populate ``calibration``, ``conformal``, ``ood``,
``selective``, ``forecasting_metrics``, ``scientific``, ``assimilation``,
``sensitivity``, ``reliability``, ``surrogate``, ``monitoring``, ``pac_bayes``,
``sbi``, and ``active``.
"""

from opifex.uncertainty.layers.bayesian import (
    BayesianLinear,
    BayesianSpectralConvolution,
)
from opifex.uncertainty.objectives import (
    ObjectiveConfig,
    scale_kl,
    UQLossComponents,
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
    "BayesianLinear",
    "BayesianSpectralConvolution",
    "Calibrator",
    "Conformalizer",
    "DefaultStrategy",
    "ObjectiveConfig",
    "PredictionInterval",
    "PredictionSet",
    "PredictiveDistribution",
    "PredictiveMode",
    "UQCapability",
    "UQLossComponents",
    "UQRegistry",
    "UncertaintyAwareModule",
    "UncertaintyEstimator",
    "VariationalModule",
    "register_uq_capability",
    "scale_kl",
]

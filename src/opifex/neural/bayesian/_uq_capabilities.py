"""UQ capability declarations for the Bayesian neural model family (Task 7.2).

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.neural.bayesian.__init__`` so the two concrete neural surfaces
(``ProbabilisticPINN`` and ``MultiFidelityPINN``) get a capability
declaration that lands in the singleton :class:`UQRegistry` at first
import.

* ``ProbabilisticPINN`` — native Bayesian PINN (``BayesianLinear`` /
  ``BayesianSpectralConvolution`` layers + variational ELBO).
* ``MultiFidelityPINN`` — deterministic NNX baseline that composes
  ensemble / conformal / calibration adapters via
  :meth:`UQCapability.with_adapter`.
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


# Native variational Bayesian PINN. Owns a Bayesian posterior over its
# own layer weights and trains with a variational ELBO objective.
_PROBABILISTIC_PINN_CAPABILITY = UQCapability(
    native_bayesian=True,
    supports_calibration=True,
    default_strategy=DefaultStrategy.VARIATIONAL,
    native_nnx_module=True,
    notes=(
        "ProbabilisticPINN owns a mean-field variational posterior over "
        "its Bayesian layers (BayesianLinear / BayesianSpectralConvolution) "
        "and supports temperature / Platt / isotonic calibration on its "
        "predictive distribution."
    ),
)


# Multi-fidelity PINN is deterministic by default and admits the three
# adapter strategies via :meth:`UQCapability.with_adapter`, mirroring the
# Task 7.1 operator baseline.
_MULTI_FIDELITY_PINN_CAPABILITY = (
    UQCapability(
        default_strategy=DefaultStrategy.DETERMINISTIC,
        native_nnx_module=True,
        notes=(
            "MultiFidelityPINN composes per-fidelity deterministic PINN "
            "subnets; UQ comes from ensemble / conformal / calibration "
            "adapters layered on the trained model."
        ),
    )
    .with_adapter("ensemble")
    .with_adapter("conformal")
    .with_adapter("calibration")
)


BAYESIAN_MODEL_CAPABILITIES: dict[str, UQCapability] = {
    "model:ProbabilisticPINN": _PROBABILISTIC_PINN_CAPABILITY,
    "model:MultiFidelityPINN": _MULTI_FIDELITY_PINN_CAPABILITY,
}


__all__ = ["BAYESIAN_MODEL_CAPABILITIES"]

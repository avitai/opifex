"""Structural UQ protocols.

Five runtime-checkable protocols for UQ-aware surfaces. Structural typing only —
no inheritance required, no parallel implementation hierarchy created.

Sibling-package coverage check:

* CalibraX ``core/protocols.py`` — benchmark / dataset / metric protocols. No
  UQ surface match.
* Artifex ``models/base.py`` — ``GenerativeModelProtocol`` covers ``__call__`` +
  ``generate`` for *generation*, not for predictive distributions, KL, or
  ELBO. No UQ surface match.
* Datarax — no calibrator/conformalizer protocols.

Local protocols justified.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING


if TYPE_CHECKING:
    import jax
    from flax import nnx

    from opifex.uncertainty.objectives import ObjectiveConfig, UQLossComponents
    from opifex.uncertainty.types import PredictiveDistribution


@runtime_checkable
class UncertaintyAwareModule(Protocol):
    """Any model that can return a :class:`PredictiveDistribution` for an input.

    Stochastic models must accept caller-owned ``nnx.Rngs`` at the method
    boundary; deterministic models may ignore it.
    """

    def predict_distribution(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> PredictiveDistribution:
        """Return a predictive distribution for inputs ``x``."""
        ...


@runtime_checkable
class VariationalModule(UncertaintyAwareModule, Protocol):
    """Bayesian / variational extension of :class:`UncertaintyAwareModule`.

    Adds the KL divergence, optimizer-facing loss decomposition, and ELBO
    surfaces that Bayesian layers and PINN/UQNO models implement.
    """

    def kl_divergence(self) -> jax.Array:
        """Return the total KL divergence of variational parameters from the prior."""
        ...

    def loss_components(
        self,
        batch: Any,
        *,
        config: ObjectiveConfig,
        rngs: nnx.Rngs | None = None,
    ) -> UQLossComponents:
        """Return the per-component loss decomposition for one batch."""
        ...

    def negative_elbo(
        self,
        batch: Any,
        *,
        config: ObjectiveConfig,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """Return the scalar negative-ELBO objective for one batch."""
        ...


@runtime_checkable
class Calibrator(Protocol):
    """Post-hoc calibrator (temperature / Platt / isotonic / beta scaling).

    ``fit`` returns immutable fitted state (typically a small mapping or a
    ``@struct.dataclass``).
    """

    def fit(self, predictions: jax.Array, targets: jax.Array) -> Any:
        """Fit calibrator parameters and return immutable fitted state."""
        ...


@runtime_checkable
class Conformalizer(Protocol):
    """Conformal-prediction calibration surface.

    ``calibrate`` returns immutable fitted state with the conformal threshold
    and assumption metadata. ``alpha`` is the miscoverage level.
    """

    def calibrate(
        self,
        predictions: jax.Array,
        targets: jax.Array,
        *,
        alpha: float,
    ) -> Any:
        """Fit conformal threshold for miscoverage ``alpha`` and return fitted state."""
        ...


@runtime_checkable
class UncertaintyEstimator(Protocol):
    """End-to-end fit/predict UQ surface.

    Used by ensemble / dropout / deterministic-adapter strategies that need a
    minimal ``fit``/``predict_distribution`` contract without the heavier
    :class:`VariationalModule` KL/ELBO surface.
    """

    def fit(
        self,
        x: jax.Array,
        y: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Fit the estimator on ``(x, y)`` training data."""
        ...

    def predict_distribution(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> PredictiveDistribution:
        """Return a predictive distribution for inputs ``x``."""
        ...


__all__ = [
    "Calibrator",
    "Conformalizer",
    "UncertaintyAwareModule",
    "UncertaintyEstimator",
    "VariationalModule",
]

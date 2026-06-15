"""UQ capability declarations and registry.

Capability containers are scalar/string/enum-only frozen+slotted dataclasses,
hashable with no array data. Used for Python-level registry lookup; never
traced as pytree data.

``calibrax.core.registry.SingletonRegistry[T]`` is reused as the backing
mechanism. Opifex adds two domain-specific policies on top:

1. Duplicate-registration rejection (CalibraX overwrites silently; rejection
   is required because every capability declaration is canonical).
2. ``require()`` enhances ``get()``'s ``KeyError`` with the sorted list of
   available names so callers get an actionable error message.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar

from calibrax.core.registry import SingletonRegistry


if TYPE_CHECKING:
    from collections.abc import Callable


_C = TypeVar("_C", bound=type)


class DefaultStrategy(StrEnum):
    """Default UQ strategy advertised by a capability declaration.

    Adding a new member must NOT require changing any pre-existing capability
    declaration — every field on :class:`UQCapability` has a sensible default
    so new members compose additively.

    Members:

    * ``DETERMINISTIC`` — Point predictions only; no uncertainty surface.
    * ``BAYESIAN`` — Native Bayesian posterior over parameters (e.g., MCMC).
    * ``VARIATIONAL`` — Variational inference (mean-field VI / SVI) ELBO objective.
    * ``ENSEMBLE`` — Deep ensemble of independently trained members.
    * ``SNAPSHOT_ENSEMBLE`` — Cyclic-LR snapshot ensemble of one training run.
    * ``BATCH_ENSEMBLE`` — Wen et al. BatchEnsemble (rank-1 perturbations).
    * ``MC_DROPOUT`` — Approximate Bayesian inference via test-time dropout sampling.
    * ``BAYESIAN_LAST_LAYER`` — Bayesian-only on the final layer of an otherwise
      deterministic backbone.
    * ``VBLL`` — Variational Bayesian Last Layer (probabilistic last layer only).
    * ``LAPLACE`` — Laplace approximation of the posterior around a MAP/MLE point.
    * ``SNGP`` — Spectral-Normalized Neural Gaussian Process last layer.
    * ``SWAG`` — Stochastic Weight Averaging Gaussian (posterior over weights).
    * ``DUE`` — Deterministic Uncertainty Estimation (deep kernel + spectral
      normalization).
    * ``TEST_TIME_AUGMENTATION`` — TTA averaging predictions across input
      augmentations.
    * ``CONFORMAL`` — Conformal prediction sets/intervals with finite-sample coverage.
    * ``CALIBRATION`` — Post-hoc calibrator (temperature / Platt / isotonic / beta).
    * ``LIKELIHOOD_FREE_SBI`` — Simulation-based inference (NPE / NLE / NRE).
    * ``ACTIVE_LEARNING`` — Acquisition-driven sample selection (BALD / EIG / residual).
    * ``PAC_BAYES`` — PAC-Bayes generalization-bound certificate.
    * ``POLYNOMIAL_CHAOS`` — Polynomial chaos expansion of input-driven uncertainty.
    * ``KARHUNEN_LOEVE`` — Karhunen-Loève expansion of stochastic input fields.
    * ``STOCHASTIC_GALERKIN`` — Stochastic Galerkin / collocation surrogate.
    * ``PROBABILISTIC_NUMERICS`` — Solver-level numerical uncertainty (probdiffeq-style).
    * ``UNSUPPORTED`` — Explicit unsupported placeholder; reuse blocks against it.
    """

    DETERMINISTIC = "deterministic"
    BAYESIAN = "bayesian"
    VARIATIONAL = "variational"
    ENSEMBLE = "ensemble"
    SNAPSHOT_ENSEMBLE = "snapshot_ensemble"
    BATCH_ENSEMBLE = "batch_ensemble"
    MC_DROPOUT = "mc_dropout"
    BAYESIAN_LAST_LAYER = "bayesian_last_layer"
    VBLL = "vbll"
    LAPLACE = "laplace"
    SNGP = "sngp"
    SWAG = "swag"
    DUE = "due"
    TEST_TIME_AUGMENTATION = "test_time_augmentation"
    CONFORMAL = "conformal"
    CALIBRATION = "calibration"
    LIKELIHOOD_FREE_SBI = "likelihood_free_sbi"
    ACTIVE_LEARNING = "active_learning"
    PAC_BAYES = "pac_bayes"
    POLYNOMIAL_CHAOS = "polynomial_chaos"
    KARHUNEN_LOEVE = "karhunen_loeve"
    STOCHASTIC_GALERKIN = "stochastic_galerkin"
    PROBABILISTIC_NUMERICS = "probabilistic_numerics"
    UNSUPPORTED = "unsupported"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class UQCapability:
    """Static declaration of one UQ-bearing surface's capabilities.

    All fields are scalars / strings / enums; the dataclass is hashable; it
    is passed by Python-level registry lookup and never consumed by jit'd code
    as pytree data.

    Fields fall into three groups:

    **Capability flags** (``native_*``, ``supports_*``) — the surface offers
    this UQ behavior end-to-end, without an external adapter:

    * ``native_bayesian`` — owns a Bayesian posterior over its own parameters
      (``BayesianLinear``-style modules, MCMC backends).
    * ``native_distributional`` — outputs a parametric predictive distribution
      directly (heteroscedastic Gaussian regressor, mixture density network).
    * ``supports_ensemble`` — can be wrapped by ``DeepEnsembleAdapter`` /
      ``SnapshotEnsembleAdapter`` / ``SWAGAdapter`` without changing its API.
    * ``supports_conformal`` — conformal calibrators (split / CQR / RAPS /
      Top-K) work against this surface.
    * ``supports_calibration`` — post-hoc calibrators (temperature / Platt /
      isotonic / beta) work against this surface.
    * ``supports_function_space`` — predicts over function/field outputs
      (UQNO, field metrics, stochastic-field inputs).
    * ``supports_solver_uncertainty`` — wraps a numerical solver and propagates
      solver-level numerical uncertainty (``SolutionDistribution``).
    * ``supports_ood_detection`` — exposes an OOD score per input.
    * ``supports_selective_risk`` — supports an abstention / risk-coverage
      curve for selective prediction.
    * ``supports_likelihood_free`` — usable as an inverse-problem target for
      SBI (NPE / NLE / NRE).
    * ``supports_active_learning`` — acquisition functions (BALD / BatchBALD /
      EIG / PINN-residual) can rank candidates against it.
    * ``supports_pac_bayes_certificate`` — PAC-Bayes bounds can be computed
      for this surface.
    * ``supports_stochastic_field_input`` — accepts a KLE/PCE-parameterized
      random-field input (stochastic-Galerkin / collocation).

    **Backend-stack flags** (``native_jax_kernel`` / ``native_nnx_module`` /
    ``requires_graph_adapter``) — distinguish where the implementation lives:

    * ``native_jax_kernel`` — pure JAX function over arrays; safe to call
      inside ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` directly.
    * ``native_nnx_module`` — Flax NNX ``Module`` that owns parameters /
      state; calls go through ``nnx.jit`` / ``nnx.grad`` / ``nnx.vmap``.
    * ``requires_graph_adapter`` — needs ``nnx.split`` / ``nnx.merge`` to
      cross a raw-JAX transform boundary; setting this to ``True`` requires
      non-empty :attr:`notes` explaining why (enforced in ``__post_init__``).

    **Provenance fields** — for honest registry display:

    * ``default_strategy`` — :class:`DefaultStrategy` enum value advertising
      what kind of UQ this surface delivers by default.
    * ``source_package`` — package name owning the implementation
      (``"opifex"``, ``"artifex"``, ``"calibrax"``, ``"datarax"``, etc.) so
      capability docs don't mislabel sibling-backed primitives as
      Opifex-local.
    * ``notes`` — free-text rationale (required when
      ``requires_graph_adapter=True``; optional otherwise).
    """

    native_bayesian: bool = False
    native_distributional: bool = False
    supports_ensemble: bool = False
    supports_conformal: bool = False
    supports_calibration: bool = False
    supports_function_space: bool = False
    supports_solver_uncertainty: bool = False
    supports_ood_detection: bool = False
    supports_selective_risk: bool = False
    supports_likelihood_free: bool = False
    supports_active_learning: bool = False
    supports_pac_bayes_certificate: bool = False
    supports_stochastic_field_input: bool = False
    native_jax_kernel: bool = False
    native_nnx_module: bool = False
    requires_graph_adapter: bool = False
    default_strategy: DefaultStrategy = DefaultStrategy.DETERMINISTIC
    source_package: str = "opifex"
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate capability invariants (graph-adapter rationale, native-kernel flag)."""
        if self.requires_graph_adapter and not self.notes:
            raise ValueError(
                "requires_graph_adapter=True must be paired with non-empty notes "
                "explaining why the graph adapter is needed."
            )
        if self.native_jax_kernel and self.requires_graph_adapter:
            raise ValueError(
                "Contradictory capability: native_jax_kernel=True implies a pure "
                "JAX kernel safe for direct transforms, which cannot also "
                "require requires_graph_adapter=True (an NNX graph/state "
                "adapter). Pick one."
            )

    @classmethod
    def deterministic_baseline(cls) -> UQCapability:
        """Return an all-flags-false baseline declaration for deterministic models."""
        return cls(default_strategy=DefaultStrategy.DETERMINISTIC)


class UQRegistry(SingletonRegistry["UQCapability"]):
    """Singleton registry mapping capability-name → :class:`UQCapability`.

    Extends CalibraX's :class:`SingletonRegistry` with two Opifex-specific
    policies: duplicate-rejection and an enhanced ``require`` error message.
    """

    def register(self, name: str, item: UQCapability) -> None:
        """Register a capability, rejecting duplicate names with ``ValueError``."""
        if name in self:
            raise ValueError(
                f"Capability {name!r} is already registered. Capability "
                "declarations are canonical — call require()/get() to read, or "
                "use a new name."
            )
        super().register(name, item)

    def require(self, name: str) -> UQCapability:
        """Return the registered capability or raise ``KeyError`` with available names."""
        if name not in self:
            available = sorted(self.list_names())
            raise KeyError(f"Capability {name!r} not registered. Available: {available!r}.")
        return self.get(name)


def register_uq_capability(name: str, capability: UQCapability) -> Callable[[_C], _C]:
    """Class decorator that registers ``capability`` under ``name`` in :class:`UQRegistry`.

    Mirrors CalibraX's ``register_benchmark`` decorator pattern so capability
    declaration tests can reuse the same decorator surface.
    """

    def decorator(cls: _C) -> _C:
        registry = UQRegistry()
        registry.register(name, capability)
        return cls

    return decorator


__all__ = [
    "DefaultStrategy",
    "UQCapability",
    "UQRegistry",
    "register_uq_capability",
]

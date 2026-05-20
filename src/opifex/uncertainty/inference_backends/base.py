"""Inference-backend protocol + base result/spec containers.

Container patterns:

* :class:`BackendDiagnostics` —
  ``@flax.struct.dataclass(slots=True, kw_only=True)``. Carries array
  diagnostic statistics (ESS, R-hat, acceptance rate, divergences) through
  ``jit``/``vmap``. Missing diagnostics are ``None`` — Bayes-vs-non-Bayes
  samplers expose different surfaces.
* :class:`BackendResult` — ``@flax.struct.dataclass``. Carries the backend's
  typed sampler-state (Artifex ``BlackJAXSamplerState`` reused directly) plus
  the :class:`BackendDiagnostics`. The ``sampler_state`` field is opaque
  (``Any``) so any Artifex / future-backend state object can flow unchanged.
* :class:`InferenceBackendSpec` —
  ``@dataclass(frozen=True, slots=True, kw_only=True)``. Scalar/string/tuple
  capability metadata; hashable; used by registries and the backend router as
  a static argument.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

from flax import nnx, struct


if TYPE_CHECKING:
    import jax

    from opifex.uncertainty.types import PredictiveDistribution


class UnsupportedBackendError(Exception):
    """Raised when an inference backend is requested but unavailable.

    Use this error type for any of:

    * optional dependency not installed,
    * requested sampler family not implemented,
    * malformed adapter routing arguments.

    The message MUST identify the backend by name (the backend router
    guarantees this).
    """

    def __init__(self, backend_name: str, *, reason: str) -> None:
        super().__init__(f"Inference backend {backend_name!r} unavailable: {reason}")
        self.backend_name = backend_name
        self.reason = reason


@struct.dataclass(slots=True, kw_only=True)
class BackendDiagnostics:
    """Typed sampler diagnostics shared across MCMC and VI families.

    Fields are ``None`` when a given sampler does not produce them — e.g.,
    ADVI has no acceptance rate, HMC has no tree depth, SMC has no R-hat.

    Fields:

    * ``ess`` — effective sample size per parameter (MCMC mixing diagnostic).
    * ``rhat`` — Gelman-Rubin ``R-hat`` between-chain / within-chain variance
      ratio (convergence diagnostic; values near 1.0 indicate convergence).
    * ``acceptance_rate`` — Metropolis acceptance probability for
      HMC / NUTS / MALA (typical healthy range 0.6–0.9).
    * ``divergences`` — count of divergent transitions in NUTS / HMC (large
      counts indicate pathological posterior geometry).
    * ``step_size`` — adaptive leapfrog step size at the end of warmup
      (HMC / NUTS).
    * ``tree_depth`` — maximum NUTS tree depth reached (clipping at the
      tree-depth cap indicates the sampler is hitting the U-turn from
      the cap rather than from the no-U-turn condition).
    """

    ess: jax.Array | None = None
    rhat: jax.Array | None = None
    acceptance_rate: jax.Array | None = None
    divergences: jax.Array | None = None
    step_size: jax.Array | None = None
    tree_depth: jax.Array | None = None


@struct.dataclass(slots=True, kw_only=True)
class BackendResult:
    """Result wrapper produced by :meth:`InferenceBackendProtocol.fit`.

    ``sampler_state`` is opaque to keep Artifex ``BlackJAXSamplerState`` (and
    other backends' state types) compatible without further wrapping.
    """

    sampler_state: Any
    diagnostics: BackendDiagnostics = struct.field(
        pytree_node=True,
        default_factory=BackendDiagnostics,
    )


@runtime_checkable
class InferenceBackendProtocol(Protocol):
    """Common surface for MCMC / VI / Laplace / Pathfinder / ADVI backends.

    Concrete implementations include BlackJAX (MCMC), PAC-Bayes, and SBI
    backends. Each implementation MUST accept caller-owned ``nnx.Rngs`` for
    every stochastic call.
    """

    def fit(self, target_log_prob: Any, *, rngs: nnx.Rngs) -> BackendResult:
        """Run inference; return typed backend state + diagnostics."""
        ...

    def predict_distribution(self, x: jax.Array, *, rngs: nnx.Rngs) -> PredictiveDistribution:
        """Return a predictive distribution for inputs ``x``."""
        ...

    def posterior_predictive(self, rngs: nnx.Rngs, x: jax.Array) -> PredictiveDistribution:
        """Return a posterior-predictive distribution sample for inputs ``x``.

        Distinct from :meth:`predict_distribution` (which may marginalize
        differently).
        """
        ...


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class InferenceBackendSpec:
    """Static capability declaration for an inference backend.

    Hashable, no array data, passed as a static arg to the backend router.
    Sequence fields are tuples.
    """

    name: str
    family: str
    sampler_names: tuple[str, ...]
    source_package: str

    def __post_init__(self) -> None:
        if not isinstance(self.sampler_names, tuple):
            raise TypeError(
                f"sampler_names must be a tuple, got {type(self.sampler_names).__name__}. "
                "Static / aux-data fields must be hashable."
            )
        if not self.name:
            raise ValueError("InferenceBackendSpec.name must be non-empty.")


__all__ = [
    "BackendDiagnostics",
    "BackendResult",
    "InferenceBackendProtocol",
    "InferenceBackendSpec",
    "UnsupportedBackendError",
]

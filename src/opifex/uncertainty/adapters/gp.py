"""Gaussian-process adapter specs.

Five Pattern-A frozen dataclasses declaring metadata for GP backends:

* :class:`GPJaxAdapterSpec` — user-installed; 9 family tags.
* :class:`TinygpAdapterSpec` — user-installed; recommended substrate for
  the future LUNO implementation.
* :class:`MarkovflowAdapterSpec` — metadata-only; algorithms vendored
  into :mod:`opifex.uncertainty.statespace`.
* :class:`BayesnewtonAdapterSpec` — metadata-only; Kalman / kernel
  algorithms vendored into :mod:`opifex.uncertainty.statespace`.
* :class:`KalmanJaxAdapterSpec` — metadata-only with
  ``DeprecationWarning`` pointing at :class:`BayesnewtonAdapterSpec`
  (per kalman-jax's own README:1).

References
----------
* Pinder, T. & Dodd, D. 2022 — *GPJax: a Gaussian process framework in
  JAX*. arXiv:2208.05459.
* Foreman-Mackey, D. et al. 2017 — *Fast and Scalable GP Inference in
  ``celerite``* (tinygp ancestry).
* Wilkinson, W. J. et al. 2023 — *Bayes-Newton Methods for Approximate
  Bayesian Inference with PSD Guarantees*. JMLR 24(229).
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _GPAdapterSpecBase:
    """Shared shape for GP adapter specs.

    Each subclass declares the strategy, source package, required
    capabilities, the supported GP family tags, free-text notes, and
    overrides ``wrap`` only when the backend can be wired immediately.
    """

    default_strategy: DefaultStrategy = DefaultStrategy.GAUSSIAN_PROCESS
    source_package: str = "opifex"
    required_capabilities: tuple[str, ...] = ()
    family_tags: tuple[str, ...] = ()
    notes: str = ""

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Raise :class:`NotImplementedError` until the backend lands."""
        del model, capability
        raise NotImplementedError(
            f"GP adapter strategy {self.default_strategy.value!r} is not "
            f"yet wired (source_package={self.source_package!r}). "
            f"Required capabilities: {self.required_capabilities!r}. "
            f"Family tags: {self.family_tags!r}."
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class GPJaxAdapterSpec(_GPAdapterSpecBase):
    """User-installed GPJax backend.

    GPJax 0.14 dropped ``flax.nnx`` in favour of ``equinox.Module``
    (`gpjax/docs/migration.md:5-9,35-41`). Users who want to wire GPJax
    into NNX-native Bayesian-PINN / UQ-NO surfaces must cross an
    ``eqx.Module ↔ nnx.Module`` PyTree boundary — opifex does not
    provide this bridge.
    """

    default_strategy: DefaultStrategy = DefaultStrategy.GAUSSIAN_PROCESS
    source_package: str = "gpjax"
    required_capabilities: tuple[str, ...] = ("native_jax",)
    family_tags: tuple[str, ...] = (
        "exact_gp",
        "conjugate_gaussian",
        "svgp",
        "non_conjugate",
        "multi_output",
        "deep_kernel",
        "stochastic_variational",
        "natural_gradient",
        "rff_approximation",
    )
    notes: str = (
        "User-installed via `uv pip install gpjax`. GPJax 0.14+ uses "
        "equinox.Module; crossing eqx<->nnx requires a user-provided "
        "PyTree bridge."
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TinygpAdapterSpec(_GPAdapterSpecBase):
    """User-installed tinygp backend (recommended LUNO substrate)."""

    default_strategy: DefaultStrategy = DefaultStrategy.GAUSSIAN_PROCESS
    source_package: str = "tinygp"
    required_capabilities: tuple[str, ...] = ("native_jax",)
    family_tags: tuple[str, ...] = (
        "exact_gp",
        "conjugate_gaussian",
        "stationary_kernel",
        "quasisep_1d_state_space",
    )
    notes: str = (
        "User-installed via `uv pip install tinygp`. Recommended "
        "substrate for the future LUNO (Laplace UQ Neural Operator) "
        "task — cleaner Transform pattern for linearised-features "
        "kernels and lighter footprint than GPJax."
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MarkovflowAdapterSpec(_GPAdapterSpecBase):
    """Metadata-only adapter for the TensorFlow-based markovflow library.

    Specific algorithms (banded-precision Cholesky, SDE→linearize) are
    vendored under :mod:`opifex.uncertainty.statespace` (or Task 6.7
    assimilation) when they prove useful — markovflow itself is
    TF-based and not directly importable from opifex.
    """

    default_strategy: DefaultStrategy = DefaultStrategy.STATE_SPACE_FILTERING
    source_package: str = "markovflow"
    required_capabilities: tuple[str, ...] = ()
    family_tags: tuple[str, ...] = (
        "state_space_gp",
        "banded_precision_cholesky",
        "sde_linearize",
    )
    notes: str = (
        "Metadata-only — algorithms are vendored into "
        "opifex.uncertainty.statespace. markovflow is TF-based and "
        "not directly importable here."
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BayesnewtonAdapterSpec(_GPAdapterSpecBase):
    """Metadata-only adapter for bayesnewton's state-space GP algorithms.

    Specific algorithms (sequential and parallel Kalman filter /
    smoother, kernel ``state_transition`` closed forms for Matern /
    Periodic / Cosine / QuasiPeriodicMatern12) are vendored into
    :mod:`opifex.uncertainty.statespace`. The bayesnewton package
    itself is not importable because its pinned ``jax==0.4.14`` +
    ``objax`` stack conflicts with the opifex JAX baseline.
    """

    default_strategy: DefaultStrategy = DefaultStrategy.STATE_SPACE_FILTERING
    source_package: str = "bayesnewton"
    required_capabilities: tuple[str, ...] = ()
    family_tags: tuple[str, ...] = (
        "sequential_kalman",
        "parallel_kalman",
        "state_space_kernel",
    )
    notes: str = (
        "Metadata-only — sequential and parallel-scan Kalman primitives "
        "plus Matern / Cosine / Periodic state-space kernels are "
        "vendored into opifex.uncertainty.statespace citing "
        "bayesnewton/bayesnewton/ops.py and kernels.py."
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class KalmanJaxAdapterSpec(_GPAdapterSpecBase):
    """Deprecated metadata-only adapter for the kalman-jax package.

    kalman-jax's own ``README.md`` (line 1) states that bayesnewton is
    the official successor. The generic ``expm(F * dt)`` LTI-SDE
    discretization (``priors.py:46`` — the sole published reference for
    the generic case) is vendored into
    :func:`opifex.uncertainty.statespace.discretize_lti_sde`.

    Emits a :class:`DeprecationWarning` at construction pointing users
    at :class:`BayesnewtonAdapterSpec`.
    """

    default_strategy: DefaultStrategy = DefaultStrategy.STATE_SPACE_FILTERING
    source_package: str = "kalman-jax"
    required_capabilities: tuple[str, ...] = ()
    family_tags: tuple[str, ...] = ("lti_sde_discretization",)
    notes: str = (
        "Deprecated — see BayesnewtonAdapterSpec. The generic "
        "expm(F*dt) LTI-SDE discretization is vendored into "
        "opifex.uncertainty.statespace.discretize_lti_sde."
    )

    def __post_init__(self) -> None:
        """Emit a ``DeprecationWarning`` pointing at the bayesnewton successor."""
        warnings.warn(
            "KalmanJaxAdapterSpec is deprecated; use BayesnewtonAdapterSpec. "
            "kalman-jax is officially obsolete (see kalman-jax/README.md:1).",
            DeprecationWarning,
            stacklevel=2,
        )


__all__ = [
    "BayesnewtonAdapterSpec",
    "GPJaxAdapterSpec",
    "KalmanJaxAdapterSpec",
    "MarkovflowAdapterSpec",
    "TinygpAdapterSpec",
]

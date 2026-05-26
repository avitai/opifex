"""Adapter specs for advanced Bayesian-quadrature methods.

Pattern-A frozen dataclasses declaring metadata for the named BQ
algorithms enumerated in the design notes:

* :class:`VanillaBayesianQuadratureAdapterSpec` — GP-prior Bayesian
  quadrature with closed-form posterior mean / variance.
* :class:`WSABILAdapterSpec` — Warped Sequential Active Bayesian
  Integration (linear). Coexists with vanilla BQ in
  ``bayesian_quadrature.py`` per the design notes.
* :class:`SOBERAdapterSpec` — Stein-thinned discrepancy Bayesian
  quadrature (point-set quadrature). Split into a separate file per
  design (the SOBER ↔ FFBQ separation).
* :class:`FFBQAdapterSpec` — Frequency-domain Bayesian quadrature.
* :class:`EmukitQuadratureAdapterSpec` — read-only adapter pointing at
  the emukit (NumPy) baselines for benchmarking — emukit is vendored,
  not user-installed, per the design unification (fix #231).

References
----------
* Gunter, T. et al. 2014 — *Sampling for Inference in Probabilistic
  Models with Fast Bayesian Quadrature*, NeurIPS. (WSABI-L)
* Adachi, M. et al. 2023 — *SOBER: Highly Parallel Bayesian
  Optimization and Bayesian Quadrature over Discrete and Mixed Spaces*,
  arXiv:2301.11832. (SOBER)
* Briol, F.-X. et al. 2019 — *Probabilistic Integration*, Statistical
  Science 34(1). (BQ overview)
"""

from __future__ import annotations

import dataclasses
from typing import Any

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _BQAdapterSpecBase:
    """Shared shape for Bayesian-quadrature adapter specs."""

    default_strategy: DefaultStrategy = DefaultStrategy.BAYESIAN_QUADRATURE
    source_package: str = "opifex"
    required_capabilities: tuple[str, ...] = ()
    family_tags: tuple[str, ...] = ()
    notes: str = ""

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Raise :class:`NotImplementedError` until the backend lands."""
        del model, capability
        raise NotImplementedError(
            f"BQ adapter strategy {self.default_strategy.value!r} "
            f"({type(self).__name__}) is not yet wired "
            f"(source_package={self.source_package!r}). Family tags: "
            f"{self.family_tags!r}."
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class VanillaBayesianQuadratureAdapterSpec(_BQAdapterSpecBase):
    """GP-prior Bayesian quadrature with closed-form mean and variance."""

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = ("gp_prior", "closed_form_posterior")
    notes: str = (
        "Vanilla BQ — uses a Gaussian-process prior over the integrand "
        "and closed-form formulas for the posterior integral mean and "
        "variance. Coexists with WSABI-L in opifex's "
        "bayesian_quadrature.py module."
    )

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Return the JAX-native closed-form Vanilla BQ callable."""
        from opifex.uncertainty.quadrature.bayesian_quadrature import (
            vanilla_bayesian_quadrature,
        )

        del model, capability
        return vanilla_bayesian_quadrature


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class WSABILAdapterSpec(_BQAdapterSpecBase):
    """Warped Sequential Active Bayesian Integration — linear approximation."""

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = ("warped_gp_prior", "linear_warp", "sequential_active")
    notes: str = (
        "WSABI-L (Gunter+ 2014). Linear approximation of the "
        "warped-GP posterior moments for non-negative integrands. "
        "Coexists with VanillaBayesianQuadratureAdapterSpec in opifex's "
        "bayesian_quadrature.py module."
    )

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Return the JAX-native WSABI-L bounded-BQ mean callable."""
        from opifex.uncertainty.quadrature.bayesian_quadrature import (
            wsabi_l_bayesian_quadrature,
        )

        del model, capability
        return wsabi_l_bayesian_quadrature


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SOBERAdapterSpec(_BQAdapterSpecBase):
    """Kernel-recombination Bayesian quadrature (point-set method)."""

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = ("point_set", "kernel_recombination", "discrete_mixed")
    notes: str = (
        "SOBER (Adachi+ 2022 NeurIPS arXiv:2206.04734 + 2023 TMLR "
        "arXiv:2301.11832). Kernel-recombination via "
        "Tchernychova-Lyons CAR + Nyström low-rank approximation; "
        "vendored in sober.py per the SOBER ↔ FFBQ design split."
    )

    def wrap(self, model: Any, capability: UQCapability) -> Any:
        """Return the JAX-native SOBER kernel-recombination callable."""
        from opifex.uncertainty.quadrature.sober import sober_kernel_recombination

        del model, capability
        return sober_kernel_recombination


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FFBQAdapterSpec(_BQAdapterSpecBase):
    """Frequency-domain Bayesian quadrature."""

    source_package: str = "opifex"
    family_tags: tuple[str, ...] = ("frequency_domain", "fourier_features")
    notes: str = (
        "FFBQ (frequency-domain Bayesian quadrature). Lives in a "
        "separate ffbq.py module per the SOBER ↔ FFBQ design split."
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EmukitQuadratureAdapterSpec(_BQAdapterSpecBase):
    """Read-only adapter pointing at the emukit (NumPy) BQ baselines.

    emukit is vendored as a reference implementation, not installed as
    a runtime dependency. Use this spec for benchmarking; concrete
    integration drives through opifex's vanilla / WSABI-L
    JAX-native implementations.
    """

    source_package: str = "emukit"
    family_tags: tuple[str, ...] = ("reference_baseline", "numpy_only")
    notes: str = (
        "Metadata-only — emukit is the NumPy reference implementation. "
        "Vendored under opifex.uncertainty.quadrature; not a runtime "
        "dependency. See ../emukit/emukit/quadrature/* for the source."
    )


__all__ = [
    "EmukitQuadratureAdapterSpec",
    "FFBQAdapterSpec",
    "SOBERAdapterSpec",
    "VanillaBayesianQuadratureAdapterSpec",
    "WSABILAdapterSpec",
]

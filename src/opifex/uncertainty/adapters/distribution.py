"""Distribution adapter specs — normalizing-flow + Distrax-family backends.

Phase 9 final-validation (``09-phase-final-validation.md:780-782``)
requires:

> ``bijx`` is declared the preferred NNX-native normalizing-flow
> backend in the distribution adapter registry; FlowJAX and
> ``distrax`` are alternative backends behind the same protocol.

This module provides the three required spec classes. Each follows
the pattern established by :class:`opifex.uncertainty.adapters.gp._GPAdapterSpecBase`
— a frozen-slotted-kw-only dataclass with an installation probe and
``family_tags`` advertising the supported distribution / flow
constructions.

The adapter specs are **declarations only**: the spec classes do not
ship a JAX-native re-implementation of the underlying library. Users
who want to wire one of these backends into the opifex distribution
protocol must install the corresponding package themselves
(``uv pip install bijx`` / ``flowjax`` / ``distrax``).

References
----------
* Hoogeboom, Cohen, Tomczak 2021+ — ``bijx`` normalizing-flow library
  (NNX-native). https://github.com/bijx
* Ward 2023 — ``flowjax``: normalizing flows in JAX.
  https://github.com/danielward27/flowjax
* DeepMind 2020+ — ``distrax``: TFP-on-JAX distribution + bijector
  library. https://github.com/google-deepmind/distrax
"""

from __future__ import annotations

import dataclasses
import importlib.util


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _DistributionAdapterSpecBase:
    """Shared metadata template for user-installed distribution backends.

    Concrete subclasses set ``source_package``, ``family_tags`` and
    ``notes`` at the class level. Subclass instances are hashable
    capability descriptors that the distribution-adapter registry
    consumes — they do not own any algorithmic state.
    """

    source_package: str = "opifex"
    required_capabilities: tuple[str, ...] = ("native_jax",)
    family_tags: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def is_installed(cls) -> bool:
        """Return ``True`` iff the underlying package is importable."""
        package_name = cls._package_name()
        return importlib.util.find_spec(package_name) is not None

    @classmethod
    def _package_name(cls) -> str:
        """Subclasses override to name their importable package."""
        raise NotImplementedError


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BijxAdapterSpec(_DistributionAdapterSpecBase):
    """User-installed ``bijx`` backend (preferred NNX-native NF backend).

    Phase 9 line 780-782 declares ``bijx`` as the preferred NNX-native
    normalizing-flow backend. The spec advertises this preference to
    the registry router so consumers requesting an NF backend without
    a specific package preference resolve to ``bijx`` first.
    """

    source_package: str = "bijx"
    family_tags: tuple[str, ...] = (
        "normalizing_flow",
        "nnx_native",
        "preferred_nf",
        "coupling_flow",
        "spline_flow",
        "autoregressive_flow",
    )
    notes: str = (
        "User-installed via `uv pip install bijx`. Preferred NNX-"
        "native normalizing-flow backend per Phase 9 :780-782; "
        "FlowJAX / distrax serve as alternative resolutions behind "
        "the same DistributionAdapterProtocol."
    )

    @classmethod
    def _package_name(cls) -> str:
        return "bijx"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FlowJAXAdapterSpec(_DistributionAdapterSpecBase):
    """User-installed ``flowjax`` backend (alternative NF backend).

    Equinox-based normalizing-flow library. Lighter weight than bijx
    in some workflows; suitable when the user already builds with
    ``equinox.Module`` rather than ``flax.nnx.Module``. Crossing the
    eqx ↔ nnx boundary requires a user-provided PyTree bridge.
    """

    source_package: str = "flowjax"
    family_tags: tuple[str, ...] = (
        "normalizing_flow",
        "equinox_native",
        "coupling_flow",
        "spline_flow",
        "masked_autoregressive_flow",
        "real_nvp",
    )
    notes: str = (
        "User-installed via `uv pip install flowjax`. Equinox-based "
        "NF library; alternative backend behind the same "
        "DistributionAdapterProtocol as `bijx` / `distrax`."
    )

    @classmethod
    def _package_name(cls) -> str:
        return "flowjax"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DistraxAdapterSpec(_DistributionAdapterSpecBase):
    """User-installed ``distrax`` backend (TFP-on-JAX distribution + bijector).

    DeepMind's distrax provides a Distrax-shaped distribution +
    bijector library on top of JAX. It is the de-facto target for
    legacy TFP-substrate workflows and is the fallback resolution for
    the distribution protocol when bijx / flowjax are unavailable.
    """

    source_package: str = "distrax"
    family_tags: tuple[str, ...] = (
        "distribution",
        "bijector",
        "tfp_substrate",
        "normalizing_flow",
        "exponential_family",
    )
    notes: str = (
        "User-installed via `uv pip install distrax`. TFP-on-JAX "
        "distribution + bijector library; fallback distribution "
        "backend when `bijx` / `flowjax` are unavailable."
    )

    @classmethod
    def _package_name(cls) -> str:
        return "distrax"


__all__ = [
    "BijxAdapterSpec",
    "DistraxAdapterSpec",
    "FlowJAXAdapterSpec",
]

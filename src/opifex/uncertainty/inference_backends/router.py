"""Backend router over flow / sampler / distribution adapter specs.

Selects the highest-priority available backend for a given family using
**Artifex-first** resolution order. Optional backends remain available by
name but raise :class:`ImportError` on instantiation when not installed.

Usage:

.. code-block:: python

    from opifex.uncertainty.inference_backends.router import BackendRouter

    router = BackendRouter()
    sampler_spec = router.resolve("sampler")                  # default = BlackJAX
    flow_spec = router.resolve("flow", name="bijx")           # specific backend
    dist_spec = router.resolve("distribution")                # default = Artifex
"""

from __future__ import annotations

from opifex.uncertainty.inference_backends.base import InferenceBackendSpec
from opifex.uncertainty.inference_backends.blackjax import BLACKJAX_BACKEND_SPEC
from opifex.uncertainty.inference_backends.optional import (
    ARTIFEX_FLOW_SPECS,
    DISTRIBUTION_SPECS,
    OPTIONAL_FLOW_SPECS,
    OPTIONAL_SAMPLER_SPECS,
    OptionalBackendSpec,
)


# Combine BlackJAX (always-available) with optional samplers; BlackJAX leads
# the sampler family in the resolution order.
_BLACKJAX_AS_OPTIONAL_SPEC = OptionalBackendSpec(
    name="BlackJAX",
    family="sampler",
    source_package="artifex",
    import_module="artifex.generative_models.core.sampling.blackjax_samplers",
    install_hint="artifex (already installed as Opifex dependency)",
    method_names=BLACKJAX_BACKEND_SPEC.sampler_names,
)


class BackendRouter:
    """Resolve backend specs by family with Artifex-first priority.

    The router holds a per-family ordered tuple of
    :class:`OptionalBackendSpec` entries. ``resolve(family)`` returns the
    first available spec; ``resolve(family, name=...)`` returns the named
    spec regardless of availability (the caller can then call
    ``spec.instantiate()`` which raises ``ImportError`` with the install
    hint when absent).
    """

    _families: dict[str, tuple[OptionalBackendSpec, ...]]

    def __init__(self) -> None:
        self._families = {
            "flow": ARTIFEX_FLOW_SPECS + OPTIONAL_FLOW_SPECS,
            "sampler": (_BLACKJAX_AS_OPTIONAL_SPEC, *OPTIONAL_SAMPLER_SPECS),
            "distribution": DISTRIBUTION_SPECS,
        }

    def available(self, family: str) -> tuple[OptionalBackendSpec, ...]:
        """Return the ordered tuple of specs registered for ``family``.

        Returns ALL specs, not just installed ones; callers filter with
        ``spec.probe()`` if needed.
        """
        self._require_known_family(family)
        return self._families[family]

    def resolve(self, family: str, *, name: str | None = None) -> OptionalBackendSpec:
        """Return the highest-priority spec for ``family``.

        ``name=None`` returns the first installed spec (Artifex-first). When
        no spec in the family is installed, returns the first registered
        spec so the caller can read its install_hint.

        ``name=<...>`` returns the spec with that ``name`` regardless of
        installation status; ``ValueError`` is raised if no spec with that
        name exists in the family.
        """
        self._require_known_family(family)
        specs = self._families[family]
        if name is not None:
            for spec in specs:
                if spec.name == name:
                    return spec
            raise ValueError(
                f"unknown backend {name!r} for family {family!r}. "
                f"Registered: {[s.name for s in specs]!r}."
            )
        for spec in specs:
            if spec.probe():
                return spec
        return specs[0]

    def to_inference_backend_spec(
        self, family: str, *, name: str | None = None
    ) -> InferenceBackendSpec:
        """Convert the resolved spec into a protocol-shaped :class:`InferenceBackendSpec`.

        Bridges the optional-router metadata to the protocol-shaped spec
        consumed by capability declarations.
        """
        spec = self.resolve(family, name=name)
        return InferenceBackendSpec(
            name=spec.name,
            family=spec.family,
            sampler_names=spec.method_names,
            source_package=spec.source_package,
        )

    def _require_known_family(self, family: str) -> None:
        if family not in self._families:
            raise ValueError(
                f"unknown family {family!r}. Registered families: "
                f"{sorted(self._families.keys())!r}."
            )


__all__ = ["BackendRouter"]

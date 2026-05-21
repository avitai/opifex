"""Stein Variational Gradient Descent (SVGD) peer backend.

SVGD evolves a finite set of particles according to a kernelised
Stein-gradient flow that minimises the KL divergence to the target
posterior. It produces deterministic non-IID samples that approximate
the posterior in expectation.

Canonical reference:
* Liu, Q. & Wang, D. 2016 — *Stein Variational Gradient Descent: A
  General Purpose Bayesian Inference Algorithm*, NeurIPS 29.

This slice ships the Pattern-A metadata + protocol surface via
:class:`_DeferredInferenceBackend`; the concrete particle-flow
algorithm lands in a follow-up.
"""

from __future__ import annotations

import dataclasses

from opifex.uncertainty.inference_backends._deferred import _DeferredInferenceBackend


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SVGDBackend(_DeferredInferenceBackend):
    """Stein Variational Gradient Descent backend."""

    name: str = "svgd"
    source_package: str = "opifex"
    method_names: tuple[str, ...] = ("svgd",)
    notes: str = (
        "SVGD (Liu+Wang 2016) — Stein kernelised-gradient particle "
        "flow minimising KL to the target posterior."
    )


__all__ = ["SVGDBackend"]

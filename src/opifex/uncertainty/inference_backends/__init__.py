"""Inference-backend protocols and base result types."""

from __future__ import annotations

from opifex.uncertainty.inference_backends.advi import ADVIBackend
from opifex.uncertainty.inference_backends.base import (
    BackendDiagnostics,
    BackendResult,
    InferenceBackendProtocol,
    InferenceBackendSpec,
    UnsupportedBackendError,
)
from opifex.uncertainty.inference_backends.blackjax import (
    BLACKJAX_BACKEND_SPEC,
    BlackJAXBackend,
)
from opifex.uncertainty.inference_backends.optional import (
    ARTIFEX_FLOW_SPECS,
    DISTRIBUTION_SPECS,
    OPTIONAL_FLOW_SPECS,
    OPTIONAL_SAMPLER_SPECS,
    OptionalBackendSpec,
)
from opifex.uncertainty.inference_backends.pathfinder import PathfinderBackend
from opifex.uncertainty.inference_backends.router import BackendRouter
from opifex.uncertainty.inference_backends.svgd import SVGDBackend


__all__ = [
    "ARTIFEX_FLOW_SPECS",
    "BLACKJAX_BACKEND_SPEC",
    "DISTRIBUTION_SPECS",
    "OPTIONAL_FLOW_SPECS",
    "OPTIONAL_SAMPLER_SPECS",
    "ADVIBackend",
    "BackendDiagnostics",
    "BackendResult",
    "BackendRouter",
    "BlackJAXBackend",
    "InferenceBackendProtocol",
    "InferenceBackendSpec",
    "OptionalBackendSpec",
    "PathfinderBackend",
    "SVGDBackend",
    "UnsupportedBackendError",
]

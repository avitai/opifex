"""Inference-backend protocols and base result types."""

from __future__ import annotations

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


__all__ = [
    "BLACKJAX_BACKEND_SPEC",
    "BackendDiagnostics",
    "BackendResult",
    "BlackJAXBackend",
    "InferenceBackendProtocol",
    "InferenceBackendSpec",
    "UnsupportedBackendError",
]

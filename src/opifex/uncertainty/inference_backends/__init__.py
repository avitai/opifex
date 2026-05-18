"""Inference-backend protocols and base result types."""

from __future__ import annotations

from opifex.uncertainty.inference_backends.base import (
    BackendDiagnostics,
    BackendResult,
    InferenceBackendProtocol,
    InferenceBackendSpec,
    UnsupportedBackendError,
)
from opifex.uncertainty.inference_backends.blackjax import BlackJAXBackend


__all__ = [
    "BackendDiagnostics",
    "BackendResult",
    "BlackJAXBackend",
    "InferenceBackendProtocol",
    "InferenceBackendSpec",
    "UnsupportedBackendError",
]

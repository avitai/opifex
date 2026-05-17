"""Inference-backend protocols and base result types."""

from __future__ import annotations

from opifex.uncertainty.inference_backends.base import (
    BackendDiagnostics,
    BackendResult,
    InferenceBackendProtocol,
    InferenceBackendSpec,
    UnsupportedBackendError,
)


__all__ = [
    "BackendDiagnostics",
    "BackendResult",
    "InferenceBackendProtocol",
    "InferenceBackendSpec",
    "UnsupportedBackendError",
]

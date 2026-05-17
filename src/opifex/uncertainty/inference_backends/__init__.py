"""Inference-backend protocols and base result types (Phase 1 Task 1.5)."""

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

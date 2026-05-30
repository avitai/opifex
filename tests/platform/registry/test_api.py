"""Tests for the registry route-layer exception translation.

The domain service (:mod:`opifex.platform.registry.core`) raises framework
agnostic :class:`~opifex.platform.registry.exceptions.RegistryError`
subclasses. The transport boundary is responsible for mapping those domain
errors onto HTTP status codes. These tests pin that mapping so the layering
contract (domain raises, route translates) stays intact.
"""

import pytest
from fastapi import HTTPException

from opifex.platform.registry.api import to_http_exception
from opifex.platform.registry.exceptions import (
    AccessDenied,
    FunctionalNotFound,
    FunctionalTooLarge,
    RegistryError,
    SerializationError,
    ValidationError,
    VersionNotFound,
)


class TestRegistryErrorToHttp:
    """``to_http_exception`` maps each domain error to its HTTP status."""

    @pytest.mark.parametrize(
        ("error", "expected_status"),
        [
            (ValidationError("bad metadata"), 400),
            (AccessDenied("Access denied"), 403),
            (FunctionalNotFound("Functional not found"), 404),
            (VersionNotFound("Version not found"), 404),
            (FunctionalTooLarge("too large"), 413),
            (SerializationError("Failed to parse"), 500),
        ],
    )
    def test_maps_domain_error_to_status(self, error: RegistryError, expected_status: int) -> None:
        """Each domain error translates to the documented HTTP status code."""
        http_error = to_http_exception(error)

        assert isinstance(http_error, HTTPException)
        assert http_error.status_code == expected_status
        # The domain message is preserved as the HTTP detail.
        assert http_error.detail == str(error)

    def test_unknown_registry_error_falls_back_to_500(self) -> None:
        """An unmapped ``RegistryError`` subclass falls back to HTTP 500."""

        class _NovelRegistryError(RegistryError):
            """A subclass with no explicit status mapping."""

        http_error = to_http_exception(_NovelRegistryError("boom"))

        assert isinstance(http_error, HTTPException)
        assert http_error.status_code == 500

    def test_translation_does_not_leak_into_domain(self) -> None:
        """The translator lives in the route layer, never in the domain core."""
        from pathlib import Path

        import opifex.platform.registry.core as core_module

        source = Path(core_module.__file__).read_text(encoding="utf-8")
        assert "to_http_exception" not in source
        assert "HTTPException" not in source

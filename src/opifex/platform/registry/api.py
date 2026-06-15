"""HTTP route-layer adapter for the Neural Functional Registry.

This module owns the transport boundary for the registry. The domain
service (:mod:`opifex.platform.registry.core`) raises framework-agnostic
:class:`~opifex.platform.registry.exceptions.RegistryError` subclasses;
this layer translates them into :class:`fastapi.HTTPException` responses.

Keeping the mapping here preserves the layering contract (R3): the domain
core never imports ``fastapi``, and HTTP status codes — a transport concern
— live only at the route boundary.

A FastAPI router/endpoints are not yet wired in; until they are, route
handlers should call :func:`to_http_exception` to convert a caught
:class:`RegistryError` into the response the client receives, e.g.::

    try:
        return await service.retrieve_functional(functional_id)
    except RegistryError as exc:
        raise to_http_exception(exc) from exc
"""

from fastapi import HTTPException

from opifex.platform.registry.exceptions import (
    AccessDenied,
    FunctionalNotFound,
    FunctionalTooLarge,
    RegistryError,
    SerializationError,
    ValidationError,
    VersionNotFound,
)


# Domain error -> HTTP status code. Ordered most-specific first is not
# required because the lookup is by exact subclass; the fallback handles any
# RegistryError without an explicit entry.
_STATUS_BY_ERROR: dict[type[RegistryError], int] = {
    ValidationError: 400,
    AccessDenied: 403,
    FunctionalNotFound: 404,
    VersionNotFound: 404,
    FunctionalTooLarge: 413,
    SerializationError: 500,
}

_FALLBACK_STATUS = 500


def http_status_for(error: RegistryError) -> int:
    """Return the HTTP status code for a registry domain error.

    Resolves by walking the error's method resolution order so subclasses of
    a mapped error inherit its status. Unmapped errors fall back to ``500``.

    Args:
        error: The domain error raised by the registry service.

    Returns:
        The HTTP status code that represents ``error`` at the transport
        boundary.
    """
    for error_type in type(error).__mro__:
        if error_type in _STATUS_BY_ERROR:
            return _STATUS_BY_ERROR[error_type]
    return _FALLBACK_STATUS


def to_http_exception(error: RegistryError) -> HTTPException:
    """Translate a registry domain error into a FastAPI ``HTTPException``.

    The domain message is preserved verbatim as the HTTP ``detail`` so the
    client receives the service's diagnostic text.

    Args:
        error: The domain error raised by the registry service.

    Returns:
        An :class:`~fastapi.HTTPException` carrying the mapped status code
        and the original error message.
    """
    return HTTPException(status_code=http_status_for(error), detail=str(error))

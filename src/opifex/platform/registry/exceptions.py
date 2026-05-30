"""Domain exceptions for the Neural Functional Registry.

These exceptions express *domain* failure conditions (validation, access
control, lookup, serialisation) without reference to any transport or web
framework. The :class:`RegistryService` raises them so the domain layer
stays independent of HTTP concerns; the route layer
(:mod:`opifex.platform.registry.api`) translates them into HTTP responses.

Layering rule (R3): domain code never imports infrastructure such as
``fastapi``. Mapping a domain failure onto an HTTP status code is a
transport concern and lives at the route boundary only.
"""


class RegistryError(Exception):
    """Base class for all neural functional registry domain errors.

    Catch this to handle any registry-originated failure generically; catch
    a specific subclass to react to a particular failure mode.
    """


class ValidationError(RegistryError):
    """Raised when submitted functional metadata fails validation.

    Covers missing required fields and unrecognised functional types. The
    message identifies the offending field or constraint.
    """


class FunctionalNotFound(RegistryError):
    """Raised when a requested neural functional does not exist."""


class VersionNotFound(RegistryError):
    """Raised when a requested version of a functional does not exist.

    Also raised when the on-disk artifact for an otherwise known
    functional/version cannot be located.
    """


class AccessDenied(RegistryError):
    """Raised when a user lacks permission for the requested operation.

    Covers both read access to private functionals and mutating operations
    (such as deletion) restricted to the owning author.
    """


class FunctionalTooLarge(RegistryError):
    """Raised when a serialised functional exceeds the configured size limit.

    The message reports the actual size and the configured maximum.
    """


class SerializationError(RegistryError):
    """Raised when stored functional data cannot be parsed back into a dict."""

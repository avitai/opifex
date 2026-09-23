"""UQ capability declarations and registry.

Capability containers are scalar/string/enum-only frozen+slotted dataclasses,
hashable with no array data. Used for Python-level registry lookup; never
traced as pytree data.

``calibrax.core.registry.SingletonRegistry[T]`` is reused as the backing
mechanism. Opifex adds two domain-specific policies on top:

1. Duplicate-registration rejection (CalibraX overwrites silently; rejection
   is required because every capability declaration is canonical).
2. ``require()`` enhances ``get()``'s ``KeyError`` with the sorted list of
   available names so callers get an actionable error message.
"""

from collections.abc import Callable
from typing import TypeVar

from calibrax.core.registry import SingletonRegistry

from opifex.core.capabilities import UQCapability


_C = TypeVar("_C", bound=type)


class UQRegistry(SingletonRegistry["UQCapability"]):
    """Singleton registry mapping capability-name → :class:`UQCapability`.

    Extends CalibraX's :class:`SingletonRegistry` with two Opifex-specific
    policies: duplicate-rejection and an enhanced ``require`` error message.
    """

    def register(self, name: str, item: UQCapability) -> None:
        """Register a capability, rejecting duplicate names with ``ValueError``."""
        if name in self:
            raise ValueError(
                f"Capability {name!r} is already registered. Capability "
                "declarations are canonical — call require()/get() to read, or "
                "use a new name."
            )
        super().register(name, item)

    def require(self, name: str) -> UQCapability:
        """Return the registered capability or raise ``KeyError`` with available names."""
        if name not in self:
            available = sorted(self.list_names())
            raise KeyError(f"Capability {name!r} not registered. Available: {available!r}.")
        return self.get(name)


def register_uq_capability(name: str, capability: UQCapability) -> Callable[[_C], _C]:
    """Class decorator that registers ``capability`` under ``name`` in :class:`UQRegistry`.

    Lets capability declaration tests register a capability through a class
    decorator at class-definition time.
    """

    def decorator(cls: _C) -> _C:
        """Register the decorated class's capability and return it unchanged."""
        registry = UQRegistry()
        registry.register(name, capability)
        return cls

    return decorator


__all__ = [
    "UQRegistry",
    "register_uq_capability",
]

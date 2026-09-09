"""The deprecation notice every one-release wrapper emits.

Functions moved to calibrax or substrax keep their opifex names for one release
as wrappers over the sibling's function; each call emits this warning and the
names are removed in 0.2.3.
"""

from __future__ import annotations

import warnings


REMOVAL_VERSION = "0.2.3"


def warn_deprecated(name: str, home: str) -> None:
    """Warn that ``name`` is a deprecated wrapper over ``home``.

    Args:
        name: The opifex-qualified name being called.
        home: The sibling-qualified function that owns the behaviour.
    """
    warnings.warn(
        f"{name} is deprecated and is removed in opifex {REMOVAL_VERSION}; use {home} instead.",
        DeprecationWarning,
        stacklevel=3,
    )

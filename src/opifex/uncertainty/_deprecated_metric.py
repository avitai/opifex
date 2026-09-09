"""The deprecation notice every metric wrapper emits.

opifex's metric functions were moved to calibrax in 0.1.3 with the same
semantics; the opifex names stay for one release as keyword-argument wrappers
over the calibrax functions and are removed in 0.2.3.
"""

from __future__ import annotations

import warnings


REMOVAL_VERSION = "0.2.3"


def warn_deprecated_metric(name: str, home: str) -> None:
    """Warn that ``name`` is a deprecated wrapper over the calibrax function ``home``.

    Args:
        name: The opifex-qualified name being called.
        home: The calibrax-qualified function that computes the metric.
    """
    warnings.warn(
        f"{name} is deprecated and is removed in opifex {REMOVAL_VERSION}; "
        f"call {home} from calibrax instead.",
        DeprecationWarning,
        stacklevel=3,
    )

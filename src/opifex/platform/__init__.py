"""Opifex Community Platform Package.

This package provides the community platform infrastructure including:
- Neural functional registry for storing and versioning neural functionals

The platform follows a modular monolith architecture with domain boundaries,
providing high-performance APIs for scientific machine learning collaboration.

Note: This module intentionally shadows the standard library 'platform' module.
Since it's namespaced under 'opifex.platform', there's no import conflict.
"""  # noqa: A005

from opifex.platform import registry


__all__ = [
    "registry",
]

__version__ = "1.0.0"

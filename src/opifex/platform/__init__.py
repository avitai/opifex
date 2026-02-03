"""Opifex Community Platform Package.

This package provides the community platform infrastructure including:
- Neural functional registry for storing and versioning neural functionals
- Collaboration tools for research teams and communities
- Dashboard and analytics for platform monitoring and insights
- Benchmarking platform integration for performance analysis

The platform follows a modular monolith architecture with domain boundaries,
providing high-performance APIs for scientific machine learning collaboration.

Note: This module intentionally shadows the standard library 'platform' module.
Since it's namespaced under 'opifex.platform', there's no import conflict.
"""  # noqa: A005

# Import only the registry package which has core implementation
from opifex.platform import registry


# Other packages are reserved for future implementation
# from opifex.platform import collaboration, dashboard, benchmarking

__all__ = [
    "registry",
    # "collaboration",
    # "dashboard",
    # "benchmarking",
]

__version__ = "1.0.0"

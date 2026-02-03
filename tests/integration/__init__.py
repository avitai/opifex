"""Integration testing module for Opifex framework.

This module contains comprehensive integration tests that validate interactions
between different components of the Opifex framework. Tests are organized into
layers:

- Component Integration: Tests between pairs of modules
- Module Integration: Complete module validation
- Workflow Integration: Cross-module workflow testing

All tests follow the layered testing ecosystem architecture with shared
testing infrastructure for consistency and efficiency.

For fixtures, import directly from integration_fixtures:
    from tests.integration.integration_fixtures import integration_framework
"""

from .framework import OpifexTestFramework, PerformanceMonitor, TestDataManager


__all__ = [
    "OpifexTestFramework",
    "PerformanceMonitor",
    "TestDataManager",
]

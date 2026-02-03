"""Neural Functional Registry Package.

Provides neural functional storage, versioning, and discovery capabilities
for the Opifex community platform. Supports registration and retrieval of:
- Learn-to-optimize (L2O) optimizers
- Neural operators (FNO, DeepONet, etc.)
- Physics-informed neural networks (PINNs)
- Neural density functional theory (Neural DFT) functionals

Key Components:
- RegistryService: Core registry operations and storage
- SearchEngine: Neural functional discovery and search
- VersionManager: Git-based versioning with dependency tracking
- ValidationEngine: Automated testing and validation
"""

# Import all available registry components
try:
    from opifex.platform.registry.core import RegistryService

    _has_core = True
except ImportError:
    _has_core = False

try:
    from opifex.platform.registry.search import (
        SearchEngine,
        SearchQuery,
        SearchResult,
        SearchType,
    )

    _has_search = True
except ImportError:
    _has_search = False

try:
    from opifex.platform.registry.version import (
        Branch,
        MergeStrategy,
        Version,
        VersionDiff,
        VersionManager,
        VersionStatus,
    )

    _has_version = True
except ImportError:
    _has_version = False

try:
    from opifex.platform.registry.validation import (
        FunctionalReport,
        TestType,
        ValidationEngine,
        ValidationResult,
        ValidationRule,
        ValidationStatus,
    )

    _has_validation = True
except ImportError:
    _has_validation = False

# Build __all__ list dynamically
__all__ = []

if _has_core:
    __all__ += ["RegistryService"]

if _has_search:
    __all__ += ["SearchEngine", "SearchQuery", "SearchResult", "SearchType"]

if _has_version:
    __all__ += [
        "Branch",
        "MergeStrategy",
        "Version",
        "VersionDiff",
        "VersionManager",
        "VersionStatus",
    ]

if _has_validation:
    __all__ += [
        "FunctionalReport",
        "TestType",
        "ValidationEngine",
        "ValidationResult",
        "ValidationRule",
        "ValidationStatus",
    ]

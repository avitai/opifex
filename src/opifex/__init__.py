"""
Opifex: Unified Scientific Machine Learning Framework

A JAX-native platform for scientific machine learning with probabilistic-first design,
high performance, and production-ready architecture.

Importing ``opifex`` mutates nothing global: no environment variable, no ``jax.config``
value, no cache directory. A process declares the JAX settings it starts with through
``substrax.runtime`` (``JaxRuntime``, ``apply_runtime``, ``runtime_environment``).
"""

from email.utils import parseaddr
from importlib.metadata import metadata, PackageNotFoundError


try:
    # Single source of truth: project metadata declared in pyproject.toml, read
    # from the installed package rather than duplicated here. ``Author-email`` is
    # the PEP 621 combined ``"Name <email>"`` form, split via ``parseaddr``.
    _metadata = metadata("opifex")
    __version__ = _metadata["Version"]
    __author__, __email__ = parseaddr(_metadata["Author-email"] or "")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
    __author__ = ""
    __email__ = ""

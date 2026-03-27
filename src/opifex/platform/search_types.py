"""Shared search types for neural functional discovery.

Provides the common SearchType enum used by both the platform registry
search engine and the scalability search engine.
"""

from enum import Enum


class SearchType(Enum):
    """Types of search operations supported."""

    TEXT = "text"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    FILTER = "filter"


__all__ = ["SearchType"]

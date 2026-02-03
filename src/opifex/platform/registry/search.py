"""Neural Functional Search Engine.

Provides comprehensive search capabilities for neural functionals including
text search, semantic search with neural embeddings, advanced filtering,
and recommendation systems for the Opifex community platform.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp


class SearchType(Enum):
    """Types of search operations."""

    TEXT = "text"
    SEMANTIC = "semantic"
    FILTER = "filter"
    HYBRID = "hybrid"


@dataclass
class SearchQuery:
    """Search query for neural functionals."""

    query_text: str = ""
    functional_type: str | None = None
    domain: str | None = None
    tags: list[str] | None = None
    author_id: str | None = None
    min_rating: float | None = None
    min_accuracy: float | None = None
    max_memory_gb: float | None = None
    gpu_required: bool | None = None
    limit: int = 50
    offset: int = 0
    search_type: SearchType = SearchType.HYBRID


@dataclass
class SearchResult:
    """Search result for a neural functional."""

    functional_id: str
    name: str
    description: str
    functional_type: str
    author_id: str
    tags: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchEngine:
    """Neural functional search engine.

    Provides comprehensive search capabilities including text-based search,
    semantic search with neural embeddings, advanced filtering, and
    recommendation systems.
    """

    def __init__(
        self,
        registry_service,
        enable_semantic_search: bool = True,
        similarity_threshold: float = 0.7,
    ):
        """Initialize search engine.

        Args:
            registry_service: Registry service for functional access
            enable_semantic_search: Whether to enable neural embeddings
            similarity_threshold: Minimum similarity for semantic matches
        """
        self.registry = registry_service
        self.enable_semantic = enable_semantic_search
        self.similarity_threshold = similarity_threshold

        # Search indices and caches
        self._text_index: dict[str, Any] = {}
        self._semantic_embeddings: dict[str, jnp.ndarray] = {}

        # Stop words for text processing
        self._stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "this",
            "but",
            "they",
            "have",
            "had",
            "what",
            "said",
            "each",
            "which",
            "do",
            "how",
            "their",
            "if",
        }

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute search query for neural functionals.

        Args:
            query: Search query with criteria and options

        Returns:
            List of matching functionals ranked by relevance
        """
        if query.search_type == SearchType.TEXT:
            results = await self._text_search(query)
        elif query.search_type == SearchType.SEMANTIC:
            results = await self._semantic_search(query)
        elif query.search_type == SearchType.FILTER:
            results = await self._filter_search(query)
        else:  # HYBRID
            results = await self._hybrid_search(query)

        # Apply pagination
        start = query.offset
        end = query.offset + query.limit
        return results[start:end]

    async def suggest_functionals(
        self, functional_id: str, limit: int = 5
    ) -> list[SearchResult]:
        """Suggest similar functionals based on a reference functional.

        Args:
            functional_id: ID of reference functional
            limit: Maximum number of suggestions

        Returns:
            List of similar functionals
        """
        # Get reference functional
        reference = await self.registry.retrieve_functional(functional_id)
        if not reference:
            return []

        # Get all functionals for comparison
        all_functionals = await self._get_all_functionals()

        if not self.enable_semantic:
            # Simple tag-based similarity
            suggestions = []
            ref_tags = set(reference.get("tags", []))

            for functional in all_functionals:
                if functional["id"] == functional_id:
                    continue

                # Calculate tag overlap
                func_tags = set(functional.get("tags", []))
                overlap = len(ref_tags & func_tags)
                if overlap > 0:
                    suggestions.append(
                        SearchResult(
                            functional_id=functional["id"],
                            name=functional["name"],
                            description=functional["description"],
                            functional_type=functional["type"],
                            author_id=functional["author_id"],
                            tags=functional.get("tags", []),
                            relevance_score=overlap
                            / max(len(ref_tags), len(func_tags)),
                            metadata=functional.get("metadata", {}),
                        )
                    )

            # Sort by relevance and return top results
            suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
            return suggestions[:limit]

        # Semantic similarity-based suggestions
        ref_embedding = await self._get_functional_embedding(reference)
        suggestions = []

        for functional in all_functionals:
            if functional["id"] == functional_id:
                continue

            func_embedding = await self._get_functional_embedding(functional)
            similarity = self._cosine_similarity(ref_embedding, func_embedding)

            if similarity >= self.similarity_threshold:
                suggestions.append(
                    SearchResult(
                        functional_id=functional["id"],
                        name=functional["name"],
                        description=functional["description"],
                        functional_type=functional["type"],
                        author_id=functional["author_id"],
                        tags=functional.get("tags", []),
                        relevance_score=float(similarity),
                        metadata=functional.get("metadata", {}),
                    )
                )

        # Sort by similarity and return top results
        suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
        return suggestions[:limit]

    async def search_by_problem(
        self, problem_description: str, domain: str | None = None, limit: int = 10
    ) -> list[SearchResult]:
        """Search for functionals suitable for a specific problem.

        Args:
            problem_description: Description of the problem to solve
            domain: Optional domain filter
            limit: Maximum number of results

        Returns:
            List of suitable functionals
        """
        # Create a search query based on problem description
        query = SearchQuery(
            query_text=problem_description,
            domain=domain,
            search_type=SearchType.HYBRID,
            limit=limit,
        )

        return await self.search(query)

    async def _text_search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute text-based search using keyword matching.

        Args:
            query: Search query with text

        Returns:
            List of functionals matching text query
        """
        if not query.query_text.strip():
            return []

        keywords = self._extract_keywords(query.query_text)
        if not keywords:
            return []

        # Get all functionals
        all_functionals = await self._get_all_functionals()
        results = []

        for functional in all_functionals:
            score = self._calculate_text_score(functional, keywords)
            if score > 0:
                results.append(
                    SearchResult(
                        functional_id=functional["id"],
                        name=functional["name"],
                        description=functional["description"],
                        functional_type=functional["type"],
                        author_id=functional["author_id"],
                        tags=functional.get("tags", []),
                        relevance_score=score,
                        metadata=functional.get("metadata", {}),
                    )
                )

        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Apply additional filters
        return self._apply_filters(results, query)

    async def _semantic_search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute semantic search using neural embeddings.

        Args:
            query: Search query with text

        Returns:
            List of functionals with semantic similarity
        """
        if not self.enable_semantic or not query.query_text.strip():
            return []

        # Generate query embedding
        query_embedding = self._generate_embedding(query.query_text)

        # Get all functionals
        all_functionals = await self._get_all_functionals()
        results = []

        for functional in all_functionals:
            func_embedding = await self._get_functional_embedding(functional)
            similarity = self._cosine_similarity(query_embedding, func_embedding)

            if similarity >= self.similarity_threshold:
                results.append(
                    SearchResult(
                        functional_id=functional["id"],
                        name=functional["name"],
                        description=functional["description"],
                        functional_type=functional["type"],
                        author_id=functional["author_id"],
                        tags=functional.get("tags", []),
                        relevance_score=float(similarity),
                        metadata=functional.get("metadata", {}),
                    )
                )

        # Sort by similarity
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Apply additional filters
        return self._apply_filters(results, query)

    async def _filter_search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute filter-based search.

        Args:
            query: Search query with filter criteria

        Returns:
            List of functionals matching filters
        """
        # Get all functionals
        all_functionals = await self._get_all_functionals()

        # Convert to search results
        results = [
            SearchResult(
                functional_id=functional["id"],
                name=functional["name"],
                description=functional["description"],
                functional_type=functional["type"],
                author_id=functional["author_id"],
                tags=functional.get("tags", []),
                relevance_score=1.0,  # All matches are equally relevant
                metadata=functional.get("metadata", {}),
            )
            for functional in all_functionals
        ]

        # Apply filters
        return self._apply_filters(results, query)

    async def _hybrid_search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute hybrid search combining text/semantic and filters.

        Args:
            query: Search query with text and filter criteria

        Returns:
            List of functionals from hybrid search
        """
        if query.query_text.strip():
            # Use semantic search if available, otherwise text search
            if self.enable_semantic:
                results = await self._semantic_search(query)
            else:
                results = await self._text_search(query)
        else:
            # No text query, fall back to filter search
            results = await self._filter_search(query)

        return results

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text for search indexing.

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        # Convert to lowercase and split by non-alphanumeric characters
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out stop words and short words
        keywords = [
            word for word in words if len(word) > 2 and word not in self._stop_words
        ]

        # Remove duplicates while preserving order
        return list(dict.fromkeys(keywords))

    def _calculate_text_score(
        self, functional: dict[str, Any], keywords: list[str]
    ) -> float:
        """Calculate text relevance score for a functional.

        Args:
            functional: Functional metadata
            keywords: Search keywords

        Returns:
            Relevance score between 0 and 1
        """
        if not keywords:
            return 0.0

        # Extract searchable text
        searchable_text = " ".join(
            [
                functional.get("name", ""),
                functional.get("description", ""),
                " ".join(functional.get("tags", [])),
            ]
        ).lower()

        if not searchable_text:
            return 0.0

        # Calculate keyword matches
        matches = 0
        total_weight = 0

        for keyword in keywords:
            weight = 1.0

            # Higher weight for name matches
            if keyword in functional.get("name", "").lower():
                weight = 3.0
                matches += weight
            # Medium weight for tag matches
            elif keyword in " ".join(functional.get("tags", [])).lower():
                weight = 2.0
                matches += weight
            # Lower weight for description matches
            elif keyword in functional.get("description", "").lower():
                weight = 1.0
                matches += weight

            total_weight += weight

        # Normalize score
        return matches / total_weight if total_weight > 0 else 0.0

    def _apply_filters(
        self, results: list[SearchResult], query: SearchQuery
    ) -> list[SearchResult]:
        """Apply filter criteria to search results.

        Args:
            results: Initial search results
            query: Search query with filter criteria

        Returns:
            Filtered search results
        """
        filtered = results

        # Filter by functional type
        if query.functional_type:
            filtered = [
                r for r in filtered if r.functional_type == query.functional_type
            ]

        # Filter by author
        if query.author_id:
            filtered = [r for r in filtered if r.author_id == query.author_id]

        # Filter by tags
        if query.tags:
            filtered = [r for r in filtered if any(tag in r.tags for tag in query.tags)]

        # Filter by domain
        if query.domain:
            filtered = [
                r
                for r in filtered
                if query.domain.lower() in r.metadata.get("domain", "").lower()
            ]

        # Filter by minimum rating
        if query.min_rating is not None:
            filtered = [
                r
                for r in filtered
                if r.metadata.get("average_rating", 0) >= query.min_rating
            ]

        # Filter by minimum accuracy
        if query.min_accuracy is not None:
            filtered = [
                r
                for r in filtered
                if r.metadata.get("accuracy", 0) >= query.min_accuracy
            ]

        # Filter by maximum memory usage
        if query.max_memory_gb is not None:
            filtered = [
                r
                for r in filtered
                if r.metadata.get("memory_gb", 0) <= query.max_memory_gb
            ]

        # Filter by GPU requirement
        if query.gpu_required is not None:
            if query.gpu_required:
                filtered = [
                    r for r in filtered if r.metadata.get("gpu_required", False)
                ]
            else:
                filtered = [
                    r for r in filtered if not r.metadata.get("gpu_required", False)
                ]

        return filtered

    def _generate_embedding(self, text: str) -> jnp.ndarray:
        """Generate neural embedding for text.

        Args:
            text: Input text

        Returns:
            Neural embedding vector
        """
        # Simple hash-based embedding for demonstration
        # In production, this would use a neural model like BERT

        if not text.strip():
            return jnp.zeros(256)

        # Create a simple hash-based embedding
        hash_value = hash(text.lower())

        # Generate deterministic random embedding from hash
        key = jax.random.PRNGKey(abs(hash_value) % (2**31))
        embedding = jax.random.normal(key, (256,))

        # Normalize to unit vector
        norm = jnp.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    async def _get_functional_embedding(
        self, functional: dict[str, Any]
    ) -> jnp.ndarray:
        """Get or generate embedding for a functional.

        Args:
            functional: Functional metadata

        Returns:
            Embedding vector for the functional
        """
        functional_id = functional["id"]

        # Check cache
        if functional_id in self._semantic_embeddings:
            return self._semantic_embeddings[functional_id]

        # Generate embedding from functional text
        text = " ".join(
            [
                functional.get("name", ""),
                functional.get("description", ""),
                " ".join(functional.get("tags", [])),
            ]
        )

        embedding = self._generate_embedding(text)

        # Cache embedding
        self._semantic_embeddings[functional_id] = embedding

        return embedding

    def _cosine_similarity(self, a: jnp.ndarray, b: jnp.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (-1 to 1)
        """
        # Handle zero vectors
        norm_a = jnp.linalg.norm(a)
        norm_b = jnp.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Calculate cosine similarity
        return float(jnp.dot(a, b) / (norm_a * norm_b))

    async def _get_all_functionals(self) -> list[dict[str, Any]]:
        """Get all functionals from registry.

        Returns:
            List of all functional metadata
        """
        # This would interface with the actual registry service
        # For now, return results from search_functionals
        return await self.registry.search_functionals()

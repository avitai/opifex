"""Search Engine for Neural Functional Discovery.

Provides comprehensive search capabilities for the neural functional registry
including text search, semantic search, filtering, and recommendation systems.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import jax.numpy as jnp


class SearchType(Enum):
    """Types of search operations supported."""

    TEXT = "text"
    SEMANTIC = "semantic"
    FILTER = "filter"
    HYBRID = "hybrid"


@dataclass
class SearchQuery:
    """Structured search query for neural functionals."""

    # Text search
    query_text: str = ""

    # Filters
    functional_type: str | None = None
    domain: str | None = None
    tags: list[str] | None = None
    author_id: str | None = None
    min_rating: float | None = None

    # Performance criteria
    min_accuracy: float | None = None
    max_memory_gb: int | None = None
    gpu_required: bool | None = None

    # Pagination
    limit: int = 50
    offset: int = 0

    # Search type
    search_type: SearchType = SearchType.HYBRID


@dataclass
class SearchResult:
    """Search result with relevance scoring."""

    functional_id: str
    name: str
    description: str
    functional_type: str
    author_id: str
    tags: list[str]
    relevance_score: float
    metadata: dict[str, Any]


class SearchEngine:
    """Neural functional search engine with semantic capabilities.

    Provides text search, semantic search, filtering, and recommendation
    systems for discovering neural functionals in the registry.
    """

    def __init__(
        self,
        registry_service,
        enable_semantic_search: bool = True,
        similarity_threshold: float = 0.7,
    ):
        """Initialize search engine.

        Args:
            registry_service: Registry service for data access
            enable_semantic_search: Whether to enable semantic search
            similarity_threshold: Minimum similarity for semantic matches
        """
        self.registry = registry_service
        self.enable_semantic = enable_semantic_search
        self.similarity_threshold = similarity_threshold

        # Search indices (would be backed by database/elasticsearch in production)
        self._text_index: dict[str, set[str]] = {}
        self._tag_index: dict[str, set[str]] = {}
        self._type_index: dict[str, set[str]] = {}
        self._semantic_embeddings: dict[str, jnp.ndarray] = {}

        # Stop words for text processing
        self._stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
        }

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute search query and return ranked results.

        Args:
            query: Structured search query

        Returns:
            List of search results sorted by relevance
        """
        if query.search_type == SearchType.TEXT:
            return await self._text_search(query)
        if query.search_type == SearchType.SEMANTIC:
            return await self._semantic_search(query)
        if query.search_type == SearchType.FILTER:
            return await self._filter_search(query)
        # HYBRID
        return await self._hybrid_search(query)

    async def suggest_functionals(
        self, functional_id: str, limit: int = 10
    ) -> list[SearchResult]:
        """Suggest similar functionals based on a given functional.

        Args:
            functional_id: ID of reference functional
            limit: Maximum suggestions to return

        Returns:
            List of similar functionals
        """
        # Get reference functional
        reference = await self.registry.retrieve_functional(functional_id)
        if not reference:
            return []

        # Create query based on reference functional
        query = SearchQuery(
            functional_type=reference["type"],
            tags=reference["tags"][:3],  # Top 3 tags
            limit=limit + 1,  # +1 to exclude reference itself
        )

        # Search for similar functionals
        results = await self.search(query)

        # Filter out the reference functional
        suggestions = [r for r in results if r.functional_id != functional_id]

        return suggestions[:limit]

    async def search_by_problem(
        self, problem_description: str, domain: str | None = None, limit: int = 20
    ) -> list[SearchResult]:
        """Search functionals suitable for a specific problem.

        Args:
            problem_description: Natural language problem description
            domain: Scientific domain (optional)
            limit: Maximum results to return

        Returns:
            List of functionals suitable for the problem
        """
        # Extract keywords from problem description
        keywords = self._extract_keywords(problem_description)

        # Build query
        query = SearchQuery(
            query_text=" ".join(keywords),
            domain=domain,
            search_type=SearchType.HYBRID,
            limit=limit,
        )

        return await self.search(query)

    # Private search methods

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
                        tags=functional["tags"],
                        relevance_score=score,
                        metadata=functional["metadata"],
                    )
                )

        # Apply filters
        results = self._apply_filters(results, query)

        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results[query.offset : query.offset + query.limit]

    async def _semantic_search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute semantic search using neural embeddings.

        Args:
            query: Search query with text for semantic matching

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
            # Get functional embedding
            functional_embedding = await self._get_functional_embedding(functional)

            # Calculate semantic similarity
            similarity = self._cosine_similarity(query_embedding, functional_embedding)

            if similarity >= self.similarity_threshold:
                results.append(
                    SearchResult(
                        functional_id=functional["id"],
                        name=functional["name"],
                        description=functional["description"],
                        functional_type=functional["type"],
                        author_id=functional["author_id"],
                        tags=functional["tags"],
                        relevance_score=similarity,
                        metadata=functional["metadata"],
                    )
                )

        # Apply filters
        results = self._apply_filters(results, query)

        # Sort by semantic similarity
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results[query.offset : query.offset + query.limit]

    async def _filter_search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute filter-based search using metadata criteria.

        Args:
            query: Search query with filter criteria

        Returns:
            List of functionals matching filter criteria
        """
        # Get all functionals
        all_functionals = await self._get_all_functionals()
        results = []

        for functional in all_functionals:
            # Default score for filter-only search
            results.append(
                SearchResult(
                    functional_id=functional["id"],
                    name=functional["name"],
                    description=functional["description"],
                    functional_type=functional["type"],
                    author_id=functional["author_id"],
                    tags=functional["tags"],
                    relevance_score=1.0,  # Equal relevance for all matches
                    metadata=functional["metadata"],
                )
            )

        # Apply filters
        results = self._apply_filters(results, query)

        # Sort by name for consistent ordering
        results.sort(key=lambda x: x.name)

        return results[query.offset : query.offset + query.limit]

    async def _hybrid_search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute hybrid search combining text and semantic search.

        Args:
            query: Search query for hybrid search

        Returns:
            List of functionals with combined relevance scoring
        """
        text_results = []
        semantic_results = []

        # Execute text search if query text provided
        if query.query_text.strip():
            text_results = await self._text_search(query)

            # Execute semantic search if enabled
            if self.enable_semantic:
                semantic_results = await self._semantic_search(query)

        # If no text query, fall back to filter search
        if not query.query_text.strip():
            return await self._filter_search(query)

        # Combine results with weighted scoring
        combined_results = {}
        text_weight = 0.6
        semantic_weight = 0.4

        # Add text search results
        for result in text_results:
            combined_results[result.functional_id] = SearchResult(
                functional_id=result.functional_id,
                name=result.name,
                description=result.description,
                functional_type=result.functional_type,
                author_id=result.author_id,
                tags=result.tags,
                relevance_score=result.relevance_score * text_weight,
                metadata=result.metadata,
            )

        # Add semantic search results
        for result in semantic_results:
            if result.functional_id in combined_results:
                # Combine scores
                combined_results[result.functional_id].relevance_score += (
                    result.relevance_score * semantic_weight
                )
            else:
                # Add new result
                combined_results[result.functional_id] = SearchResult(
                    functional_id=result.functional_id,
                    name=result.name,
                    description=result.description,
                    functional_type=result.functional_type,
                    author_id=result.author_id,
                    tags=result.tags,
                    relevance_score=result.relevance_score * semantic_weight,
                    metadata=result.metadata,
                )

        # Convert to list and sort by combined score
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return final_results[query.offset : query.offset + query.limit]

    # Helper methods

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text.

        Args:
            text: Input text to process

        Returns:
            List of extracted keywords
        """
        # Clean and normalize text
        text = text.lower().strip()

        # Remove special characters and split into words
        words = re.findall(r"\b[a-zA-Z]+\b", text)

        # Filter out stop words and short words
        keywords = [
            word for word in words if word not in self._stop_words and len(word) > 2
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        return unique_keywords

    def _calculate_text_score(
        self, functional: dict[str, Any], keywords: list[str]
    ) -> float:
        """Calculate text relevance score for a functional.

        Args:
            functional: Functional metadata
            keywords: Search keywords

        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not keywords:
            return 0.0

        # Text fields to search
        searchable_text = " ".join(
            [
                functional.get("name", ""),
                functional.get("description", ""),
                " ".join(functional.get("tags", [])),
                functional.get("type", ""),
            ]
        ).lower()

        # Count keyword matches
        matches = 0
        total_keywords = len(keywords)

        for keyword in keywords:
            if keyword in searchable_text:
                matches += 1

        # Calculate base score
        base_score = matches / total_keywords if total_keywords > 0 else 0.0

        # Boost score for exact name matches
        name_boost = 0.0
        if any(keyword in functional.get("name", "").lower() for keyword in keywords):
            name_boost = 0.2

        # Boost score for type matches
        type_boost = 0.0
        if any(keyword in functional.get("type", "").lower() for keyword in keywords):
            type_boost = 0.1

        # Final score (capped at 1.0)
        return min(1.0, base_score + name_boost + type_boost)

    def _apply_filters(
        self, results: list[SearchResult], query: SearchQuery
    ) -> list[SearchResult]:
        """Apply filter criteria to search results.

        Args:
            results: List of search results
            query: Search query with filter criteria

        Returns:
            Filtered search results
        """
        filtered_results = []

        for result in results:
            if self._passes_basic_filters(
                result, query
            ) and self._passes_performance_filters(result, query):
                filtered_results.append(result)

        return filtered_results

    def _passes_basic_filters(self, result: SearchResult, query: SearchQuery) -> bool:
        """Check if result passes basic filters."""
        # Check functional type filter
        if query.functional_type and result.functional_type != query.functional_type:
            return False

        # Check author filter
        if query.author_id and result.author_id != query.author_id:
            return False

        # Check tags filter
        if query.tags and not any(tag in result.tags for tag in query.tags):
            return False

        # Check domain filter (in metadata)
        if query.domain:
            metadata_domain = result.metadata.get("domain", "")
            if query.domain.lower() not in metadata_domain.lower():
                return False

        # Check rating filter
        if query.min_rating is not None:
            rating = result.metadata.get("average_rating", 0.0)
            if rating < query.min_rating:
                return False

        return True

    def _passes_performance_filters(
        self, result: SearchResult, query: SearchQuery
    ) -> bool:
        """Check if result passes performance filters."""
        # Check performance criteria
        if query.min_accuracy is not None:
            accuracy = result.metadata.get("accuracy", 0.0)
            if accuracy < query.min_accuracy:
                return False

        if query.max_memory_gb is not None:
            memory_gb = result.metadata.get("memory_gb", float("inf"))
            if memory_gb > query.max_memory_gb:
                return False

        if query.gpu_required is not None:
            gpu_required = result.metadata.get("gpu_required", False)
            if query.gpu_required and not gpu_required:
                return False

        return True

    def _generate_embedding(self, text: str) -> jnp.ndarray:
        """Generate neural embedding for text.

        Args:
            text: Input text

        Returns:
            Text embedding vector
        """
        # Simple bag-of-words embedding for this implementation
        # In production, would use transformer models like BERT/SciBERT

        # Extract keywords
        keywords = self._extract_keywords(text)

        # Create a simple vocabulary-based embedding
        # This is a simplified implementation - production would use pre-trained models
        embedding_dim = 256

        # Initialize embedding vector
        embedding = jnp.zeros(embedding_dim)

        # Hash-based embedding (simplified)
        for keyword in keywords:
            # Simple hash to embedding index
            hash_val = hash(keyword) % embedding_dim
            embedding = embedding.at[hash_val].add(1.0)

        # Normalize embedding
        norm = jnp.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    async def _get_functional_embedding(
        self, functional: dict[str, Any]
    ) -> jnp.ndarray:
        """Get or generate embedding for a functional.

        Args:
            functional: Functional metadata

        Returns:
            Functional embedding vector
        """
        functional_id = functional["id"]

        # Check cache
        if functional_id in self._semantic_embeddings:
            return self._semantic_embeddings[functional_id]

        # Generate embedding from functional text
        functional_text = " ".join(
            [
                functional.get("name", ""),
                functional.get("description", ""),
                " ".join(functional.get("tags", [])),
                functional.get("type", ""),
            ]
        )

        embedding = self._generate_embedding(functional_text)

        # Cache embedding
        self._semantic_embeddings[functional_id] = embedding

        return embedding

    def _cosine_similarity(self, a: jnp.ndarray, b: jnp.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (-1.0 to 1.0)
        """
        # Compute dot product
        dot_product = jnp.dot(a, b)

        # Compute norms
        norm_a = jnp.linalg.norm(a)
        norm_b = jnp.linalg.norm(b)

        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = dot_product / (norm_a * norm_b)

        return float(similarity)

    async def _get_all_functionals(self) -> list[dict[str, Any]]:
        """Retrieve all functionals from registry.

        Returns:
            List of all functional metadata
        """
        # In production, this would implement pagination and caching
        # For now, we'll call the registry search with empty query
        try:
            # Use registry's search method to get all functionals
            return await self.registry.search_functionals(
                query="",  # Empty query to get all
                limit=10000,  # Large limit for development
            )
        except Exception:
            # Fallback to empty list if registry is not available
            return []

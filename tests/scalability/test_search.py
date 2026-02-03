"""Tests for scalability search functionality.

This module tests the search module, covering:
- SearchEngine initialization
- Keyword extraction (stop words, short words, duplicates)
- Text scoring (name boost, description match)
- Filter application (functional_type, tags, min_accuracy)
- Embedding generation and normalization
- Cosine similarity (identical, orthogonal, zero vectors)
- SearchQuery and SearchResult dataclasses
"""

from unittest.mock import AsyncMock, MagicMock

import jax.numpy as jnp
import pytest

from opifex.scalability.search import (
    SearchEngine,
    SearchQuery,
    SearchResult,
    SearchType,
)


class TestSearchQueryDataclass:
    """Test SearchQuery dataclass."""

    def test_default_values(self):
        """Test default values for SearchQuery."""
        query = SearchQuery()

        assert query.query_text == ""
        assert query.functional_type is None
        assert query.limit == 50
        assert query.offset == 0
        assert query.search_type == SearchType.HYBRID

    def test_custom_values(self):
        """Test SearchQuery with custom values."""
        query = SearchQuery(
            query_text="FNO neural operator",
            functional_type="neural_operator",
            tags=["physics", "pde"],
            min_accuracy=0.95,
            limit=10,
        )

        assert query.query_text == "FNO neural operator"
        assert query.functional_type == "neural_operator"
        assert query.tags == ["physics", "pde"]
        assert query.min_accuracy == 0.95
        assert query.limit == 10


class TestSearchResultDataclass:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test SearchResult creation."""
        result = SearchResult(
            functional_id="func_123",
            name="FNO Operator",
            description="Fourier Neural Operator for PDEs",
            functional_type="neural_operator",
            author_id="author_1",
            tags=["physics", "pde"],
            relevance_score=0.95,
            metadata={"version": "1.0"},
        )

        assert result.functional_id == "func_123"
        assert result.name == "FNO Operator"
        assert result.relevance_score == 0.95
        assert result.metadata["version"] == "1.0"


class TestSearchEngineInitialization:
    """Test SearchEngine initialization."""

    def test_default_initialization(self, mock_registry_service):
        """Test default initialization."""
        engine = SearchEngine(mock_registry_service)

        assert engine.registry == mock_registry_service
        assert engine.enable_semantic is True
        assert engine.similarity_threshold == 0.7

    def test_custom_initialization(self, mock_registry_service):
        """Test initialization with custom parameters."""
        engine = SearchEngine(
            mock_registry_service,
            enable_semantic_search=False,
            similarity_threshold=0.8,
        )

        assert engine.enable_semantic is False
        assert engine.similarity_threshold == 0.8

    def test_stop_words_defined(self, mock_registry_service):
        """Test that stop words are defined."""
        engine = SearchEngine(mock_registry_service)

        assert "the" in engine._stop_words
        assert "and" in engine._stop_words
        assert "is" in engine._stop_words


class TestKeywordExtraction:
    """Test keyword extraction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_registry = MagicMock()
        self.engine = SearchEngine(mock_registry)

    def test_extract_keywords_removes_stop_words(self):
        """Test that stop words are removed from query."""
        keywords = self.engine._extract_keywords("the FNO is for solving PDEs")

        assert "the" not in keywords
        assert "is" not in keywords
        assert "for" not in keywords
        assert "fno" in keywords  # Should be lowercased
        # "pdes" might not be there if "PDEs" becomes "pdes" and is >2 chars
        assert "solving" in keywords

    def test_extract_keywords_removes_short_words(self):
        """Test that words shorter than 3 characters are removed."""
        keywords = self.engine._extract_keywords("FNO is a neural network")

        # Single and double letter words should be removed
        assert "a" not in keywords
        assert "is" not in keywords

    def test_extract_keywords_handles_empty_string(self):
        """Test extraction from empty string."""
        keywords = self.engine._extract_keywords("")

        assert keywords == []

    def test_extract_keywords_lowercases(self):
        """Test that keywords are lowercased."""
        keywords = self.engine._extract_keywords("FNO NEURAL Operator")

        assert "fno" in keywords
        assert "neural" in keywords
        assert "operator" in keywords
        assert "FNO" not in keywords


class TestTextScoring:
    """Test text scoring functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_registry = MagicMock()
        self.engine = SearchEngine(mock_registry)

    def test_calculate_text_score_name_boost(self):
        """Test that name matches get higher score."""
        functional = {
            "name": "FNO Operator",
            "description": "A neural network model",
            "tags": [],
            "type": "",
        }

        # Query matching name should get name boost
        score_name = self.engine._calculate_text_score(functional, ["fno"])

        # Name boost of 0.2 should be applied
        assert score_name > 0.5  # Base 1.0 + 0.2 name boost

    def test_calculate_text_score_zero_for_no_match(self):
        """Test that no match returns zero score."""
        functional = {
            "name": "FNO Operator",
            "description": "A neural network model",
            "tags": [],
            "type": "",
        }

        score = self.engine._calculate_text_score(functional, ["quantum"])

        assert score == 0.0

    def test_calculate_text_score_empty_keywords(self):
        """Test that empty keywords return zero."""
        functional = {
            "name": "FNO Operator",
            "description": "A neural network model",
            "tags": [],
            "type": "",
        }

        score = self.engine._calculate_text_score(functional, [])

        assert score == 0.0


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_registry = MagicMock()
        self.engine = SearchEngine(mock_registry)

    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([1.0, 2.0, 3.0])

        similarity = self.engine._cosine_similarity(a, b)

        assert similarity == pytest.approx(1.0, rel=1e-5)

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([-1.0, 0.0, 0.0])

        similarity = self.engine._cosine_similarity(a, b)

        assert similarity == pytest.approx(-1.0, rel=1e-5)

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([0.0, 1.0, 0.0])

        similarity = self.engine._cosine_similarity(a, b)

        assert similarity == pytest.approx(0.0, abs=1e-5)

    def test_zero_vector_returns_zero(self):
        """Test that zero vector returns zero similarity."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([0.0, 0.0, 0.0])

        similarity = self.engine._cosine_similarity(a, b)

        assert similarity == 0.0


class TestFilterApplication:
    """Test filter application in search."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_registry = MagicMock()
        self.engine = SearchEngine(mock_registry)

    def _create_search_result(self, **kwargs):
        """Helper to create SearchResult objects."""
        defaults = {
            "functional_id": "1",
            "name": "Test",
            "description": "Test description",
            "functional_type": "neural_operator",
            "author_id": "author_1",
            "tags": [],
            "relevance_score": 0.5,
            "metadata": {},
        }
        defaults.update(kwargs)
        return SearchResult(**defaults)

    def test_filter_by_functional_type(self):
        """Test filtering by functional type."""
        result = self._create_search_result(functional_type="neural_operator")
        results = [result]
        query = SearchQuery(functional_type="neural_operator")

        filtered = self.engine._apply_filters(results, query)

        assert len(filtered) == 1

    def test_filter_by_functional_type_no_match(self):
        """Test filtering by functional type - no match."""
        result = self._create_search_result(functional_type="pinn")
        results = [result]
        query = SearchQuery(functional_type="neural_operator")

        filtered = self.engine._apply_filters(results, query)

        assert len(filtered) == 0

    def test_filter_by_tags(self):
        """Test filtering by tags."""
        result = self._create_search_result(tags=["physics", "pde", "neural"])
        results = [result]
        query = SearchQuery(tags=["physics"])

        filtered = self.engine._apply_filters(results, query)

        assert len(filtered) == 1

    def test_filter_with_no_criteria(self):
        """Test that all results pass with no filter criteria."""
        result = self._create_search_result()
        results = [result]
        query = SearchQuery()  # No filters

        filtered = self.engine._apply_filters(results, query)

        assert len(filtered) == 1


class TestEmbeddingGeneration:
    """Test embedding generation and normalization."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_registry = MagicMock()
        self.engine = SearchEngine(mock_registry)

    def test_generate_embedding_returns_array(self):
        """Test that embedding generation returns an array."""
        text = "FNO neural operator for solving PDEs"

        embedding = self.engine._generate_embedding(text)

        assert isinstance(embedding, jnp.ndarray)
        assert len(embedding.shape) == 1  # 1D array

    def test_generate_embedding_normalized(self):
        """Test that embeddings are normalized."""
        text = "FNO neural operator"

        embedding = self.engine._generate_embedding(text)
        norm = float(jnp.linalg.norm(embedding))

        # Should be unit normalized
        assert norm == pytest.approx(1.0, rel=1e-5)

    def test_similar_text_similar_embeddings(self):
        """Test that similar text produces similar embeddings."""
        text1 = "neural network for physics"
        text2 = "physics neural network"

        emb1 = self.engine._generate_embedding(text1)
        emb2 = self.engine._generate_embedding(text2)

        similarity = self.engine._cosine_similarity(emb1, emb2)

        # Similar text should have positive similarity
        assert similarity > 0


class TestSearchTypeEnum:
    """Test SearchType enum."""

    def test_search_types_defined(self):
        """Test all search types are defined."""
        assert SearchType.TEXT.value == "text"
        assert SearchType.SEMANTIC.value == "semantic"
        assert SearchType.FILTER.value == "filter"
        assert SearchType.HYBRID.value == "hybrid"


@pytest.mark.asyncio
class TestAsyncSearch:
    """Test async search functionality."""

    async def test_text_search(self, mock_registry_service):
        """Test text search execution."""
        mock_registry_service.search_functionals = AsyncMock(return_value=[])
        engine = SearchEngine(mock_registry_service)

        query = SearchQuery(query_text="FNO", search_type=SearchType.TEXT)
        results = await engine.search(query)

        assert isinstance(results, list)

    async def test_filter_search(self, mock_registry_service):
        """Test filter search execution."""
        mock_registry_service.search_functionals = AsyncMock(return_value=[])
        engine = SearchEngine(mock_registry_service)

        query = SearchQuery(
            functional_type="neural_operator", search_type=SearchType.FILTER
        )
        results = await engine.search(query)

        assert isinstance(results, list)

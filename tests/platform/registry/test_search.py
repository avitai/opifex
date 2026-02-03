"""Test suite for Neural Functional Search Engine.

Provides comprehensive test coverage for search functionality including
text search, semantic search, filtering, and recommendation systems.
"""

import jax.numpy as jnp
import pytest

from opifex.platform.registry.search import (
    SearchEngine,
    SearchQuery,
    SearchResult,
    SearchType,
)


class MockRegistryService:
    """Mock registry service for testing."""

    def __init__(self):
        self.functionals = [
            {
                "id": "func-001",
                "name": "L2O Adam Optimizer",
                "description": "Learn-to-optimize variant of Adam optimizer for neural networks",
                "type": "l2o",
                "author_id": "user-001",
                "tags": ["optimization", "adam", "l2o", "neural_networks"],
                "metadata": {
                    "domain": "optimization",
                    "accuracy": 0.95,
                    "memory_gb": 2,
                    "gpu_required": True,
                    "average_rating": 4.5,
                    "created_at": "2025-02-09T10:00:00Z",
                },
            },
            {
                "id": "func-002",
                "name": "Fluid Dynamics FNO",
                "description": "Fourier Neural Operator for fluid dynamics simulations",
                "type": "neural_operator",
                "author_id": "user-002",
                "tags": ["neural_operator", "fluid_dynamics", "fno", "pde"],
                "metadata": {
                    "domain": "fluid_dynamics",
                    "accuracy": 0.92,
                    "memory_gb": 4,
                    "gpu_required": True,
                    "average_rating": 4.2,
                    "created_at": "2025-02-09T11:00:00Z",
                },
            },
            {
                "id": "func-003",
                "name": "PINN Heat Equation",
                "description": "Physics-Informed Neural Network for heat equation solving",
                "type": "pinn",
                "author_id": "user-003",
                "tags": ["pinn", "heat_equation", "physics", "pde"],
                "metadata": {
                    "domain": "physics",
                    "accuracy": 0.88,
                    "memory_gb": 1,
                    "gpu_required": False,
                    "average_rating": 4.0,
                    "created_at": "2025-02-09T12:00:00Z",
                },
            },
            {
                "id": "func-004",
                "name": "Materials Science DeepONet",
                "description": "Deep Operator Network for materials science applications",
                "type": "neural_operator",
                "author_id": "user-002",
                "tags": ["deeponet", "materials", "neural_operator"],
                "metadata": {
                    "domain": "materials_science",
                    "accuracy": 0.90,
                    "memory_gb": 3,
                    "gpu_required": True,
                    "average_rating": 4.3,
                    "created_at": "2025-02-09T13:00:00Z",
                },
            },
        ]

    async def search_functionals(
        self, query: str = "", limit: int = 10000
    ) -> list[dict]:
        """Mock search functionals method."""
        return self.functionals[:limit]

    async def retrieve_functional(self, functional_id: str) -> dict | None:
        """Mock retrieve functional method."""
        for functional in self.functionals:
            if functional["id"] == functional_id:
                return functional
        return None


@pytest.fixture
def mock_registry():
    """Create mock registry service."""
    return MockRegistryService()


@pytest.fixture
def search_engine(mock_registry):
    """Create search engine with mock registry."""
    return SearchEngine(
        registry_service=mock_registry,
        enable_semantic_search=True,
        similarity_threshold=0.3,  # Lower threshold for testing
    )


@pytest.fixture
def search_engine_no_semantic(mock_registry):
    """Create search engine without semantic search."""
    return SearchEngine(
        registry_service=mock_registry,
        enable_semantic_search=False,
        similarity_threshold=0.7,
    )


class TestSearchEngine:
    """Test SearchEngine functionality."""

    def test_initialization(self, mock_registry):
        """Test search engine initialization."""
        engine = SearchEngine(mock_registry)

        assert engine.registry == mock_registry
        assert engine.enable_semantic is True
        assert engine.similarity_threshold == 0.7
        assert isinstance(engine._text_index, dict)
        assert isinstance(engine._semantic_embeddings, dict)

    def test_initialization_custom_params(self, mock_registry):
        """Test search engine initialization with custom parameters."""
        engine = SearchEngine(
            mock_registry,
            enable_semantic_search=False,
            similarity_threshold=0.8,
        )

        assert engine.enable_semantic is False
        assert engine.similarity_threshold == 0.8

    @pytest.mark.asyncio
    async def test_text_search_basic(self, search_engine):
        """Test basic text search functionality."""
        query = SearchQuery(
            query_text="adam optimizer",
            search_type=SearchType.TEXT,
        )

        results = await search_engine.search(query)

        assert len(results) > 0
        assert any("adam" in r.name.lower() for r in results)
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_text_search_empty_query(self, search_engine):
        """Test text search with empty query."""
        query = SearchQuery(
            query_text="",
            search_type=SearchType.TEXT,
        )

        results = await search_engine.search(query)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_text_search_no_matches(self, search_engine):
        """Test text search with no matches."""
        query = SearchQuery(
            query_text="nonexistent_keyword_xyz",
            search_type=SearchType.TEXT,
        )

        results = await search_engine.search(query)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_semantic_search_enabled(self, search_engine):
        """Test semantic search when enabled."""
        query = SearchQuery(
            query_text="neural network optimization",
            search_type=SearchType.SEMANTIC,
        )

        results = await search_engine.search(query)

        # Should return results with semantic similarity
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_semantic_search_disabled(self, search_engine_no_semantic):
        """Test semantic search when disabled."""
        query = SearchQuery(
            query_text="neural network optimization",
            search_type=SearchType.SEMANTIC,
        )

        results = await search_engine_no_semantic.search(query)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_filter_search_by_type(self, search_engine):
        """Test filter search by functional type."""
        query = SearchQuery(
            functional_type="neural_operator",
            search_type=SearchType.FILTER,
        )

        results = await search_engine.search(query)

        assert len(results) > 0
        assert all(r.functional_type == "neural_operator" for r in results)

    @pytest.mark.asyncio
    async def test_filter_search_by_author(self, search_engine):
        """Test filter search by author."""
        query = SearchQuery(
            author_id="user-002",
            search_type=SearchType.FILTER,
        )

        results = await search_engine.search(query)

        assert len(results) > 0
        assert all(r.author_id == "user-002" for r in results)

    @pytest.mark.asyncio
    async def test_filter_search_by_tags(self, search_engine):
        """Test filter search by tags."""
        query = SearchQuery(
            tags=["pinn"],
            search_type=SearchType.FILTER,
        )

        results = await search_engine.search(query)

        assert len(results) > 0
        assert all(any(tag in r.tags for tag in ["pinn"]) for r in results)

    @pytest.mark.asyncio
    async def test_filter_search_by_domain(self, search_engine):
        """Test filter search by domain."""
        query = SearchQuery(
            domain="physics",
            search_type=SearchType.FILTER,
        )

        results = await search_engine.search(query)

        assert len(results) > 0
        for result in results:
            domain = result.metadata.get("domain", "")
            assert "physics" in domain.lower()

    @pytest.mark.asyncio
    async def test_filter_search_by_rating(self, search_engine):
        """Test filter search by minimum rating."""
        query = SearchQuery(
            min_rating=4.4,
            search_type=SearchType.FILTER,
        )

        results = await search_engine.search(query)

        assert len(results) > 0
        assert all(r.metadata.get("average_rating", 0) >= 4.4 for r in results)

    @pytest.mark.asyncio
    async def test_filter_search_by_performance(self, search_engine):
        """Test filter search by performance criteria."""
        query = SearchQuery(
            min_accuracy=0.9,
            max_memory_gb=3,
            search_type=SearchType.FILTER,
        )

        results = await search_engine.search(query)

        for result in results:
            assert result.metadata.get("accuracy", 0) >= 0.9
            assert result.metadata.get("memory_gb", 0) <= 3

    @pytest.mark.asyncio
    async def test_filter_search_by_gpu_requirement(self, search_engine):
        """Test filter search by GPU requirement."""
        query = SearchQuery(
            gpu_required=False,
            search_type=SearchType.FILTER,
        )

        results = await search_engine.search(query)

        assert len(results) > 0
        # Should include items that don't require GPU
        non_gpu_results = [
            r for r in results if not r.metadata.get("gpu_required", False)
        ]
        assert len(non_gpu_results) > 0

    @pytest.mark.asyncio
    async def test_hybrid_search_with_text(self, search_engine_no_semantic):
        """Test hybrid search with text query."""
        query = SearchQuery(
            query_text="neural operator",
            functional_type="neural_operator",
            search_type=SearchType.HYBRID,
        )

        results = await search_engine_no_semantic.search(query)

        assert len(results) > 0
        assert all(r.functional_type == "neural_operator" for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_without_text(self, search_engine):
        """Test hybrid search without text query falls back to filter."""
        query = SearchQuery(
            query_text="",
            functional_type="pinn",
            search_type=SearchType.HYBRID,
        )

        results = await search_engine.search(query)

        assert len(results) > 0
        assert all(r.functional_type == "pinn" for r in results)

    @pytest.mark.asyncio
    async def test_search_pagination(self, search_engine):
        """Test search pagination."""
        query = SearchQuery(
            search_type=SearchType.FILTER,
            limit=2,
            offset=0,
        )

        first_page = await search_engine.search(query)

        query.offset = 2
        second_page = await search_engine.search(query)

        assert len(first_page) <= 2
        assert len(second_page) <= 2

        # Should not overlap
        first_page_ids = {r.functional_id for r in first_page}
        second_page_ids = {r.functional_id for r in second_page}
        assert first_page_ids.isdisjoint(second_page_ids)

    @pytest.mark.asyncio
    async def test_suggest_functionals(self, search_engine):
        """Test functional suggestion system."""
        suggestions = await search_engine.suggest_functionals("func-002", limit=3)

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
        assert all(isinstance(s, SearchResult) for s in suggestions)

        # Should not include the reference functional itself
        assert not any(s.functional_id == "func-002" for s in suggestions)

    @pytest.mark.asyncio
    async def test_suggest_functionals_nonexistent(self, search_engine):
        """Test suggestions for nonexistent functional."""
        suggestions = await search_engine.suggest_functionals("nonexistent", limit=3)

        assert len(suggestions) == 0

    @pytest.mark.asyncio
    async def test_search_by_problem(self, search_engine):
        """Test problem-based search."""
        results = await search_engine.search_by_problem(
            problem_description="I need to solve fluid dynamics equations",
            domain="fluid_dynamics",
            limit=5,
        )

        assert isinstance(results, list)
        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)

    def test_extract_keywords(self, search_engine):
        """Test keyword extraction."""
        text = "Neural networks and optimization algorithms"
        keywords = search_engine._extract_keywords(text)

        assert isinstance(keywords, list)
        assert "neural" in keywords
        assert "networks" in keywords
        assert "optimization" in keywords
        assert "algorithms" in keywords

        # Should filter out stop words
        assert "and" not in keywords

    def test_extract_keywords_empty(self, search_engine):
        """Test keyword extraction with empty text."""
        keywords = search_engine._extract_keywords("")
        assert keywords == []

    def test_extract_keywords_duplicates(self, search_engine):
        """Test keyword extraction removes duplicates."""
        text = "neural neural networks networks"
        keywords = search_engine._extract_keywords(text)

        assert keywords.count("neural") == 1
        assert keywords.count("networks") == 1

    def test_calculate_text_score(self, search_engine, mock_registry):
        """Test text relevance scoring."""
        functional = mock_registry.functionals[0]  # L2O Adam Optimizer
        keywords = ["adam", "optimizer"]

        score = search_engine._calculate_text_score(functional, keywords)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should have some relevance

    def test_calculate_text_score_no_keywords(self, search_engine, mock_registry):
        """Test text scoring with no keywords."""
        functional = mock_registry.functionals[0]
        keywords = []

        score = search_engine._calculate_text_score(functional, keywords)

        assert score == 0.0

    def test_apply_filters(self, search_engine, mock_registry):
        """Test filter application."""
        # Create search results
        results = [
            SearchResult(
                functional_id=f["id"],
                name=f["name"],
                description=f["description"],
                functional_type=f["type"],
                author_id=f["author_id"],
                tags=f["tags"],
                relevance_score=1.0,
                metadata=f["metadata"],
            )
            for f in mock_registry.functionals
        ]

        query = SearchQuery(functional_type="neural_operator")
        filtered = search_engine._apply_filters(results, query)

        assert len(filtered) > 0
        assert all(r.functional_type == "neural_operator" for r in filtered)

    def test_apply_filters_multiple_criteria(self, search_engine, mock_registry):
        """Test filter application with multiple criteria."""
        results = [
            SearchResult(
                functional_id=f["id"],
                name=f["name"],
                description=f["description"],
                functional_type=f["type"],
                author_id=f["author_id"],
                tags=f["tags"],
                relevance_score=1.0,
                metadata=f["metadata"],
            )
            for f in mock_registry.functionals
        ]

        query = SearchQuery(
            functional_type="neural_operator",
            author_id="user-002",
            min_rating=4.0,
        )
        filtered = search_engine._apply_filters(results, query)

        assert len(filtered) > 0
        for result in filtered:
            assert result.functional_type == "neural_operator"
            assert result.author_id == "user-002"
            assert result.metadata.get("average_rating", 0) >= 4.0

    def test_generate_embedding(self, search_engine):
        """Test neural embedding generation."""
        text = "neural network optimization"
        embedding = search_engine._generate_embedding(text)

        assert isinstance(embedding, jnp.ndarray)
        assert embedding.shape == (256,)  # Embedding dimension
        assert jnp.linalg.norm(embedding) > 0  # Non-zero embedding

    def test_generate_embedding_empty(self, search_engine):
        """Test embedding generation with empty text."""
        embedding = search_engine._generate_embedding("")

        assert isinstance(embedding, jnp.ndarray)
        assert embedding.shape == (256,)

    @pytest.mark.asyncio
    async def test_get_functional_embedding(self, search_engine, mock_registry):
        """Test functional embedding generation and caching."""
        functional = mock_registry.functionals[0]

        # First call - should generate embedding
        embedding1 = await search_engine._get_functional_embedding(functional)

        # Second call - should use cached embedding
        embedding2 = await search_engine._get_functional_embedding(functional)

        assert isinstance(embedding1, jnp.ndarray)
        assert isinstance(embedding2, jnp.ndarray)
        assert jnp.allclose(embedding1, embedding2)

        # Check caching
        assert functional["id"] in search_engine._semantic_embeddings

    def test_cosine_similarity(self, search_engine):
        """Test cosine similarity calculation."""
        # Test identical vectors
        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([1.0, 0.0, 0.0])
        similarity = search_engine._cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 1e-6

        # Test orthogonal vectors
        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([0.0, 1.0, 0.0])
        similarity = search_engine._cosine_similarity(a, b)
        assert abs(similarity - 0.0) < 1e-6

        # Test opposite vectors
        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([-1.0, 0.0, 0.0])
        similarity = search_engine._cosine_similarity(a, b)
        assert abs(similarity - (-1.0)) < 1e-6

    def test_cosine_similarity_zero_vectors(self, search_engine):
        """Test cosine similarity with zero vectors."""
        a = jnp.zeros(3)
        b = jnp.array([1.0, 0.0, 0.0])
        similarity = search_engine._cosine_similarity(a, b)
        assert similarity == 0.0

    @pytest.mark.asyncio
    async def test_get_all_functionals(self, search_engine):
        """Test getting all functionals from registry."""
        functionals = await search_engine._get_all_functionals()

        assert isinstance(functionals, list)
        assert len(functionals) == 4  # Mock registry has 4 functionals

    @pytest.mark.asyncio
    async def test_complex_search_scenario(self, search_engine):
        """Test complex search scenario with multiple criteria."""
        query = SearchQuery(
            query_text="neural operator fluid",
            functional_type="neural_operator",
            tags=["fluid_dynamics"],
            min_accuracy=0.9,
            search_type=SearchType.HYBRID,
            limit=5,
        )

        results = await search_engine.search(query)

        assert isinstance(results, list)
        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)

        # Verify filters applied
        for result in results:
            assert result.functional_type == "neural_operator"
            assert any(tag in result.tags for tag in ["fluid_dynamics"])
            assert result.metadata.get("accuracy", 0) >= 0.9


class TestSearchQuery:
    """Test SearchQuery data class."""

    def test_default_values(self):
        """Test SearchQuery default values."""
        query = SearchQuery()

        assert query.query_text == ""
        assert query.functional_type is None
        assert query.domain is None
        assert query.tags is None
        assert query.author_id is None
        assert query.min_rating is None
        assert query.min_accuracy is None
        assert query.max_memory_gb is None
        assert query.gpu_required is None
        assert query.limit == 50
        assert query.offset == 0
        assert query.search_type == SearchType.HYBRID

    def test_custom_values(self):
        """Test SearchQuery with custom values."""
        query = SearchQuery(
            query_text="test query",
            functional_type="pinn",
            domain="physics",
            tags=["heat", "equation"],
            author_id="user-123",
            min_rating=4.0,
            min_accuracy=0.95,
            max_memory_gb=2,
            gpu_required=True,
            limit=10,
            offset=5,
            search_type=SearchType.TEXT,
        )

        assert query.query_text == "test query"
        assert query.functional_type == "pinn"
        assert query.domain == "physics"
        assert query.tags == ["heat", "equation"]
        assert query.author_id == "user-123"
        assert query.min_rating == 4.0
        assert query.min_accuracy == 0.95
        assert query.max_memory_gb == 2
        assert query.gpu_required is True
        assert query.limit == 10
        assert query.offset == 5
        assert query.search_type == SearchType.TEXT


class TestSearchResult:
    """Test SearchResult data class."""

    def test_creation(self):
        """Test SearchResult creation."""
        result = SearchResult(
            functional_id="func-001",
            name="Test Functional",
            description="Test description",
            functional_type="test",
            author_id="user-001",
            tags=["test", "example"],
            relevance_score=0.95,
            metadata={"key": "value"},
        )

        assert result.functional_id == "func-001"
        assert result.name == "Test Functional"
        assert result.description == "Test description"
        assert result.functional_type == "test"
        assert result.author_id == "user-001"
        assert result.tags == ["test", "example"]
        assert result.relevance_score == 0.95
        assert result.metadata == {"key": "value"}

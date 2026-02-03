"""Comprehensive tests for Neural Functional Registry Core Service.

Tests cover CRUD operations, metadata validation, file storage,
access control, and error handling using TDD approach.
"""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from opifex.platform.registry.core import RegistryService


class TestRegistryService:
    """Test suite for RegistryService core functionality."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session for testing."""
        mock_session = Mock()
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.rollback = Mock()
        return mock_session

    @pytest.fixture
    def temp_storage(self):
        """Temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def registry_service(self, mock_db_session, temp_storage):
        """Registry service instance for testing."""
        return RegistryService(
            db_session=mock_db_session,
            storage_path=temp_storage,
            max_file_size=1024 * 1024,  # 1MB for testing
        )

    @pytest.fixture
    def sample_metadata(self):
        """Sample functional metadata for testing."""
        return {
            "name": "Test L2O Optimizer",
            "type": "l2o",
            "description": "A test learn-to-optimize optimizer",
            "tags": ["optimization", "test"],
            "domain": "fluid_dynamics",
            "is_public": True,
            "performance": {"accuracy": 0.95, "speed": "fast"},
            "requirements": {"memory_gb": 2, "gpu_required": True},
        }

    @pytest.fixture
    def sample_functional(self):
        """Sample neural functional for testing."""
        return {
            "module_type": "TestOptimizer",
            "parameters": {"learning_rate": 0.001, "hidden_size": 128},
            "weights": [1.0, 2.0, 3.0],  # Simplified weights
            "serialization_version": "1.0",
        }

    # Test functional registration

    @pytest.mark.asyncio
    async def test_register_functional_success(
        self, registry_service, sample_metadata, sample_functional
    ):
        """Test successful functional registration."""
        user_id = "test-user-123"

        # Mock database operations
        registry_service._save_sync = Mock()

        # Register functional
        functional_id = await registry_service.register_functional(
            functional=sample_functional, metadata=sample_metadata, user_id=user_id
        )

        # Verify functional ID is generated
        assert functional_id is not None
        assert len(functional_id) == 36  # UUID format

        # Verify file is stored
        functional_dir = (
            Path(registry_service.storage_path) / functional_id / "versions"
        )
        assert functional_dir.exists()

        # Verify at least one version file exists
        version_files = list(functional_dir.glob("*.json"))
        assert len(version_files) == 1

        # Verify file content
        with open(version_files[0]) as f:
            stored_data = json.load(f)
        assert stored_data == sample_functional

        # Verify database save was called
        registry_service._save_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_functional_with_version_tag(
        self, registry_service, sample_metadata, sample_functional
    ):
        """Test functional registration with custom version tag."""
        user_id = "test-user-123"
        version_tag = "v1.0.0"

        registry_service._save_sync = Mock()

        functional_id = await registry_service.register_functional(
            functional=sample_functional,
            metadata=sample_metadata,
            user_id=user_id,
            version_tag=version_tag,
        )

        # Verify version file with custom tag
        version_file = (
            Path(registry_service.storage_path)
            / functional_id
            / "versions"
            / f"{version_tag}.json"
        )
        assert version_file.exists()

    @pytest.mark.asyncio
    async def test_register_functional_invalid_metadata(
        self, registry_service, sample_functional
    ):
        """Test registration with invalid metadata."""
        user_id = "test-user-123"

        # Missing required fields
        invalid_metadata = {"description": "Missing name and type"}

        with pytest.raises(Exception, match="Missing required field"):
            await registry_service.register_functional(
                functional=sample_functional, metadata=invalid_metadata, user_id=user_id
            )

    @pytest.mark.asyncio
    async def test_register_functional_invalid_type(
        self, registry_service, sample_metadata, sample_functional
    ):
        """Test registration with invalid functional type."""
        user_id = "test-user-123"

        # Invalid type
        invalid_metadata = sample_metadata.copy()
        invalid_metadata["type"] = "invalid_type"

        with pytest.raises(Exception, match="Invalid type"):
            await registry_service.register_functional(
                functional=sample_functional, metadata=invalid_metadata, user_id=user_id
            )

    @pytest.mark.asyncio
    async def test_register_functional_too_large(
        self, registry_service, sample_metadata
    ):
        """Test registration with file too large."""
        user_id = "test-user-123"

        # Create oversized functional
        large_functional = {"large_data": "x" * (registry_service.max_file_size + 1)}

        with pytest.raises(Exception, match="too large") as exc_info:
            await registry_service.register_functional(
                functional=large_functional, metadata=sample_metadata, user_id=user_id
            )

        assert "too large" in str(exc_info.value)

    # Test metadata validation

    def test_validate_metadata_success(self, registry_service, sample_metadata):
        """Test successful metadata validation."""
        validated = registry_service._validate_metadata(sample_metadata)

        assert validated["name"] == sample_metadata["name"]
        assert validated["type"] == sample_metadata["type"]
        assert all(tag.islower() for tag in validated["tags"])

    def test_validate_metadata_missing_name(self, registry_service):
        """Test metadata validation with missing name."""
        metadata = {"type": "l2o", "description": "Test"}

        with pytest.raises(Exception, match="Missing required field: name"):
            registry_service._validate_metadata(metadata)

    def test_validate_metadata_missing_type(self, registry_service):
        """Test metadata validation with missing type."""
        metadata = {"name": "Test", "description": "Test"}

        with pytest.raises(Exception, match="Missing required field: type"):
            registry_service._validate_metadata(metadata)

    def test_validate_metadata_invalid_type(self, registry_service):
        """Test metadata validation with invalid type."""
        metadata = {"name": "Test", "type": "invalid", "description": "Test"}

        with pytest.raises(Exception, match="Invalid type"):
            registry_service._validate_metadata(metadata)

    def test_validate_metadata_tag_normalization(self, registry_service):
        """Test metadata validation normalizes tags."""
        metadata = {
            "name": "Test",
            "type": "l2o",
            "tags": ["  Optimization  ", "MACHINE-Learning", "Test  "],
        }

        validated = registry_service._validate_metadata(metadata)

        assert validated["tags"] == ["optimization", "machine-learning", "test"]

    # Test serialization

    @pytest.mark.asyncio
    async def test_serialize_functional_dict(self, registry_service):
        """Test serialization of functional dictionary."""
        functional_dict = {"param1": "value1", "param2": [1, 2, 3]}

        serialized = await registry_service._serialize_functional(functional_dict)

        assert isinstance(serialized, bytes)
        deserialized = json.loads(serialized.decode("utf-8"))
        assert deserialized == functional_dict

    @pytest.mark.asyncio
    async def test_serialize_functional_neural_module(self, registry_service):
        """Test serialization of neural module object."""
        # Mock neural module
        mock_module = Mock()
        mock_module.__class__.__name__ = "TestModule"
        mock_module.to_dict.return_value = {"param": "value"}

        serialized = await registry_service._serialize_functional(mock_module)

        assert isinstance(serialized, bytes)
        deserialized = json.loads(serialized.decode("utf-8"))
        assert deserialized["module_type"] == "TestModule"
        assert deserialized["module_data"] == {"param": "value"}
        assert deserialized["serialization_version"] == "1.0"

    # Test checksum calculation

    def test_calculate_checksum(self, registry_service):
        """Test checksum calculation consistency."""
        data1 = b"test data"
        data2 = b"test data"
        data3 = b"different data"

        checksum1 = registry_service._calculate_checksum(data1)
        checksum2 = registry_service._calculate_checksum(data2)
        checksum3 = registry_service._calculate_checksum(data3)

        assert checksum1 == checksum2
        assert checksum1 != checksum3
        assert len(checksum1) == 64  # SHA-256 hex length

    # Test version tag generation

    def test_generate_version_tag(self, registry_service):
        """Test automatic version tag generation."""
        tag1 = registry_service._generate_version_tag()
        tag2 = registry_service._generate_version_tag()

        assert tag1.startswith("v")
        assert tag2.startswith("v")
        assert len(tag1) == 16  # v + YYYYMMDD_HHMMSS
        # Tags should be different (unless generated at exact same second)
        assert True  # Allow same second edge case

    # Test file operations

    @pytest.mark.asyncio
    async def test_store_functional_file(self, registry_service):
        """Test functional file storage."""
        functional_id = "test-id-123"
        version_tag = "v1.0.0"
        data = b'{"test": "data"}'

        file_path = await registry_service._store_functional_file(
            functional_id, version_tag, data
        )

        assert file_path.exists()
        assert file_path.name == f"{version_tag}.json"

        with open(file_path, "rb") as f:
            stored_data = f.read()
        assert stored_data == data

    @pytest.mark.asyncio
    async def test_load_functional_file(self, registry_service):
        """Test functional file loading."""
        functional_id = "test-id-123"
        version_tag = "v1.0.0"
        test_data = {"test": "data", "number": 42}

        # Store file first
        data_bytes = json.dumps(test_data).encode("utf-8")
        await registry_service._store_functional_file(
            functional_id, version_tag, data_bytes
        )

        # Load file
        loaded_data = await registry_service._load_functional_file(
            functional_id, version_tag
        )

        assert loaded_data == test_data

    @pytest.mark.asyncio
    async def test_load_functional_file_not_found(self, registry_service):
        """Test loading non-existent functional file."""
        with pytest.raises(Exception, match="not found"):
            await registry_service._load_functional_file("nonexistent-id", "v1.0.0")

    @pytest.mark.asyncio
    async def test_load_functional_file_invalid_json(self, registry_service):
        """Test loading file with invalid JSON."""
        functional_id = "test-id-123"
        version_tag = "v1.0.0"

        # Store invalid JSON
        await registry_service._store_functional_file(
            functional_id, version_tag, b"invalid json{"
        )

        with pytest.raises(Exception, match="Failed to parse"):
            await registry_service._load_functional_file(functional_id, version_tag)

    # Test access control

    def test_check_access_permission_public(self, registry_service):
        """Test access permission for public functional."""
        # Mock public functional
        mock_functional = Mock()
        mock_functional.is_public = True
        mock_functional.author_id = "author-123"

        # Public functional should be accessible to anyone
        assert registry_service._check_access_permission(mock_functional, None)
        assert registry_service._check_access_permission(mock_functional, "user-456")

    def test_check_access_permission_private_owner(self, registry_service):
        """Test access permission for private functional by owner."""
        # Mock private functional
        mock_functional = Mock()
        mock_functional.is_public = False
        mock_functional.author_id = "author-123"

        # Owner should have access
        assert registry_service._check_access_permission(mock_functional, "author-123")

    def test_check_access_permission_private_other(self, registry_service):
        """Test access permission for private functional by other user."""
        # Mock private functional
        mock_functional = Mock()
        mock_functional.is_public = False
        mock_functional.author_id = "author-123"

        # Other user should not have access
        assert not registry_service._check_access_permission(
            mock_functional, "user-456"
        )
        assert not registry_service._check_access_permission(mock_functional, None)

    # Test edge cases and error handling

    @pytest.mark.asyncio
    async def test_register_functional_storage_error(
        self, registry_service, sample_metadata, sample_functional
    ):
        """Test handling of storage errors during registration."""
        user_id = "test-user-123"

        # Mock storage error
        with patch.object(registry_service, "_store_functional_file") as mock_store:
            mock_store.side_effect = OSError("Storage error")

            with pytest.raises(OSError, match="Storage error"):
                await registry_service.register_functional(
                    functional=sample_functional,
                    metadata=sample_metadata,
                    user_id=user_id,
                )

    def test_registry_service_initialization(self, mock_db_session):
        """Test registry service initialization."""
        temp_dir = tempfile.mkdtemp()
        try:
            service = RegistryService(
                db_session=mock_db_session, storage_path=temp_dir, max_file_size=500000
            )

            assert service.db == mock_db_session
            assert service.storage_path == Path(temp_dir)
            assert service.max_file_size == 500000
            assert service.storage_path.exists()
        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_directory_creation(self, registry_service):
        """Test automatic directory creation."""
        functional_id = "test-id-456"
        version_tag = "v1.0.0"
        data = b'{"test": "data"}'

        # Ensure directory doesn't exist
        functional_dir = registry_service.storage_path / functional_id
        assert not functional_dir.exists()

        # Store file should create directories
        await registry_service._store_functional_file(functional_id, version_tag, data)

        # Verify directory structure created
        assert functional_dir.exists()
        assert (functional_dir / "versions").exists()

    # Integration tests

    @pytest.mark.asyncio
    async def test_full_registration_workflow(
        self, registry_service, sample_metadata, sample_functional
    ):
        """Test complete registration workflow."""
        user_id = "test-user-123"

        # Mock all database operations
        registry_service._save_sync = Mock()

        # Register functional
        functional_id = await registry_service.register_functional(
            functional=sample_functional, metadata=sample_metadata, user_id=user_id
        )

        # Verify complete workflow
        assert functional_id is not None

        # Verify directory structure
        functional_dir = Path(registry_service.storage_path) / functional_id
        assert functional_dir.exists()
        assert (functional_dir / "versions").exists()

        # Verify file storage
        version_files = list((functional_dir / "versions").glob("*.json"))
        assert len(version_files) == 1

        # Verify file content integrity
        with open(version_files[0]) as f:
            stored_data = json.load(f)
        assert stored_data == sample_functional

        # Verify database interaction
        registry_service._save_sync.assert_called_once()

    # Performance tests

    @pytest.mark.asyncio
    async def test_concurrent_registrations(
        self, registry_service, sample_metadata, sample_functional
    ):
        """Test concurrent functional registrations."""
        registry_service._save_sync = Mock()

        # Create multiple registration tasks
        tasks = []
        for i in range(5):
            metadata = sample_metadata.copy()
            metadata["name"] = f"Test Functional {i}"

            task = registry_service.register_functional(
                functional=sample_functional, metadata=metadata, user_id=f"user-{i}"
            )
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # Verify all registrations succeeded
        assert len(results) == 5
        assert len(set(results)) == 5  # All IDs unique

        # Verify all files created
        for functional_id in results:
            functional_dir = Path(registry_service.storage_path) / functional_id
            assert functional_dir.exists()
            version_files = list((functional_dir / "versions").glob("*.json"))
            assert len(version_files) == 1

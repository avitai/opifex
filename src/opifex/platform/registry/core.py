"""Core Neural Functional Registry Service.

Provides the core functionality for storing, retrieving, and managing
neural functionals in the Opifex community platform. Implements CRUD
operations with metadata management and version control integration.
"""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from flax import nnx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from opifex.platform.registry.models import (
    FunctionalMetadata,
    FunctionalVersion,
    NeuralFunctional,
)


class RegistryService:
    """Core service for neural functional registry operations.

    Handles storage, retrieval, and management of neural functionals
    with comprehensive metadata tracking and version control.
    """

    def __init__(
        self,
        db_session: Session | AsyncSession,
        storage_path: str = "data/registry",
        max_file_size: int = 100 * 1024 * 1024,  # 100MB default
    ):
        """Initialize registry service.

        Args:
            db_session: Database session for metadata storage
            storage_path: File system path for functional storage
            max_file_size: Maximum file size for neural functionals (bytes)
        """
        self.db = db_session
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size

    async def register_functional(
        self,
        functional: nnx.Module | dict[str, Any],
        metadata: dict[str, Any],
        user_id: str,
        version_tag: str | None = None,
    ) -> str:
        """Register a new neural functional in the registry.

        Args:
            functional: Neural functional (module or serialized dict)
            metadata: Functional metadata (description, tags, etc.)
            user_id: ID of user registering the functional
            version_tag: Optional version tag (auto-generated if None)

        Returns:
            Unique functional ID

        Raises:
            HTTPException: If validation fails or storage errors occur
        """
        # Generate unique functional ID
        functional_id = str(uuid.uuid4())

        # Validate metadata
        validated_metadata = self._validate_metadata(metadata)

        # Serialize functional
        serialized_data = await self._serialize_functional(functional)

        # Calculate checksum for integrity
        checksum = self._calculate_checksum(serialized_data)

        # Generate version if not provided
        if version_tag is None:
            version_tag = self._generate_version_tag()

        # Store functional file
        storage_path = await self._store_functional_file(
            functional_id, version_tag, serialized_data
        )

        # Create database records
        db_functional = NeuralFunctional(
            id=functional_id,
            name=validated_metadata["name"],
            description=validated_metadata.get("description", ""),
            functional_type=validated_metadata["type"],
            author_id=user_id,
            tags=validated_metadata.get("tags", []),
            is_public=validated_metadata.get("is_public", True),
            created_at=datetime.now(UTC),
        )

        db_metadata = FunctionalMetadata(
            functional_id=functional_id,
            metadata=validated_metadata,
            checksum=checksum,
            file_size=len(serialized_data),
            storage_path=str(storage_path),
        )

        db_version = FunctionalVersion(
            functional_id=functional_id,
            version_tag=version_tag,
            checksum=checksum,
            created_at=datetime.now(UTC),
            is_latest=True,
        )

        # Save to database
        if isinstance(self.db, AsyncSession):
            await self._save_async(db_functional, db_metadata, db_version)
        else:
            self._save_sync(db_functional, db_metadata, db_version)

        return functional_id

    async def retrieve_functional(
        self,
        functional_id: str,
        version_tag: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve a neural functional from the registry.

        Args:
            functional_id: Unique functional identifier
            version_tag: Specific version (latest if None)
            user_id: User requesting the functional (for access control)

        Returns:
            Dictionary containing functional data and metadata

        Raises:
            HTTPException: If functional not found or access denied
        """
        # Get functional metadata
        functional = await self._get_functional_by_id(functional_id)
        if not functional:
            raise HTTPException(status_code=404, detail="Functional not found")

        # Check access permissions
        if not self._check_access_permission(functional, user_id):
            raise HTTPException(status_code=403, detail="Access denied")

        # Get specific version or latest
        version = await self._get_functional_version(functional_id, version_tag)
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")

        # Load functional data
        functional_data = await self._load_functional_file(
            functional_id, version.version_tag
        )

        # Get metadata
        metadata = await self._get_functional_metadata(functional_id)

        return {
            "id": functional_id,
            "name": functional.name,
            "description": functional.description,
            "type": functional.functional_type,
            "version": version.version_tag,
            "author_id": functional.author_id,
            "tags": functional.tags,
            "created_at": functional.created_at.isoformat(),
            "data": functional_data,
            "metadata": metadata.metadata_json if metadata else {},
            "checksum": version.checksum,
        }

    async def search_functionals(
        self,
        query: str = "",
        functional_type: str | None = None,
        tags: list[str] | None = None,
        author_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Search neural functionals in the registry.

        Args:
            query: Text search query
            functional_type: Filter by functional type
            tags: Filter by tags
            author_id: Filter by author
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of functional summaries matching search criteria
        """
        # Build search filters
        filters = {}
        if functional_type:
            filters["functional_type"] = functional_type
        if author_id:
            filters["author_id"] = author_id

        # Execute search query
        functionals = await self._search_functionals_db(
            query, filters, tags, limit, offset
        )

        # Format results
        results = []
        for functional in functionals:
            results.append(
                {
                    "id": functional.id,
                    "name": functional.name,
                    "description": functional.description,
                    "type": functional.functional_type,
                    "author_id": functional.author_id,
                    "tags": functional.tags,
                    "created_at": functional.created_at.isoformat(),
                    "is_public": functional.is_public,
                }
            )

        return results

    async def delete_functional(
        self,
        functional_id: str,
        user_id: str,
        version_tag: str | None = None,
    ) -> bool:
        """Delete a neural functional or specific version.

        Args:
            functional_id: Unique functional identifier
            user_id: User requesting deletion (must be author or admin)
            version_tag: Specific version to delete (all if None)

        Returns:
            True if deletion successful

        Raises:
            HTTPException: If not authorized or functional not found
        """
        # Get functional and check ownership
        functional = await self._get_functional_by_id(functional_id)
        if not functional:
            raise HTTPException(status_code=404, detail="Functional not found")

        if functional.author_id != user_id:
            # TODO: Add admin check
            raise HTTPException(status_code=403, detail="Not authorized")

        if version_tag:
            # Delete specific version
            await self._delete_version(functional_id, version_tag)
        else:
            # Delete entire functional
            await self._delete_functional_complete(functional_id)

        return True

    # Private helper methods

    def _validate_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Validate and sanitize functional metadata."""
        required_fields = ["name", "type"]
        for field in required_fields:
            if field not in metadata:
                raise HTTPException(
                    status_code=400, detail=f"Missing required field: {field}"
                )

        # Validate functional type
        valid_types = ["l2o", "neural_operator", "pinn", "neural_dft", "custom"]
        if metadata["type"] not in valid_types:
            raise HTTPException(
                status_code=400, detail=f"Invalid type. Must be one of: {valid_types}"
            )

        # Sanitize tags
        if "tags" in metadata:
            metadata["tags"] = [tag.strip().lower() for tag in metadata["tags"]]

        return metadata

    async def _serialize_functional(
        self, functional: nnx.Module | dict[str, Any]
    ) -> bytes:
        """Serialize neural functional for storage."""
        if isinstance(functional, dict):
            # Already serialized
            data = functional
        else:
            # Serialize neural module
            # Check if module has to_dict method before calling it
            # Use getattr with None default to avoid pyright warnings
            to_dict_method = getattr(functional, "to_dict", None)
            if to_dict_method is not None and callable(to_dict_method):
                module_data = to_dict_method()
            else:
                # Fallback for modules without to_dict method
                module_data = {"error": "Module does not support serialization"}

            data = {
                "module_type": functional.__class__.__name__,
                "module_data": module_data,
                "serialization_version": "1.0",
            }

        # Convert to JSON bytes
        json_str = json.dumps(data, indent=2)
        json_bytes = json_str.encode("utf-8")

        # Check size limit
        if len(json_bytes) > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Functional too large: {len(json_bytes)} bytes > "
                    f"{self.max_file_size}"
                ),
            )

        return json_bytes

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum of functional data."""
        return hashlib.sha256(data).hexdigest()

    def _generate_version_tag(self) -> str:
        """Generate automatic version tag."""
        timestamp = datetime.now(UTC)
        return f"v{timestamp.strftime('%Y%m%d_%H%M%S')}"

    async def _store_functional_file(
        self, functional_id: str, version_tag: str, data: bytes
    ) -> Path:
        """Store functional data to file system."""
        # Create directory structure: storage_path/functional_id/versions/
        functional_dir = self.storage_path / functional_id / "versions"
        functional_dir.mkdir(parents=True, exist_ok=True)

        # Store file: version_tag.json
        file_path = functional_dir / f"{version_tag}.json"

        # Write data asynchronously
        def write_file():
            with open(file_path, "wb") as f:
                f.write(data)

        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(None, write_file)

        return file_path

    async def _load_functional_file(
        self, functional_id: str, version_tag: str
    ) -> dict[str, Any]:
        """Load functional data from file system."""
        file_path = (
            self.storage_path / functional_id / "versions" / f"{version_tag}.json"
        )

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Functional file not found")

        # Read file asynchronously
        def read_file():
            with open(file_path, "rb") as f:
                return f.read()

        data = await asyncio.get_event_loop().run_in_executor(None, read_file)

        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to parse functional data: {e}"
            ) from e

    def _check_access_permission(
        self, functional: NeuralFunctional, user_id: str | None
    ) -> bool:
        """Check if user has access to functional."""
        # Public functionals are accessible to all
        # Handle both SQLAlchemy models and plain objects
        is_public = getattr(functional, "is_public", False)
        # Use ternary operator for cleaner code
        public_access = is_public if isinstance(is_public, bool) else bool(is_public)

        if public_access:
            return True

        # Private functionals require ownership
        if user_id is None:
            return False

        # Handle SQLAlchemy UUID comparison
        author_id = getattr(functional, "author_id", None)
        if author_id is None:
            return False

        # Convert to string for comparison to avoid SQLAlchemy type issues
        return str(author_id) == str(user_id)

    # Database Operation Interfaces (to be implemented with specific ORM)

    async def _save_async(self, *objects):
        """Save objects to database asynchronously."""
        # TODO: Implement with actual async database operations

    def _save_sync(self, *objects):
        """Save objects to database synchronously."""
        # TODO: Implement with actual database operations

    async def _get_functional_by_id(self, functional_id: str):
        """Get functional by ID from database."""
        # TODO: Implement database query
        return

    async def _get_functional_version(
        self, functional_id: str, version_tag: str | None
    ):
        """Get functional version from database."""
        # TODO: Implement database query
        return

    async def _get_functional_metadata(self, functional_id: str):
        """Get functional metadata from database."""
        # TODO: Implement database query
        return

    async def _search_functionals_db(
        self, query: str, filters: dict, tags: list[str] | None, limit: int, offset: int
    ):
        """Search functionals in database."""
        # TODO: Implement database search
        return []

    async def _delete_version(self, functional_id: str, version_tag: str):
        """Delete specific version from database and storage."""
        # TODO: Implement version deletion

    async def _delete_functional_complete(self, functional_id: str):
        """Delete entire functional from database and storage."""
        # TODO: Implement complete deletion

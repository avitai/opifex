"""Core Neural Functional Registry Service.

Provides the core functionality for storing, retrieving, and managing
neural functionals in the Opifex community platform. Implements CRUD
operations with metadata management and version control integration.

Persistence is backed by an async SQLAlchemy engine
(:class:`~opifex.platform.registry.database.RegistryDatabase`). The default
backend is on-disk SQLite via ``aiosqlite``, so the registry works out of the
box and survives process restarts; the same ORM transparently targets
PostgreSQL (``postgresql+asyncpg``) in production. The engine/URL is
injectable through the constructor and the engine is created lazily — no
database connection is opened at import time.

The public async API (``register_functional``, ``retrieve_functional``,
``search_functionals``, ``delete_functional``) reads and writes the ORM
records (:class:`~opifex.platform.registry.models.NeuralFunctional`,
:class:`~opifex.platform.registry.models.FunctionalVersion`,
:class:`~opifex.platform.registry.models.FunctionalMetadata`) using
parameterized queries within ``async with`` session scopes.
"""

import asyncio
import hashlib
import json
import shutil
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from flax import nnx
from sqlalchemy import delete as sa_delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from opifex.platform.registry.database import default_sqlite_url, RegistryDatabase
from opifex.platform.registry.exceptions import (
    AccessDenied,
    FunctionalNotFound,
    FunctionalTooLarge,
    SerializationError,
    ValidationError,
    VersionNotFound,
)
from opifex.platform.registry.models import (
    FunctionalMetadata,
    FunctionalVersion,
    NeuralFunctional,
)


class RegistryService:
    """Core service for neural functional registry operations.

    Handles storage, retrieval, and management of neural functionals
    with full metadata tracking and version control.
    """

    def __init__(
        self,
        db_session: Session | AsyncSession | None = None,
        storage_path: str = "data/registry",
        max_file_size: int = 100 * 1024 * 1024,  # 100MB default
        *,
        database_url: str | None = None,
        database: RegistryDatabase | None = None,
    ) -> None:
        """Initialize registry service.

        The persistence backend is resolved with the following precedence:
        an explicitly injected ``database``; otherwise a
        :class:`~opifex.platform.registry.database.RegistryDatabase` built from
        ``database_url``; otherwise — when only a concrete ``db_session`` is
        supplied — that legacy session-backed adapter path is used; otherwise
        the default on-disk SQLite database under ``storage_path``.

        Args:
            db_session: Optional pre-built SQLAlchemy session. Retained for
                callers that drive their own session lifecycle; when omitted
                the service owns an async engine via ``database``.
            storage_path: File-system path for functional artifact storage
                (and the location of the default SQLite database file).
            max_file_size: Maximum file size for neural functionals (bytes).
            database_url: Async SQLAlchemy URL for the persistence backend
                (e.g. ``sqlite+aiosqlite:///…`` or ``postgresql+asyncpg://…``).
                Ignored when ``database`` is provided.
            database: Pre-constructed async database backend (dependency
                injection). Takes precedence over ``database_url``.
        """
        self.db = db_session
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size

        # Resolve the async persistence backend. A concrete (non-async)
        # ``db_session`` signals the legacy session-adapter path used by
        # callers that override the persistence hooks; in that case no async
        # backend is created. Otherwise an async ``RegistryDatabase`` is used,
        # defaulting to on-disk SQLite under ``storage_path``.
        self._database: RegistryDatabase | None
        if database is not None:
            self._database = database
        elif database_url is not None:
            self._database = RegistryDatabase(database_url)
        elif db_session is None:
            self._database = RegistryDatabase(default_sqlite_url(self.storage_path))
        else:
            self._database = None

    async def close(self) -> None:
        """Release the async persistence backend, if the service owns one."""
        if self._database is not None:
            await self._database.close()

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
            ValidationError: If the supplied metadata is invalid.
            FunctionalTooLarge: If the serialised functional exceeds the
                configured size limit.
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
            metadata_json=validated_metadata,
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

        # Save to database. The async backend is used when the service owns
        # one or a concrete async session was supplied; otherwise the legacy
        # synchronous session-adapter hook is dispatched.
        if self._database is not None or isinstance(self.db, AsyncSession):
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
            FunctionalNotFound: If the functional or version does not exist.
            AccessDenied: If the requesting user lacks read access.
        """
        # Get functional metadata
        functional = await self._get_functional_by_id(functional_id)
        if not functional:
            raise FunctionalNotFound("Functional not found")

        # Check access permissions
        if not self._check_access_permission(functional, user_id):
            raise AccessDenied("Access denied")

        # Get specific version or latest
        version = await self._get_functional_version(functional_id, version_tag)
        if not version:
            raise VersionNotFound("Version not found")

        # Load functional data
        functional_data = await self._load_functional_file(functional_id, version.version_tag)

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
        functionals = await self._search_functionals_db(query, filters, tags, limit, offset)

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
            FunctionalNotFound: If the functional does not exist.
            AccessDenied: If the requesting user is not the author.
        """
        # Get functional and check ownership
        functional = await self._get_functional_by_id(functional_id)
        if not functional:
            raise FunctionalNotFound("Functional not found")

        if functional.author_id != user_id:
            # Only the author can delete; admin override not yet implemented
            raise AccessDenied("Not authorized to delete this functional")

        if version_tag:
            # Delete specific version
            await self._delete_version(functional_id, version_tag)
        else:
            # Delete entire functional
            await self._delete_functional_complete(functional_id)

        return True

    # Private helper methods

    def _validate_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Validate metadata and return a normalised copy.

        The caller's dictionary is never mutated (POLA / R5): a new dict is
        built with sanitised tags and returned. Validation failures surface
        as :class:`~opifex.platform.registry.exceptions.ValidationError`.
        """
        required_fields = ["name", "type"]
        for field in required_fields:
            if field not in metadata:
                raise ValidationError(f"Missing required field: {field}")

        # Validate functional type
        valid_types = ["l2o", "neural_operator", "pinn", "neural_dft", "custom"]
        if metadata["type"] not in valid_types:
            raise ValidationError(f"Invalid type. Must be one of: {valid_types}")

        # Build a normalised copy without mutating the caller's dict.
        normalised = {**metadata}
        if "tags" in normalised:
            normalised["tags"] = [tag.strip().lower() for tag in normalised["tags"]]

        return normalised

    async def _serialize_functional(self, functional: nnx.Module | dict[str, Any]) -> bytes:
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
            raise FunctionalTooLarge(
                f"Functional too large: {len(json_bytes)} bytes > {self.max_file_size}"
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
        def write_file() -> None:
            with open(file_path, "wb") as f:
                f.write(data)

        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(None, write_file)

        return file_path

    async def _load_functional_file(self, functional_id: str, version_tag: str) -> dict[str, Any]:
        """Load functional data from file system."""
        file_path = self.storage_path / functional_id / "versions" / f"{version_tag}.json"

        if not file_path.exists():
            raise VersionNotFound("Functional file not found")

        # Read file asynchronously
        def read_file():
            with open(file_path, "rb") as f:
                return f.read()

        data = await asyncio.get_event_loop().run_in_executor(None, read_file)

        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise SerializationError(f"Failed to parse functional data: {e}") from e

    def _check_access_permission(self, functional: NeuralFunctional, user_id: str | None) -> bool:
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

    # Database Operation Interfaces — async SQLAlchemy ORM persistence.

    _NO_BACKEND = (
        "RegistryService has no async persistence backend; supply a "
        "database_url / database, or override this hook in a session adapter."
    )

    def _require_database(self) -> RegistryDatabase:
        """Return the owned async backend or fail fast if absent."""
        if self._database is None:
            raise RuntimeError(self._NO_BACKEND)
        return self._database

    async def _save_async(self, *objects: object) -> None:
        """Persist ORM records within a single transactional session."""
        database = self._require_database()
        async with database.session() as session:
            for obj in objects:
                session.add(obj)

    def _save_sync(self, *objects: object) -> None:
        """Persist ORM records synchronously (legacy session adapter hook).

        Raises:
            TypeError: When no synchronous :class:`~sqlalchemy.orm.Session` is
                available; the async backend path uses :meth:`_save_async`.
        """
        if not isinstance(self.db, Session):
            raise TypeError(self._NO_BACKEND)
        for obj in objects:
            self.db.add(obj)
        self.db.commit()

    async def _get_functional_by_id(self, functional_id: str) -> NeuralFunctional | None:
        """Load a non-deleted functional record by id, or ``None``."""
        database = self._require_database()
        async with database.session() as session:
            statement = select(NeuralFunctional).where(
                NeuralFunctional.id == uuid.UUID(functional_id),
                NeuralFunctional.is_deleted.is_(False),
            )
            return (await session.execute(statement)).scalar_one_or_none()

    async def _get_functional_version(
        self, functional_id: str, version_tag: str | None
    ) -> FunctionalVersion | None:
        """Load a specific version, or the latest when ``version_tag`` is None."""
        database = self._require_database()
        async with database.session() as session:
            statement = select(FunctionalVersion).where(
                FunctionalVersion.functional_id == uuid.UUID(functional_id)
            )
            if version_tag is not None:
                statement = statement.where(FunctionalVersion.version_tag == version_tag)
            else:
                statement = statement.where(FunctionalVersion.is_latest.is_(True))
            return (await session.execute(statement)).scalars().first()

    async def _get_functional_metadata(self, functional_id: str) -> FunctionalMetadata | None:
        """Load the metadata record for a functional, or ``None``."""
        database = self._require_database()
        async with database.session() as session:
            statement = select(FunctionalMetadata).where(
                FunctionalMetadata.functional_id == uuid.UUID(functional_id)
            )
            return (await session.execute(statement)).scalar_one_or_none()

    async def _search_functionals_db(
        self,
        query: str,
        filters: dict[str, Any],
        tags: list[str] | None,
        limit: int,
        offset: int,
    ) -> list[NeuralFunctional]:
        """Search functionals by text, scalar filters and tags."""
        database = self._require_database()
        async with database.session() as session:
            statement = select(NeuralFunctional).where(NeuralFunctional.is_deleted.is_(False))
            if query:
                statement = statement.where(NeuralFunctional.name.ilike(f"%{query}%"))
            functional_type = filters.get("functional_type")
            if functional_type:
                statement = statement.where(NeuralFunctional.functional_type == functional_type)
            author_id = filters.get("author_id")
            if author_id:
                statement = statement.where(NeuralFunctional.author_id == author_id)
            statement = (
                statement.order_by(NeuralFunctional.created_at.desc()).limit(limit).offset(offset)
            )
            candidates = list((await session.execute(statement)).scalars().all())

        if not tags:
            return candidates
        # Tag membership is filtered in Python so the portable ``StringArray``
        # column (JSON-encoded on SQLite) matches identically across dialects.
        required = set(tags)
        return [c for c in candidates if required.issubset(set(c.tags or []))]

    async def _delete_version(self, functional_id: str, version_tag: str) -> None:
        """Delete one version row and its on-disk artifact."""
        database = self._require_database()
        functional_uuid = uuid.UUID(functional_id)
        async with database.session() as session:
            statement = sa_delete(FunctionalVersion).where(
                FunctionalVersion.functional_id == functional_uuid,
                FunctionalVersion.version_tag == version_tag,
            )
            await session.execute(statement)
        self._remove_version_file(functional_id, version_tag)

    async def _delete_functional_complete(self, functional_id: str) -> None:
        """Soft-delete the functional and remove all on-disk artifacts.

        The row is flagged ``is_deleted`` (preserving audit/download history
        via the soft-delete columns) while the on-disk artifacts are removed.
        """
        database = self._require_database()
        functional_uuid = uuid.UUID(functional_id)
        async with database.session() as session:
            statement = select(NeuralFunctional).where(NeuralFunctional.id == functional_uuid)
            functional = (await session.execute(statement)).scalar_one_or_none()
            if functional is not None:
                functional.is_deleted = True
                functional.deleted_at = datetime.now(UTC)
        self._remove_functional_dir(functional_id)

    def _remove_version_file(self, functional_id: str, version_tag: str) -> None:
        """Delete a single version artifact file if present."""
        file_path = self.storage_path / functional_id / "versions" / f"{version_tag}.json"
        file_path.unlink(missing_ok=True)

    def _remove_functional_dir(self, functional_id: str) -> None:
        """Recursively delete a functional's artifact directory if present."""
        functional_dir = self.storage_path / functional_id
        if functional_dir.exists():
            shutil.rmtree(functional_dir)

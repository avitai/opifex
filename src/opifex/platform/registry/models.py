"""Database models for Neural Functional Registry.

Defines SQLAlchemy models for storing neural functional metadata,
versions, and related information in the community platform database.

The column types are dialect-agnostic: :class:`GUID` and :class:`StringArray`
are :class:`~sqlalchemy.types.TypeDecorator` adapters that use native
PostgreSQL ``UUID`` / ``ARRAY`` columns under the ``postgresql`` dialect and
portable ``CHAR(36)`` / JSON-encoded ``TEXT`` columns everywhere else (e.g.
SQLite). This lets the same ORM back both the production PostgreSQL deployment
and the default on-disk SQLite registry without per-dialect model variants.

Mappings use SQLAlchemy 2.0 typed ``Mapped[...]`` / ``mapped_column()``
declarations so attribute access on instances is correctly typed.
"""

import json
import uuid
from datetime import datetime, UTC
from typing import Any

from sqlalchemy import (
    CHAR,
    ForeignKey,
    JSON,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY, UUID as PG_UUID
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeEngine


class Base(DeclarativeBase):
    """Declarative base for all registry ORM models."""


class GUID(TypeDecorator[uuid.UUID]):
    """Platform-independent UUID column.

    Uses PostgreSQL's native ``UUID`` type when available and falls back to a
    ``CHAR(36)`` string elsewhere. Values are stored and returned as
    :class:`uuid.UUID` instances regardless of backend.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        """Return the native type for ``dialect`` (UUID on PostgreSQL)."""
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value: object, dialect: Dialect) -> str | None:  # noqa: ARG002 - SQLAlchemy TypeDecorator interface receives the dialect
        """Coerce a Python value to its canonical string representation.

        Both the native PostgreSQL ``UUID`` and the portable ``CHAR(36)``
        fallback accept the canonical hyphenated string form.
        """
        if value is None:
            return None
        coerced = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return str(coerced)

    def process_result_value(self, value: object, dialect: Dialect) -> uuid.UUID | None:  # noqa: ARG002 - SQLAlchemy TypeDecorator interface receives the dialect
        """Coerce a stored value back to :class:`uuid.UUID`."""
        if value is None:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class StringArray(TypeDecorator[list[str]]):
    """Platform-independent list-of-strings column.

    Uses PostgreSQL's native ``ARRAY(String)`` when available and falls back
    to a JSON-encoded ``TEXT`` column elsewhere. Values are stored and
    returned as ``list[str]`` regardless of backend.
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        """Return the native type for ``dialect`` (ARRAY on PostgreSQL)."""
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_ARRAY(String))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: object, dialect: Dialect) -> object:
        """Coerce a Python list to its stored representation."""
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(list(value))  # type: ignore[arg-type]

    def process_result_value(self, value: object, dialect: Dialect) -> list[str] | None:
        """Coerce a stored value back to ``list[str]``."""
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)  # type: ignore[arg-type]
        return list(json.loads(value))  # type: ignore[arg-type]


def _now_utc() -> datetime:
    """Return the current timezone-aware UTC timestamp (column default)."""
    return datetime.now(UTC)


class NeuralFunctional(Base):
    """Core neural functional record.

    Stores basic information about a neural functional including
    metadata, authorship, and access control information.
    """

    __tablename__ = "neural_functionals"

    # Primary identification
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)

    # Functional classification: 'l2o', 'neural_operator', 'pinn', 'neural_dft', 'custom'
    functional_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Authorship and access. ``author_id`` is an opaque external identity
    # reference (not necessarily a UUID).
    author_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    is_public: Mapped[bool] = mapped_column(default=True, nullable=False)

    # Categorization
    tags: Mapped[list[str]] = mapped_column(StringArray(), default=list)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        default=_now_utc,
        nullable=False,
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        default=_now_utc,
        onupdate=_now_utc,
    )

    # Soft delete support
    is_deleted: Mapped[bool] = mapped_column(default=False, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    metadata_record: Mapped["FunctionalMetadata"] = relationship(
        back_populates="functional",
        uselist=False,
        cascade="all, delete-orphan",
    )
    versions: Mapped[list["FunctionalVersion"]] = relationship(
        back_populates="functional", cascade="all, delete-orphan"
    )
    downloads: Mapped[list["FunctionalDownload"]] = relationship(
        back_populates="functional", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        """Return a concise developer representation."""
        return (
            f"<NeuralFunctional(id={self.id}, name='{self.name}', type='{self.functional_type}')>"
        )


class FunctionalMetadata(Base):
    """Extended metadata for neural functionals.

    Stores detailed metadata including performance metrics,
    computational requirements, and scientific domain information.
    """

    __tablename__ = "functional_metadata"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    functional_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("neural_functionals.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Metadata storage. Structure example::
    #
    #   {"domain": "fluid_dynamics", "performance": {"accuracy": 0.95},
    #    "requirements": {"memory_gb": 4}, "license": "MIT"}
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # File information
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA-256
    file_size: Mapped[int] = mapped_column(nullable=False)  # bytes
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # Validation status: 'pending', 'passed', 'failed', 'warning'
    validation_status: Mapped[str | None] = mapped_column(String(20), default="pending")
    validation_report: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=_now_utc, nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(default=_now_utc, onupdate=_now_utc)

    # Relationships
    functional: Mapped["NeuralFunctional"] = relationship(back_populates="metadata_record")

    def __repr__(self) -> str:
        """Return a concise developer representation."""
        return (
            f"<FunctionalMetadata(functional_id={self.functional_id}, "
            f"checksum={self.checksum[:8]}...)>"
        )


class FunctionalVersion(Base):
    """Version tracking for neural functionals.

    Maintains version history with checksums and change tracking
    for reproducibility and rollback capabilities.
    """

    __tablename__ = "functional_versions"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    functional_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("neural_functionals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Version information
    version_tag: Mapped[str] = mapped_column(String(100), nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA-256
    is_latest: Mapped[bool] = mapped_column(default=False, nullable=False)

    # Change tracking
    change_summary: Mapped[str | None] = mapped_column(Text)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("functional_versions.id"), nullable=True
    )

    # Performance benchmarks (optional)
    benchmark_results: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=_now_utc, nullable=False)

    # Relationships
    functional: Mapped["NeuralFunctional"] = relationship(back_populates="versions")
    parent_version: Mapped["FunctionalVersion | None"] = relationship(remote_side=[id])

    # Constraints
    __table_args__ = (
        UniqueConstraint("functional_id", "version_tag", name="uq_functional_version"),
    )

    def __repr__(self) -> str:
        """Return a concise developer representation."""
        return (
            f"<FunctionalVersion(functional_id={self.functional_id}, version='{self.version_tag}')>"
        )


class FunctionalDownload(Base):
    """Download tracking for neural functionals.

    Tracks downloads for analytics and usage monitoring
    while respecting user privacy.
    """

    __tablename__ = "functional_downloads"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    functional_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("neural_functionals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Download information
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    version_tag: Mapped[str] = mapped_column(String(100), nullable=False)
    user_agent: Mapped[str | None] = mapped_column(String(500))
    ip_address_hash: Mapped[str | None] = mapped_column(String(64))  # Hashed for privacy

    # Context: 'api', 'web', 'cli'
    download_context: Mapped[str | None] = mapped_column(String(50))

    # Timestamp
    downloaded_at: Mapped[datetime] = mapped_column(default=_now_utc, nullable=False, index=True)

    # Relationships
    functional: Mapped["NeuralFunctional"] = relationship(back_populates="downloads")

    def __repr__(self) -> str:
        """Return a concise developer representation."""
        return (
            f"<FunctionalDownload(functional_id={self.functional_id}, "
            f"version='{self.version_tag}')>"
        )


class FunctionalRating(Base):
    """User ratings and reviews for neural functionals.

    Enables community feedback and quality assessment
    of registered neural functionals.
    """

    __tablename__ = "functional_ratings"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    functional_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("neural_functionals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Rating information
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    rating: Mapped[int] = mapped_column(nullable=False)  # 1-5 stars
    review_text: Mapped[str | None] = mapped_column(Text)

    # Rating categories (1-5)
    ease_of_use: Mapped[int | None] = mapped_column()
    performance: Mapped[int | None] = mapped_column()
    documentation: Mapped[int | None] = mapped_column()
    reproducibility: Mapped[int | None] = mapped_column()

    # Moderation
    is_approved: Mapped[bool] = mapped_column(default=True, nullable=False)
    moderation_note: Mapped[str | None] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=_now_utc, nullable=False)
    updated_at: Mapped[datetime | None] = mapped_column(default=_now_utc, onupdate=_now_utc)

    # Relationships
    functional: Mapped["NeuralFunctional"] = relationship()

    # Constraints
    __table_args__ = (
        UniqueConstraint("functional_id", "user_id", name="uq_user_functional_rating"),
    )

    def __repr__(self) -> str:
        """Return a concise developer representation."""
        return f"<FunctionalRating(functional_id={self.functional_id}, rating={self.rating})>"


class FunctionalTag(Base):
    """Tag definitions for categorizing neural functionals.

    Provides structured tagging system with descriptions
    and hierarchical relationships for better organization.
    """

    __tablename__ = "functional_tags"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)

    # Tag information
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    color: Mapped[str | None] = mapped_column(String(7))  # Hex color code

    # Hierarchy support
    parent_tag_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("functional_tags.id"), nullable=True
    )

    # Usage tracking
    usage_count: Mapped[int] = mapped_column(default=0, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=_now_utc, nullable=False)

    # Relationships
    parent_tag: Mapped["FunctionalTag | None"] = relationship(remote_side=[id])

    def __repr__(self) -> str:
        """Return a concise developer representation."""
        return f"<FunctionalTag(name='{self.name}', usage_count={self.usage_count})>"

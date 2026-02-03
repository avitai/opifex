"""Database models for Neural Functional Registry.

Defines SQLAlchemy models for storing neural functional metadata,
versions, and related information in the community platform database.
"""

import uuid
from datetime import datetime, UTC

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship


Base = declarative_base()


class NeuralFunctional(Base):
    """Core neural functional record.

    Stores basic information about a neural functional including
    metadata, authorship, and access control information.
    """

    __tablename__ = "neural_functionals"

    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)

    # Functional classification
    functional_type = Column(String(50), nullable=False, index=True)
    # Types: 'l2o', 'neural_operator', 'pinn', 'neural_dft', 'custom'

    # Authorship and access
    author_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    is_public = Column(Boolean, default=True, nullable=False)

    # Categorization
    tags = Column(ARRAY(String), default=list)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Soft delete support
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    metadata_record = relationship(
        "FunctionalMetadata",
        back_populates="functional",
        uselist=False,
        cascade="all, delete-orphan",
    )
    versions = relationship(
        "FunctionalVersion", back_populates="functional", cascade="all, delete-orphan"
    )
    downloads = relationship(
        "FunctionalDownload", back_populates="functional", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<NeuralFunctional(id={self.id}, name='{self.name}', "
            f"type='{self.functional_type}')>"
        )


class FunctionalMetadata(Base):
    """Extended metadata for neural functionals.

    Stores detailed metadata including performance metrics,
    computational requirements, and scientific domain information.
    """

    __tablename__ = "functional_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    functional_id = Column(
        UUID(as_uuid=True),
        ForeignKey("neural_functionals.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Metadata storage
    metadata_json = Column(JSON, nullable=False, default=dict)
    # Structure: {
    #   "domain": "fluid_dynamics",
    #   "problem_types": ["pde_solving", "optimization"],
    #   "performance": {"accuracy": 0.95, "speed": "fast"},
    #   "requirements": {"memory_gb": 4, "gpu_required": true},
    #   "papers": ["doi:10.1000/example"],
    #   "license": "MIT",
    #   "dependencies": ["jax>=0.4.0", "flax>=0.7.0"]
    # }

    # File information
    checksum = Column(String(64), nullable=False)  # SHA-256
    file_size = Column(Integer, nullable=False)  # bytes
    storage_path = Column(String(500), nullable=False)

    # Validation status
    validation_status = Column(String(20), default="pending")
    # Status: 'pending', 'passed', 'failed', 'warning'
    validation_report = Column(JSON)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Relationships
    functional = relationship("NeuralFunctional", back_populates="metadata_record")

    def __repr__(self) -> str:
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

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    functional_id = Column(
        UUID(as_uuid=True),
        ForeignKey("neural_functionals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Version information
    version_tag = Column(String(100), nullable=False)
    checksum = Column(String(64), nullable=False)  # SHA-256
    is_latest = Column(Boolean, default=False, nullable=False)

    # Change tracking
    change_summary = Column(Text)
    parent_version_id = Column(
        UUID(as_uuid=True), ForeignKey("functional_versions.id"), nullable=True
    )

    # Performance benchmarks (optional)
    benchmark_results = Column(JSON)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    functional = relationship("NeuralFunctional", back_populates="versions")
    parent_version = relationship("FunctionalVersion", remote_side=[id])

    # Constraints
    __table_args__ = (
        UniqueConstraint("functional_id", "version_tag", name="uq_functional_version"),
    )

    def __repr__(self) -> str:
        return (
            f"<FunctionalVersion(functional_id={self.functional_id}, "
            f"version='{self.version_tag}')>"
        )


class FunctionalDownload(Base):
    """Download tracking for neural functionals.

    Tracks downloads for analytics and usage monitoring
    while respecting user privacy.
    """

    __tablename__ = "functional_downloads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    functional_id = Column(
        UUID(as_uuid=True),
        ForeignKey("neural_functionals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Download information
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)  # Anonymous allowed
    version_tag = Column(String(100), nullable=False)
    user_agent = Column(String(500))
    ip_address_hash = Column(String(64))  # Hashed for privacy

    # Context
    download_context = Column(String(50))  # 'api', 'web', 'cli'

    # Timestamp
    downloaded_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )

    # Relationships
    functional = relationship("NeuralFunctional", back_populates="downloads")

    def __repr__(self) -> str:
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

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    functional_id = Column(
        UUID(as_uuid=True),
        ForeignKey("neural_functionals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Rating information
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    rating = Column(Integer, nullable=False)  # 1-5 stars
    review_text = Column(Text)

    # Rating categories
    ease_of_use = Column(Integer)  # 1-5
    performance = Column(Integer)  # 1-5
    documentation = Column(Integer)  # 1-5
    reproducibility = Column(Integer)  # 1-5

    # Moderation
    is_approved = Column(Boolean, default=True, nullable=False)
    moderation_note = Column(Text)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Relationships
    functional = relationship("NeuralFunctional")

    # Constraints
    __table_args__ = (
        UniqueConstraint("functional_id", "user_id", name="uq_user_functional_rating"),
    )

    def __repr__(self) -> str:
        return (
            f"<FunctionalRating(functional_id={self.functional_id}, "
            f"rating={self.rating})>"
        )


class FunctionalTag(Base):
    """Tag definitions for categorizing neural functionals.

    Provides structured tagging system with descriptions
    and hierarchical relationships for better organization.
    """

    __tablename__ = "functional_tags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Tag information
    name = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(Text)
    color = Column(String(7))  # Hex color code

    # Hierarchy support
    parent_tag_id = Column(
        UUID(as_uuid=True), ForeignKey("functional_tags.id"), nullable=True
    )

    # Usage tracking
    usage_count = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    parent_tag = relationship("FunctionalTag", remote_side=[id])

    def __repr__(self) -> str:
        return f"<FunctionalTag(name='{self.name}', usage_count={self.usage_count})>"

"""Version Management System for Neural Functionals.

Provides Git-based version control, dependency tracking, and branch management
for neural functionals in the Opifex registry.
"""

import json
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class VersionStatus(Enum):
    """Status of a version."""

    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class MergeStrategy(Enum):
    """Merge strategy for version management."""

    FAST_FORWARD = "fast_forward"
    MERGE_COMMIT = "merge_commit"
    SQUASH = "squash"
    REBASE = "rebase"


@dataclass
class Version:
    """Version metadata for a neural functional."""

    id: str
    functional_id: str
    version_tag: str
    commit_hash: str
    parent_version: str | None = None
    branch: str = "main"
    status: VersionStatus = VersionStatus.DRAFT
    author_id: str = ""
    message: str = ""
    changes: list[str] = field(default_factory=list)
    dependencies: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    size_bytes: int = 0


@dataclass
class Branch:
    """Branch information for version management."""

    name: str
    functional_id: str
    head_commit: str
    created_from: str | None = None
    author_id: str = ""
    description: str = ""
    is_protected: bool = False
    created_at: str = ""


@dataclass
class VersionDiff:
    """Differences between two versions."""

    version_a: str
    version_b: str
    added_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    removed_files: list[str] = field(default_factory=list)
    metadata_changes: dict[str, Any] = field(default_factory=dict)


class VersionManager:
    """Version management system for neural functionals.

    Provides Git-based version control with dependency tracking,
    branch management, and merge capabilities.
    """

    def __init__(
        self,
        storage_root: Path,
        registry_service,
        enable_git: bool = True,
    ):
        """Initialize version manager.

        Args:
            storage_root: Root directory for functional storage
            registry_service: Registry service for data access
            enable_git: Whether to use Git for version control
        """
        self.storage_root = Path(storage_root)
        self.registry = registry_service
        self.enable_git = enable_git

        # Ensure storage directory exists
        self.storage_root.mkdir(parents=True, exist_ok=True)

        # Initialize Git repository if enabled
        if self.enable_git:
            self._initialize_git_repo()

    def _initialize_git_repo(self) -> None:
        """Initialize Git repository in storage root."""
        if not (self.storage_root / ".git").exists():
            subprocess.run(
                ["git", "init"],
                cwd=self.storage_root,
                check=True,
                capture_output=True,
            )

            # Configure Git
            subprocess.run(
                ["git", "config", "user.name", "Opifex Registry"],
                cwd=self.storage_root,
                check=False,
            )
            subprocess.run(
                ["git", "config", "user.email", "registry@opifex.io"],
                cwd=self.storage_root,
                check=False,
            )

    async def create_version(
        self,
        functional_id: str,
        version_tag: str,
        author_id: str,
        message: str,
        files: dict[str, bytes],
        metadata: dict[str, Any] | None = None,
        parent_version: str | None = None,
        branch: str = "main",
    ) -> Version:
        """Create a new version of a functional.

        Args:
            functional_id: ID of the functional
            version_tag: Version tag (e.g., "v1.0.0")
            author_id: ID of the author
            message: Commit message
            files: Dictionary of file names to content
            metadata: Version metadata
            parent_version: Parent version ID
            branch: Branch name

        Returns:
            Created version object
        """
        # Create functional directory
        functional_dir = self.storage_root / functional_id
        functional_dir.mkdir(parents=True, exist_ok=True)

        # Switch to branch if needed
        if self.enable_git:
            await self._ensure_branch(functional_id, branch, author_id)

        # Write files
        total_size = 0
        changed_files = []

        for file_name, content in files.items():
            file_path = functional_dir / file_name
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file content
            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(str(content), encoding="utf-8")

            total_size += len(content)
            changed_files.append(file_name)

        # Write version metadata
        version_metadata = {
            "functional_id": functional_id,
            "version_tag": version_tag,
            "author_id": author_id,
            "message": message,
            "parent_version": parent_version,
            "branch": branch,
            "files": list(files.keys()),
            "metadata": metadata or {},
            "size_bytes": total_size,
        }

        metadata_file = functional_dir / f".version-{version_tag}.json"
        metadata_file.write_text(json.dumps(version_metadata, indent=2))

        # Commit to Git if enabled
        commit_hash = ""
        if self.enable_git:
            commit_hash = await self._commit_changes(
                functional_id, version_tag, message, author_id
            )

        # Create version object
        version = Version(
            id=f"{functional_id}-{version_tag}",
            functional_id=functional_id,
            version_tag=version_tag,
            commit_hash=commit_hash,
            parent_version=parent_version,
            branch=branch,
            status=VersionStatus.DRAFT,
            author_id=author_id,
            message=message,
            changes=changed_files,
            metadata=metadata or {},
            size_bytes=total_size,
        )

        # Tag version if on main branch
        if branch == "main" and self.enable_git:
            await self._create_git_tag(functional_id, version_tag, commit_hash)

        return version

    async def get_version(self, functional_id: str, version_tag: str) -> Version | None:
        """Get a specific version of a functional.

        Args:
            functional_id: ID of the functional
            version_tag: Version tag to retrieve

        Returns:
            Version object or None if not found
        """
        functional_dir = self.storage_root / functional_id
        metadata_file = functional_dir / f".version-{version_tag}.json"

        if not metadata_file.exists():
            return None

        # Load metadata
        metadata = json.loads(metadata_file.read_text())

        # Get commit hash if Git is enabled
        commit_hash = ""
        if self.enable_git:
            commit_hash = await self._get_commit_for_tag(functional_id, version_tag)

        return Version(
            id=f"{functional_id}-{version_tag}",
            functional_id=functional_id,
            version_tag=version_tag,
            commit_hash=commit_hash,
            parent_version=metadata.get("parent_version"),
            branch=metadata.get("branch", "main"),
            author_id=metadata.get("author_id", ""),
            message=metadata.get("message", ""),
            changes=metadata.get("files", []),
            metadata=metadata.get("metadata", {}),
            size_bytes=metadata.get("size_bytes", 0),
        )

    async def list_versions(
        self, functional_id: str, branch: str | None = None
    ) -> list[Version]:
        """List all versions of a functional.

        Args:
            functional_id: ID of the functional
            branch: Filter by branch (optional)

        Returns:
            List of versions
        """
        functional_dir = self.storage_root / functional_id
        if not functional_dir.exists():
            return []

        versions = []
        for metadata_file in functional_dir.glob(".version-*.json"):
            version_tag = metadata_file.stem.replace(".version-", "")
            version = await self.get_version(functional_id, version_tag)

            if version and (branch is None or version.branch == branch):
                versions.append(version)

        # Sort by creation time
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions

    async def delete_version(
        self, functional_id: str, version_tag: str, author_id: str
    ) -> bool:
        """Delete a version of a functional.

        Args:
            functional_id: ID of the functional
            version_tag: Version tag to delete
            author_id: ID of the requesting user

        Returns:
            True if deleted successfully
        """
        version = await self.get_version(functional_id, version_tag)
        if not version:
            return False

        # Check authorization (only author or admin can delete)
        if version.author_id != author_id:
            # Would check admin permissions here
            return False

        # Remove version metadata file
        functional_dir = self.storage_root / functional_id
        metadata_file = functional_dir / f".version-{version_tag}.json"

        if metadata_file.exists():
            metadata_file.unlink()

        # Remove Git tag if exists
        if self.enable_git:
            await self._delete_git_tag(functional_id, version_tag)

        return True

    async def create_branch(
        self,
        functional_id: str,
        branch_name: str,
        author_id: str,
        from_version: str | None = None,
        description: str = "",
    ) -> Branch:
        """Create a new branch for development.

        Args:
            functional_id: ID of the functional
            branch_name: Name of the new branch
            author_id: ID of the author
            from_version: Version to branch from (default: latest)
            description: Branch description

        Returns:
            Created branch object
        """
        if not self.enable_git:
            raise ValueError("Git must be enabled for branch management")

        # Get source commit
        if from_version:
            source_commit = await self._get_commit_for_tag(functional_id, from_version)
        else:
            source_commit = await self._get_latest_commit(functional_id)

        # Create Git branch
        functional_dir = self.storage_root / functional_id
        subprocess.run(
            ["git", "checkout", "-b", branch_name, source_commit],
            cwd=functional_dir,
            check=True,
            capture_output=True,
        )

        return Branch(
            name=branch_name,
            functional_id=functional_id,
            head_commit=source_commit,
            created_from=from_version,
            author_id=author_id,
            description=description,
        )

    async def merge_branch(
        self,
        functional_id: str,
        source_branch: str,
        target_branch: str,
        author_id: str,
        strategy: MergeStrategy = MergeStrategy.MERGE_COMMIT,
        message: str = "",
    ) -> bool:
        """Merge one branch into another.

        Args:
            functional_id: ID of the functional
            source_branch: Source branch to merge from
            target_branch: Target branch to merge into
            author_id: ID of the user performing merge
            strategy: Merge strategy to use
            message: Merge commit message

        Returns:
            True if merge was successful
        """
        if not self.enable_git:
            return False

        functional_dir = self.storage_root / functional_id

        try:
            # Switch to target branch
            subprocess.run(
                ["git", "checkout", target_branch],
                cwd=functional_dir,
                check=True,
                capture_output=True,
            )

            # Perform merge based on strategy
            if strategy == MergeStrategy.FAST_FORWARD:
                subprocess.run(
                    ["git", "merge", "--ff-only", source_branch],
                    cwd=functional_dir,
                    check=True,
                    capture_output=True,
                )
            elif strategy == MergeStrategy.SQUASH:
                subprocess.run(
                    ["git", "merge", "--squash", source_branch],
                    cwd=functional_dir,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", message or f"Merge {source_branch}"],
                    cwd=functional_dir,
                    check=True,
                    capture_output=True,
                )
            else:  # MERGE_COMMIT
                subprocess.run(
                    [
                        "git",
                        "merge",
                        "--no-ff",
                        "-m",
                        message or f"Merge {source_branch}",
                        source_branch,
                    ],
                    cwd=functional_dir,
                    check=True,
                    capture_output=True,
                )

            return True

        except subprocess.CalledProcessError:
            # Merge conflict or other error
            return False

    async def get_version_diff(
        self, functional_id: str, version_a: str, version_b: str
    ) -> VersionDiff:
        """Get differences between two versions.

        Args:
            functional_id: ID of the functional
            version_a: First version
            version_b: Second version

        Returns:
            Differences between versions
        """
        diff = VersionDiff(version_a=version_a, version_b=version_b)

        if not self.enable_git:
            return diff

        functional_dir = self.storage_root / functional_id

        try:
            # Get commit hashes
            commit_a = await self._get_commit_for_tag(functional_id, version_a)
            commit_b = await self._get_commit_for_tag(functional_id, version_b)

            # Get file differences
            result = subprocess.run(
                ["git", "diff", "--name-status", commit_a, commit_b],
                cwd=functional_dir,
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse diff output
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                status, file_path = line.split("\t", 1)

                if status == "A":
                    diff.added_files.append(file_path)
                elif status == "M":
                    diff.modified_files.append(file_path)
                elif status == "D":
                    diff.removed_files.append(file_path)

        except subprocess.CalledProcessError:
            pass  # Return empty diff on error

        return diff

    async def _ensure_branch(
        self, functional_id: str, branch: str, author_id: str
    ) -> None:
        """Ensure branch exists and switch to it."""
        if not self.enable_git:
            return

        functional_dir = self.storage_root / functional_id

        # Check if branch exists
        result = subprocess.run(
            ["git", "branch", "--list", branch],
            check=False,
            cwd=functional_dir,
            capture_output=True,
            text=True,
        )

        if not result.stdout.strip():
            # Branch doesn't exist, create it
            subprocess.run(
                ["git", "checkout", "-b", branch],
                cwd=functional_dir,
                check=True,
                capture_output=True,
            )
        else:
            # Branch exists, switch to it
            subprocess.run(
                ["git", "checkout", branch],
                cwd=functional_dir,
                check=True,
                capture_output=True,
            )

    async def _commit_changes(
        self, functional_id: str, version_tag: str, message: str, author_id: str
    ) -> str:
        """Commit changes to Git and return commit hash."""
        functional_dir = self.storage_root / functional_id

        # Add all files
        subprocess.run(
            ["git", "add", "."],
            cwd=functional_dir,
            check=True,
            capture_output=True,
        )

        # Commit changes
        subprocess.run(
            ["git", "commit", "-m", f"{version_tag}: {message}"],
            cwd=functional_dir,
            check=True,
            capture_output=True,
        )

        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=functional_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    async def _create_git_tag(
        self, functional_id: str, version_tag: str, commit_hash: str
    ) -> None:
        """Create Git tag for version."""
        functional_dir = self.storage_root / functional_id

        subprocess.run(
            [
                "git",
                "tag",
                "-a",
                version_tag,
                "-m",
                f"Version {version_tag}",
                commit_hash,
            ],
            cwd=functional_dir,
            check=True,
            capture_output=True,
        )

    async def _delete_git_tag(self, functional_id: str, version_tag: str) -> None:
        """Delete Git tag."""
        functional_dir = self.storage_root / functional_id

        subprocess.run(
            ["git", "tag", "-d", version_tag],
            cwd=functional_dir,
            check=False,  # Don't fail if tag doesn't exist
            capture_output=True,
        )

    async def _get_commit_for_tag(self, functional_id: str, version_tag: str) -> str:
        """Get commit hash for a version tag."""
        functional_dir = self.storage_root / functional_id

        result = subprocess.run(
            ["git", "rev-list", "-n", "1", version_tag],
            check=False,
            cwd=functional_dir,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip() if result.returncode == 0 else ""

    async def _get_latest_commit(self, functional_id: str) -> str:
        """Get latest commit hash."""
        functional_dir = self.storage_root / functional_id

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            cwd=functional_dir,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip() if result.returncode == 0 else ""

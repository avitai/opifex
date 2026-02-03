"""Test suite for Neural Functional Version Management.

Provides comprehensive test coverage for version control functionality including
Git operations, branching, merging, and version tracking.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from opifex.platform.registry.version import (
    Branch,
    MergeStrategy,
    Version,
    VersionDiff,
    VersionManager,
    VersionStatus,
)


class MockRegistryService:
    """Mock registry service for testing."""

    def __init__(self):
        self.functionals = {}

    async def retrieve_functional(self, functional_id: str, version: str | None = None):
        """Mock retrieve functional method."""
        key = f"{functional_id}-{version}" if version else functional_id
        return self.functionals.get(key)


@pytest.fixture
def mock_registry():
    """Create mock registry service."""
    return MockRegistryService()


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def version_manager_no_git(mock_registry, temp_storage):
    """Create version manager without Git."""
    return VersionManager(
        storage_root=temp_storage,
        registry_service=mock_registry,
        enable_git=False,
    )


@pytest.fixture
def version_manager_with_git(mock_registry, temp_storage):
    """Create version manager with Git (mocked)."""
    with patch("subprocess.run"):
        manager = VersionManager(
            storage_root=temp_storage,
            registry_service=mock_registry,
            enable_git=True,
        )
        return manager


class TestVersionManager:
    """Test VersionManager functionality."""

    def test_initialization_no_git(self, mock_registry, temp_storage):
        """Test version manager initialization without Git."""
        manager = VersionManager(
            storage_root=temp_storage,
            registry_service=mock_registry,
            enable_git=False,
        )

        assert manager.storage_root == temp_storage
        assert manager.registry == mock_registry
        assert manager.enable_git is False
        assert temp_storage.exists()

    @patch("subprocess.run")
    def test_initialization_with_git(
        self, mock_subprocess, mock_registry, temp_storage
    ):
        """Test version manager initialization with Git."""
        manager = VersionManager(
            storage_root=temp_storage,
            registry_service=mock_registry,
            enable_git=True,
        )

        assert manager.enable_git is True
        # Should attempt to initialize Git repo
        assert mock_subprocess.call_count >= 1

    @patch("subprocess.run")
    def test_git_repo_initialization(
        self, mock_subprocess, mock_registry, temp_storage
    ):
        """Test Git repository initialization."""
        # Simulate .git directory doesn't exist
        mock_subprocess.return_value.returncode = 0

        _ = VersionManager(
            storage_root=temp_storage,
            registry_service=mock_registry,
            enable_git=True,
        )

        # Should call git init, config commands
        git_calls = [
            call for call in mock_subprocess.call_args_list if call[0][0][0] == "git"
        ]
        assert len(git_calls) >= 1

    @pytest.mark.asyncio
    async def test_create_version_basic(self, version_manager_no_git):
        """Test basic version creation without Git."""
        files = {
            "model.py": b"# Neural functional code",
            "metadata.json": b'{"type": "l2o"}',
        }

        version = await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.0.0",
            author_id="user-001",
            message="Initial version",
            files=files,
            metadata={"accuracy": 0.95},
        )

        assert isinstance(version, Version)
        assert version.functional_id == "func-001"
        assert version.version_tag == "v1.0.0"
        assert version.author_id == "user-001"
        assert version.message == "Initial version"
        assert version.status == VersionStatus.DRAFT
        assert version.metadata == {"accuracy": 0.95}
        assert version.size_bytes > 0

    @pytest.mark.asyncio
    async def test_create_version_with_files(
        self, version_manager_no_git, temp_storage
    ):
        """Test version creation with file storage."""
        files = {
            "model.py": "# Neural functional implementation\nprint('Hello World')",
            "config.yaml": "learning_rate: 0.001\nbatch_size: 32",
            "subdir/helper.py": "def helper_function():\n    pass",
        }

        _ = await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.0.0",
            author_id="user-001",
            message="Initial version",
            files=files,
        )

        # Check files were written
        functional_dir = temp_storage / "func-001"
        assert functional_dir.exists()
        assert (functional_dir / "model.py").exists()
        assert (functional_dir / "config.yaml").exists()
        assert (functional_dir / "subdir" / "helper.py").exists()

        # Check metadata file
        metadata_file = functional_dir / ".version-v1.0.0.json"
        assert metadata_file.exists()

        metadata = json.loads(metadata_file.read_text())
        assert metadata["functional_id"] == "func-001"
        assert metadata["version_tag"] == "v1.0.0"
        assert len(metadata["files"]) == 3

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_create_version_with_git(
        self, mock_subprocess, version_manager_with_git
    ):
        """Test version creation with Git operations."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "abc123"

        files = {"model.py": b"# Neural functional code"}

        version = await version_manager_with_git.create_version(
            functional_id="func-001",
            version_tag="v1.0.0",
            author_id="user-001",
            message="Initial version",
            files=files,
        )

        assert version.commit_hash == "abc123"

        # Should have called Git commands
        git_calls = [
            call for call in mock_subprocess.call_args_list if "git" in call[0][0]
        ]
        assert len(git_calls) > 0

    @pytest.mark.asyncio
    async def test_get_version_existing(self, version_manager_no_git):
        """Test retrieving existing version."""
        # Create version first
        files = {"model.py": b"# Code"}
        await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.0.0",
            author_id="user-001",
            message="Test version",
            files=files,
        )

        # Retrieve version
        version = await version_manager_no_git.get_version("func-001", "v1.0.0")

        assert version is not None
        assert version.functional_id == "func-001"
        assert version.version_tag == "v1.0.0"

    @pytest.mark.asyncio
    async def test_get_version_nonexistent(self, version_manager_no_git):
        """Test retrieving nonexistent version."""
        version = await version_manager_no_git.get_version("nonexistent", "v1.0.0")
        assert version is None

    @pytest.mark.asyncio
    async def test_list_versions(self, version_manager_no_git):
        """Test listing versions of a functional."""
        # Create multiple versions
        files = {"model.py": b"# Code"}

        await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.0.0",
            author_id="user-001",
            message="Version 1",
            files=files,
        )

        await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.1.0",
            author_id="user-001",
            message="Version 2",
            files=files,
        )

        versions = await version_manager_no_git.list_versions("func-001")

        assert len(versions) == 2
        assert all(isinstance(v, Version) for v in versions)
        assert {v.version_tag for v in versions} == {"v1.0.0", "v1.1.0"}

    @pytest.mark.asyncio
    async def test_list_versions_by_branch(self, version_manager_no_git):
        """Test listing versions filtered by branch."""
        files = {"model.py": b"# Code"}

        # Create versions on different branches
        await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.0.0",
            author_id="user-001",
            message="Main version",
            files=files,
            branch="main",
        )

        await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.0.1-dev",
            author_id="user-001",
            message="Dev version",
            files=files,
            branch="development",
        )

        main_versions = await version_manager_no_git.list_versions(
            "func-001", branch="main"
        )
        dev_versions = await version_manager_no_git.list_versions(
            "func-001", branch="development"
        )

        assert len(main_versions) == 1
        assert len(dev_versions) == 1
        assert main_versions[0].branch == "main"
        assert dev_versions[0].branch == "development"

    @pytest.mark.asyncio
    async def test_list_versions_empty(self, version_manager_no_git):
        """Test listing versions for nonexistent functional."""
        versions = await version_manager_no_git.list_versions("nonexistent")
        assert len(versions) == 0

    @pytest.mark.asyncio
    async def test_delete_version_by_author(self, version_manager_no_git):
        """Test version deletion by author."""
        files = {"model.py": b"# Code"}

        # Create version
        await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.0.0",
            author_id="user-001",
            message="Test version",
            files=files,
        )

        # Delete by author
        result = await version_manager_no_git.delete_version(
            "func-001", "v1.0.0", "user-001"
        )
        assert result is True

        # Verify deletion
        version = await version_manager_no_git.get_version("func-001", "v1.0.0")
        assert version is None

    @pytest.mark.asyncio
    async def test_delete_version_unauthorized(self, version_manager_no_git):
        """Test version deletion by unauthorized user."""
        files = {"model.py": b"# Code"}

        # Create version
        await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.0.0",
            author_id="user-001",
            message="Test version",
            files=files,
        )

        # Try to delete by different user
        result = await version_manager_no_git.delete_version(
            "func-001", "v1.0.0", "user-002"
        )
        assert result is False

        # Verify version still exists
        version = await version_manager_no_git.get_version("func-001", "v1.0.0")
        assert version is not None

    @pytest.mark.asyncio
    async def test_delete_version_nonexistent(self, version_manager_no_git):
        """Test deleting nonexistent version."""
        result = await version_manager_no_git.delete_version(
            "func-001", "v1.0.0", "user-001"
        )
        assert result is False

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_create_branch(self, mock_subprocess, version_manager_with_git):
        """Test branch creation."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "abc123"

        branch = await version_manager_with_git.create_branch(
            functional_id="func-001",
            branch_name="feature-branch",
            author_id="user-001",
            description="Feature development",
        )

        assert isinstance(branch, Branch)
        assert branch.name == "feature-branch"
        assert branch.functional_id == "func-001"
        assert branch.author_id == "user-001"
        assert branch.description == "Feature development"

    @pytest.mark.asyncio
    async def test_create_branch_no_git(self, version_manager_no_git):
        """Test branch creation without Git should raise error."""
        with pytest.raises(ValueError, match="Git must be enabled"):
            await version_manager_no_git.create_branch(
                functional_id="func-001",
                branch_name="feature-branch",
                author_id="user-001",
            )

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_merge_branch_success(
        self, mock_subprocess, version_manager_with_git
    ):
        """Test successful branch merge."""
        mock_subprocess.return_value.returncode = 0

        result = await version_manager_with_git.merge_branch(
            functional_id="func-001",
            source_branch="feature-branch",
            target_branch="main",
            author_id="user-001",
            strategy=MergeStrategy.MERGE_COMMIT,
            message="Merge feature branch",
        )

        assert result is True

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_merge_branch_conflict(
        self, mock_subprocess, version_manager_with_git
    ):
        """Test branch merge with conflict."""
        import subprocess

        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # checkout success
            subprocess.CalledProcessError(1, "git merge"),  # merge conflict
        ]

        result = await version_manager_with_git.merge_branch(
            functional_id="func-001",
            source_branch="feature-branch",
            target_branch="main",
            author_id="user-001",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_merge_branch_no_git(self, version_manager_no_git):
        """Test branch merge without Git."""
        result = await version_manager_no_git.merge_branch(
            functional_id="func-001",
            source_branch="feature-branch",
            target_branch="main",
            author_id="user-001",
        )

        assert result is False

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_merge_strategies(self, mock_subprocess, version_manager_with_git):
        """Test different merge strategies."""
        mock_subprocess.return_value.returncode = 0

        # Test fast-forward merge
        await version_manager_with_git.merge_branch(
            functional_id="func-001",
            source_branch="feature-branch",
            target_branch="main",
            author_id="user-001",
            strategy=MergeStrategy.FAST_FORWARD,
        )

        # Test squash merge
        await version_manager_with_git.merge_branch(
            functional_id="func-001",
            source_branch="feature-branch",
            target_branch="main",
            author_id="user-001",
            strategy=MergeStrategy.SQUASH,
            message="Squash merge",
        )

        # Verify appropriate git commands were called
        git_calls = [
            call for call in mock_subprocess.call_args_list if "git" in call[0][0]
        ]
        assert len(git_calls) > 0

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_get_version_diff(self, mock_subprocess, version_manager_with_git):
        """Test getting version differences."""
        # Mock git diff output
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = (
            "A\tmodel.py\nM\tconfig.yaml\nD\told_file.py\n"
        )

        diff = await version_manager_with_git.get_version_diff(
            functional_id="func-001",
            version_a="v1.0.0",
            version_b="v1.1.0",
        )

        assert isinstance(diff, VersionDiff)
        assert diff.version_a == "v1.0.0"
        assert diff.version_b == "v1.1.0"
        assert "model.py" in diff.added_files
        assert "config.yaml" in diff.modified_files
        assert "old_file.py" in diff.removed_files

    @pytest.mark.asyncio
    async def test_get_version_diff_no_git(self, version_manager_no_git):
        """Test getting version diff without Git."""
        diff = await version_manager_no_git.get_version_diff(
            functional_id="func-001",
            version_a="v1.0.0",
            version_b="v1.1.0",
        )

        assert isinstance(diff, VersionDiff)
        assert len(diff.added_files) == 0
        assert len(diff.modified_files) == 0
        assert len(diff.removed_files) == 0

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_ensure_branch_existing(
        self, mock_subprocess, version_manager_with_git
    ):
        """Test ensuring existing branch."""
        # Mock branch exists
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "  feature-branch\n"

        await version_manager_with_git._ensure_branch(
            "func-001", "feature-branch", "user-001"
        )

        # Should checkout existing branch
        checkout_calls = [
            call for call in mock_subprocess.call_args_list if "checkout" in call[0][0]
        ]
        assert len(checkout_calls) > 0

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_ensure_branch_new(self, mock_subprocess, version_manager_with_git):
        """Test ensuring new branch."""
        # Mock branch doesn't exist
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = ""

        await version_manager_with_git._ensure_branch(
            "func-001", "new-branch", "user-001"
        )

        # Should create new branch
        checkout_calls = [
            call
            for call in mock_subprocess.call_args_list
            if "checkout" in call[0][0] and "-b" in call[0][0]
        ]
        assert len(checkout_calls) > 0

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_commit_changes(self, mock_subprocess, version_manager_with_git):
        """Test committing changes to Git."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "abc123def456"

        commit_hash = await version_manager_with_git._commit_changes(
            functional_id="func-001",
            version_tag="v1.0.0",
            message="Test commit",
            author_id="user-001",
        )

        assert commit_hash == "abc123def456"

        # Should call git add and git commit
        add_calls = [
            call for call in mock_subprocess.call_args_list if "add" in call[0][0]
        ]
        commit_calls = [
            call for call in mock_subprocess.call_args_list if "commit" in call[0][0]
        ]
        assert len(add_calls) > 0
        assert len(commit_calls) > 0

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_create_git_tag(self, mock_subprocess, version_manager_with_git):
        """Test creating Git tag."""
        mock_subprocess.return_value.returncode = 0

        await version_manager_with_git._create_git_tag(
            functional_id="func-001",
            version_tag="v1.0.0",
            commit_hash="abc123",
        )

        # Should call git tag
        tag_calls = [
            call for call in mock_subprocess.call_args_list if "tag" in call[0][0]
        ]
        assert len(tag_calls) > 0

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_delete_git_tag(self, mock_subprocess, version_manager_with_git):
        """Test deleting Git tag."""
        mock_subprocess.return_value.returncode = 0

        await version_manager_with_git._delete_git_tag("func-001", "v1.0.0")

        # Should call git tag -d
        delete_calls = [
            call
            for call in mock_subprocess.call_args_list
            if "tag" in call[0][0] and "-d" in call[0][0]
        ]
        assert len(delete_calls) > 0

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_get_commit_for_tag(self, mock_subprocess, version_manager_with_git):
        """Test getting commit hash for tag."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "abc123def456\n"

        commit_hash = await version_manager_with_git._get_commit_for_tag(
            "func-001", "v1.0.0"
        )

        assert commit_hash == "abc123def456"

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_get_latest_commit(self, mock_subprocess, version_manager_with_git):
        """Test getting latest commit hash."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "latest123\n"

        commit_hash = await version_manager_with_git._get_latest_commit("func-001")

        assert commit_hash == "latest123"


class TestVersionDataClasses:
    """Test version-related data classes."""

    def test_version_creation(self):
        """Test Version data class creation."""
        version = Version(
            id="func-001-v1.0.0",
            functional_id="func-001",
            version_tag="v1.0.0",
            commit_hash="abc123",
            author_id="user-001",
            message="Initial version",
        )

        assert version.id == "func-001-v1.0.0"
        assert version.functional_id == "func-001"
        assert version.version_tag == "v1.0.0"
        assert version.commit_hash == "abc123"
        assert version.status == VersionStatus.DRAFT  # Default
        assert version.branch == "main"  # Default

    def test_version_with_metadata(self):
        """Test Version with metadata and dependencies."""
        version = Version(
            id="func-001-v1.0.0",
            functional_id="func-001",
            version_tag="v1.0.0",
            commit_hash="abc123",
            status=VersionStatus.PUBLISHED,
            changes=["model.py", "config.yaml"],
            dependencies={"jax": "0.4.0", "flax": "0.7.0"},
            metadata={"accuracy": 0.95, "memory_gb": 2},
        )

        assert version.status == VersionStatus.PUBLISHED
        assert len(version.changes) == 2
        assert "jax" in version.dependencies
        assert version.metadata["accuracy"] == 0.95

    def test_branch_creation(self):
        """Test Branch data class creation."""
        branch = Branch(
            name="feature-branch",
            functional_id="func-001",
            head_commit="abc123",
            author_id="user-001",
            description="Feature development",
        )

        assert branch.name == "feature-branch"
        assert branch.functional_id == "func-001"
        assert branch.head_commit == "abc123"
        assert branch.is_protected is False  # Default

    def test_version_diff_creation(self):
        """Test VersionDiff data class creation."""
        diff = VersionDiff(
            version_a="v1.0.0",
            version_b="v1.1.0",
            added_files=["new_file.py"],
            modified_files=["model.py"],
            removed_files=["old_file.py"],
        )

        assert diff.version_a == "v1.0.0"
        assert diff.version_b == "v1.1.0"
        assert len(diff.added_files) == 1
        assert len(diff.modified_files) == 1
        assert len(diff.removed_files) == 1

    def test_version_status_enum(self):
        """Test VersionStatus enum values."""
        assert VersionStatus.DRAFT.value == "draft"
        assert VersionStatus.PUBLISHED.value == "published"
        assert VersionStatus.DEPRECATED.value == "deprecated"
        assert VersionStatus.ARCHIVED.value == "archived"

    def test_merge_strategy_enum(self):
        """Test MergeStrategy enum values."""
        assert MergeStrategy.FAST_FORWARD.value == "fast_forward"
        assert MergeStrategy.MERGE_COMMIT.value == "merge_commit"
        assert MergeStrategy.SQUASH.value == "squash"
        assert MergeStrategy.REBASE.value == "rebase"


class TestVersionManagerIntegration:
    """Integration tests for version management workflows."""

    @pytest.mark.asyncio
    async def test_complete_version_workflow(self, version_manager_no_git):
        """Test complete version management workflow."""
        # Create initial version
        files_v1 = {"model.py": b"# Version 1", "config.yaml": b"version: 1"}
        _ = await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.0.0",
            author_id="user-001",
            message="Initial version",
            files=files_v1,
        )

        # Create updated version
        files_v2 = {"model.py": b"# Version 2", "config.yaml": b"version: 2"}
        _ = await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.1.0",
            author_id="user-001",
            message="Updated version",
            files=files_v2,
            parent_version="v1.0.0",
        )

        # List all versions
        versions = await version_manager_no_git.list_versions("func-001")
        assert len(versions) == 2

        # Get specific version
        retrieved_v1 = await version_manager_no_git.get_version("func-001", "v1.0.0")
        assert retrieved_v1.version_tag == "v1.0.0"

        # Delete old version
        deleted = await version_manager_no_git.delete_version(
            "func-001", "v1.0.0", "user-001"
        )
        assert deleted is True

        # Verify deletion
        versions_after_delete = await version_manager_no_git.list_versions("func-001")
        assert len(versions_after_delete) == 1
        assert versions_after_delete[0].version_tag == "v1.1.0"

    @pytest.mark.asyncio
    async def test_version_lineage(self, version_manager_no_git):
        """Test version lineage tracking."""
        files = {"model.py": b"# Code"}

        # Create version chain
        v1 = await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.0.0",
            author_id="user-001",
            message="Initial",
            files=files,
        )

        v2 = await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.1.0",
            author_id="user-001",
            message="Update",
            files=files,
            parent_version="v1.0.0",
        )

        v3 = await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.2.0",
            author_id="user-001",
            message="Another update",
            files=files,
            parent_version="v1.1.0",
        )

        # Verify lineage
        assert v1.parent_version is None
        assert v2.parent_version == "v1.0.0"
        assert v3.parent_version == "v1.1.0"

    @pytest.mark.asyncio
    async def test_multi_author_versions(self, version_manager_no_git):
        """Test versions from multiple authors."""
        files = {"model.py": b"# Code"}

        # Create versions by different authors
        await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.0.0",
            author_id="user-001",
            message="User 1 version",
            files=files,
        )

        await version_manager_no_git.create_version(
            functional_id="func-001",
            version_tag="v1.1.0",
            author_id="user-002",
            message="User 2 version",
            files=files,
        )

        # User 1 cannot delete User 2's version
        deleted = await version_manager_no_git.delete_version(
            "func-001", "v1.1.0", "user-001"
        )
        assert deleted is False

        # User 2 can delete their own version
        deleted = await version_manager_no_git.delete_version(
            "func-001", "v1.1.0", "user-002"
        )
        assert deleted is True

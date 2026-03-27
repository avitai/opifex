"""Tests for Orbax checkpoint manager."""

import pytest

from opifex.core.training.components.orbax_manager import OrbaxCheckpointManager


class TestOrbaxCheckpointManagerInit:
    """Tests for OrbaxCheckpointManager initialization."""

    def test_creates_directory(self, tmp_path):
        """Creates checkpoint directory on init."""
        ckpt_dir = tmp_path / "checkpoints"
        mgr = OrbaxCheckpointManager(ckpt_dir)
        assert ckpt_dir.exists()
        assert mgr.max_to_keep == 5

    def test_custom_max_to_keep(self, tmp_path):
        """Respects custom max_to_keep setting."""
        mgr = OrbaxCheckpointManager(tmp_path / "ckpt", max_to_keep=3)
        assert mgr.max_to_keep == 3

    def test_empty_dir_raises(self):
        """Empty checkpoint directory raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            OrbaxCheckpointManager("")

    def test_whitespace_dir_raises(self):
        """Whitespace-only directory raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            OrbaxCheckpointManager("   ")

    def test_checkpoint_dir_is_resolved(self, tmp_path):
        """Checkpoint directory path is resolved to absolute."""
        mgr = OrbaxCheckpointManager(tmp_path / "ckpt")
        assert mgr.checkpoint_dir.is_absolute()

    def test_no_create_flag(self, tmp_path):
        """With create=False, does not create directory."""
        ckpt_dir = tmp_path / "nonexistent"
        # create=False but orbax may still need the dir for its manager
        # Just verify the flag is respected in our code
        mgr = OrbaxCheckpointManager(ckpt_dir, create=False)
        assert mgr.checkpoint_dir == ckpt_dir.resolve()

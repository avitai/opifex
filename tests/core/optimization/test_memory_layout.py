"""Tests for memory layout optimization."""

from opifex.core.optimization.memory_layout import (
    LayoutOptimizer,
    MemoryLayout,
)


class TestMemoryLayout:
    """Tests for MemoryLayout enum."""

    def test_nchw_value(self):
        """NCHW has correct string value."""
        assert MemoryLayout.NCHW.value == "NCHW"

    def test_nhwc_value(self):
        """NHWC has correct string value."""
        assert MemoryLayout.NHWC.value == "NHWC"

    def test_3d_layouts_exist(self):
        """3D layout variants exist."""
        assert MemoryLayout.NDHWC.value == "NDHWC"
        assert MemoryLayout.NCDHW.value == "NCDHW"


class TestLayoutOptimizer:
    """Tests for LayoutOptimizer."""

    def test_detects_backend(self):
        """Optimizer detects the current JAX backend."""
        opt = LayoutOptimizer()
        assert opt.backend in ("cpu", "gpu", "tpu")

    def test_returns_optimal_layouts(self):
        """Returns layout recommendations for standard operations."""
        opt = LayoutOptimizer()
        layouts = opt.optimal_layouts
        assert "convolution" in layouts
        assert "matmul" in layouts
        assert "default" in layouts
        assert all(isinstance(v, MemoryLayout) for v in layouts.values())

    def test_cpu_prefers_nchw(self):
        """CPU backend defaults include NCHW for some ops."""
        opt = LayoutOptimizer()
        if opt.backend == "cpu":
            assert opt.optimal_layouts["default"] == MemoryLayout.NCHW

    def test_conversion_cache_starts_empty(self):
        """Conversion cache is empty on init."""
        opt = LayoutOptimizer()
        assert opt.conversion_cache == {}

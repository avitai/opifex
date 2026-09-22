"""Tests for the profiling harness optimization report."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from calibrax.profiling import HardwareSpec
from flax import nnx

from opifex.benchmarking.profiling import profiling_harness
from opifex.benchmarking.profiling.profiling_harness import (
    OpifexProfilingHarness,
    OptimizationReport,
)


class TestOptimizationReport:
    """Tests for OptimizationReport."""

    def test_empty_report(self):
        """New report has empty sections."""
        report = OptimizationReport()
        assert report.sections == {}
        assert report.executive_summary == {}
        assert report.priority_recommendations == []

    def test_add_section(self):
        """Sections can be added by title."""
        report = OptimizationReport()
        report.add_section("Memory", {"peak_mb": 512})
        assert "Memory" in report.sections
        assert report.sections["Memory"]["peak_mb"] == 512

    def test_set_executive_summary(self):
        """Executive summary can be set."""
        report = OptimizationReport()
        report.set_executive_summary({"status": "healthy", "score": 85})
        assert report.executive_summary["status"] == "healthy"

    def test_add_priority_recommendation(self):
        """Recommendations are appended with impact/effort."""
        report = OptimizationReport()
        report.add_priority_recommendation("Use mixed precision", impact="high", effort="low")
        assert len(report.priority_recommendations) == 1
        rec = report.priority_recommendations[0]
        assert rec["recommendation"] == "Use mixed precision"
        assert rec["impact"] == "high"
        assert rec["effort"] == "low"

    def test_recommendation_defaults(self):
        """Recommendation defaults to medium impact and effort."""
        report = OptimizationReport()
        report.add_priority_recommendation("Optimize layout")
        rec = report.priority_recommendations[0]
        assert rec["impact"] == "medium"
        assert rec["effort"] == "medium"

    def test_render_text(self):
        """Text rendering includes sections and recommendations."""
        report = OptimizationReport()
        report.set_executive_summary({"backend": "gpu"})
        report.add_priority_recommendation("Enable JIT")
        text = report.render("text")
        assert "Optimization Report" in text
        assert "EXECUTIVE SUMMARY" in text
        assert "PRIORITY RECOMMENDATIONS" in text
        assert "Enable JIT" in text

    def test_render_json(self):
        """JSON rendering produces valid JSON."""
        report = OptimizationReport()
        report.add_section("test", {"value": 42})
        output = report.render("json")
        data = json.loads(output)
        assert "sections" in data
        assert data["sections"]["test"]["value"] == 42

    def test_render_html(self):
        """HTML rendering produces HTML with sections."""
        report = OptimizationReport()
        report.add_section("Memory", {"peak": 256})
        html = report.render("html")
        assert "<h1>" in html or "<html>" in html.lower() or "Memory" in html

    def test_render_unsupported_format_raises(self):
        """Unsupported format raises ValueError."""
        report = OptimizationReport()
        with pytest.raises(ValueError, match="Unsupported format"):
            report.render("yaml")

    def test_multiple_sections(self):
        """Multiple sections are stored independently."""
        report = OptimizationReport()
        report.add_section("A", {"x": 1})
        report.add_section("B", {"y": 2})
        assert len(report.sections) == 2
        assert report.sections["A"]["x"] == 1
        assert report.sections["B"]["y"] == 2


_SPEC = HardwareSpec(name="test-device", peak_flops=1.0e12, memory_bandwidth=1.0e11)


def _summed_sine(x: jax.Array) -> jax.Array:
    return jnp.sum(jnp.sin(x) * x)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> OpifexProfilingHarness:
    monkeypatch.setattr(profiling_harness, "resolve_hardware_spec", lambda *, dtype: _SPEC)
    return OpifexProfilingHarness(trace_dir=str(tmp_path / "trace"))


class TestOpifexProfilingHarness:
    """The harness reports calibrax's roofline and compilation results and the hardware spec."""

    def test_hardware_section_names_the_spec_and_the_devices(
        self, harness: OpifexProfilingHarness
    ) -> None:
        results, report = harness.profile_function(_summed_sine, [jnp.ones((64,))])

        assert results["hardware_analysis"]["hardware_info"] == _SPEC.to_dict()
        section = report.sections["Hardware Analysis"]
        assert section["Hardware"] == "test-device"
        assert section["Device Count"] == jax.device_count()

    def test_roofline_section_reports_calibrax_timing_and_recommendations(
        self, harness: OpifexProfilingHarness
    ) -> None:
        results, report = harness.profile_function(_summed_sine, [jnp.ones((64,))])

        roofline = results["roofline_analysis"]
        section = report.sections["Roofline Analysis"]
        assert roofline["execution_time_ms"] > 0.0
        assert section["Execution Time"] == f"{roofline['execution_time_ms']:.2f} ms"
        assert section["Recommendations"] == roofline["recommendations"]

    def test_compilation_section_reports_calibrax_cache_statistics(
        self, harness: OpifexProfilingHarness
    ) -> None:
        results, report = harness.profile_function(_summed_sine, [jnp.ones((64,))])

        compilation = results["compilation_analysis"]
        section = report.sections["Compilation Analysis"]
        assert compilation["total_calls"] >= 1
        assert section["Total Calls"] == compilation["total_calls"]
        assert section["Cache Hit Rate"] == f"{compilation['cache_hit_rate']:.2%}"
        assert section["Average Compilation Time"] == (
            f"{compilation['avg_compilation_time_ms']:.1f} ms"
        )
        assert report.executive_summary["Cache Hit Rate"] == section["Cache Hit Rate"]

    def test_operation_name_defaults_to_the_module_class(
        self, harness: OpifexProfilingHarness
    ) -> None:
        model = nnx.Linear(4, 2, rngs=nnx.Rngs(0))

        results, _ = harness.profile_neural_operator(model, [jnp.ones((3, 4))])

        assert results["operation_name"] == "Linear"

    def test_comparison_picks_the_fastest_by_calibrax_execution_time(
        self, harness: OpifexProfilingHarness
    ) -> None:
        comparison = harness.compare_operations(
            [
                ("small", _summed_sine, [jnp.ones((16,))]),
                ("large", _summed_sine, [jnp.ones((1 << 20,))]),
            ]
        )

        times = {
            name: comparison["comparison_analysis"]["metrics_summary"][name]["execution_time_ms"]
            for name in ("small", "large")
        }
        assert all(value > 0.0 for value in times.values())
        fastest = min(times, key=lambda name: times[name])
        assert comparison["comparison_analysis"]["fastest_execution"] == fastest

"""Tests for the profiling harness optimization report."""

import json

import pytest

from opifex.benchmarking.profiling.profiling_harness import OptimizationReport


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

"""
Comprehensive JAX Profiling Harness for Opifex.

Main interface for the comprehensive profiling system that coordinates
hardware-aware profiling, roofline analysis, compilation profiling,
and generates actionable optimization reports.
"""

import json
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import jax
from flax import nnx

from .compilation_profiler import CompilationProfiler
from .event_coordinator import EventCoordinator
from .hardware_profiler import HardwareAwareProfiler
from .roofline_analyzer import RooflineAnalyzer


class OptimizationReport:
    """Structured optimization report with actionable recommendations."""

    def __init__(self):
        self.sections = {}
        self.executive_summary = {}
        self.priority_recommendations = []

    def add_section(self, title: str, content: Any):
        """Add a section to the report."""
        self.sections[title] = content

    def set_executive_summary(self, summary: dict[str, Any]):
        """Set the executive summary."""
        self.executive_summary = summary

    def add_priority_recommendation(
        self, recommendation: str, impact: str = "medium", effort: str = "medium"
    ):
        """Add a priority recommendation."""
        self.priority_recommendations.append(
            {"recommendation": recommendation, "impact": impact, "effort": effort}
        )

    def render(self, output_format: str = "text") -> str:
        """Render the report in specified format."""
        renderers = {
            "text": self._render_text,
            "json": self._render_json,
            "html": self._render_html,
        }
        renderer = renderers.get(output_format)
        if renderer:
            return renderer()
        raise ValueError(f"Unsupported format: {output_format}")

    def _render_text(self) -> str:
        """Render report as formatted text."""

        lines = []
        lines.append("=" * 80)
        lines.append("🚀 Opifex JAX Performance Optimization Report")
        lines.append("=" * 80)

        # Executive Summary
        if self.executive_summary:
            lines.append("\n📊 EXECUTIVE SUMMARY")
            lines.append("-" * 40)
            for key, value in self.executive_summary.items():
                lines.append(f"  {key}: {value}")

        # Priority Recommendations
        if self.priority_recommendations:
            lines.append("\n🎯 PRIORITY RECOMMENDATIONS")
            lines.append("-" * 40)
            for i, rec in enumerate(self.priority_recommendations[:5], 1):
                lines.append(f"  {i}. {rec['recommendation']}")
                lines.append(f"     Impact: {rec['impact']}, Effort: {rec['effort']}")

        # Detailed Sections
        for title, content in self.sections.items():
            lines.append(f"\n📋 {title.upper()}")
            lines.append("-" * 40)
            lines.extend(self._render_section_text(content))

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def _render_section_text(self, content: Any) -> list[str]:
        """Render a single section's content as text lines."""
        lines = []
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, list):
                    lines.append(f"  {key}:")
                    for item in value:
                        lines.append(f"    • {item}")
                else:
                    lines.append(f"  {key}: {value}")
        elif isinstance(content, list):
            for item in content:
                lines.append(f"  • {item}")
        else:
            lines.append(f"  {content}")
        return lines

    def _render_json(self) -> str:
        """Render report as JSON."""

        report_data = {
            "executive_summary": self.executive_summary,
            "priority_recommendations": self.priority_recommendations,
            "sections": self.sections,
        }

        return json.dumps(report_data, indent=2, default=str)

    def _render_html(self) -> str:
        """Render report as HTML."""

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Opifex JAX Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f8ff; padding: 20px; border-radius: 5px; }
                .section {
                    margin: 20px 0; padding: 15px; border-left: 4px solid #007acc;
                }
                .recommendation {
                    background: #fff3cd; padding: 10px; margin: 5px 0;
                    border-radius: 3px;
                }
                .metric {
                    display: inline-block; margin: 10px; padding: 10px;
                    background: #e9ecef; border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Opifex JAX Performance Optimization Report</h1>
            </div>
        """

        # Executive Summary
        if self.executive_summary:
            html += '<div class="section"><h2>📊 Executive Summary</h2>'
            for key, value in self.executive_summary.items():
                html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
            html += "</div>"

        # Priority Recommendations
        if self.priority_recommendations:
            html += '<div class="section"><h2>🎯 Priority Recommendations</h2>'
            for i, rec in enumerate(self.priority_recommendations, 1):
                html += f'<div class="recommendation">{i}. {rec["recommendation"]}<br>'
                html += (
                    f"<small>Impact: {rec['impact']}, Effort: {rec['effort']}</small>"
                    "</div>"
                )
            html += "</div>"

        # Sections
        for title, content in self.sections.items():
            html += f'<div class="section"><h2>📋 {title}</h2>'
            if isinstance(content, dict):
                for key, value in content.items():
                    html += f"<p><strong>{key}:</strong> {value}</p>"
            elif isinstance(content, list):
                html += "<ul>"
                for item in content:
                    html += f"<li>{item}</li>"
                html += "</ul>"
            else:
                html += f"<p>{content}</p>"
            html += "</div>"

        html += "</body></html>"
        return html


class OpifexProfilingHarness:
    """Comprehensive JAX profiling harness for Opifex applications."""

    def __init__(
        self,
        enable_hardware_profiling: bool = True,
        enable_compilation_profiling: bool = True,
        enable_roofline_analysis: bool = True,
        trace_dir: str | None = None,
    ):
        # Initialize event coordinator
        self.coordinator = EventCoordinator()
        if trace_dir is None:
            from pathlib import Path

            trace_dir = str(Path(tempfile.gettempdir()) / "opifex_jax_trace")
        self.trace_dir = trace_dir

        # Initialize profilers based on configuration
        self.profilers = {}

        if enable_hardware_profiling:
            self.hardware_profiler = HardwareAwareProfiler(self.coordinator)
            self.profilers["hardware"] = self.hardware_profiler

        if enable_roofline_analysis:
            self.roofline_analyzer = RooflineAnalyzer(self.coordinator)
            self.profilers["roofline"] = self.roofline_analyzer

        if enable_compilation_profiling:
            self.compilation_profiler = CompilationProfiler(self.coordinator)
            self.profilers["compilation"] = self.compilation_profiler

        # Performance tracking
        self.profiling_sessions = []

    @contextmanager
    def profiling_session(self, enable_jax_profiler: bool = True):
        """Context manager for comprehensive profiling session."""

        session_start = time.time()

        with self.coordinator.profiling_session(enable_jax_profiler, self.trace_dir):
            session_data = {
                "start_time": session_start,
                "profilers_active": list(self.profilers.keys()),
                "jax_profiler_enabled": enable_jax_profiler,
            }

            try:
                yield self
                session_data["success"] = True
            except Exception as e:
                session_data["success"] = False
                session_data["error"] = str(e)
                raise
            finally:
                session_data["end_time"] = time.time()
                session_data["duration"] = (
                    session_data["end_time"] - session_data["start_time"]
                )
                self.profiling_sessions.append(session_data)

    def profile_neural_operator(  # noqa: PLR0912
        self,
        operator: nnx.Module | Callable,
        inputs: list[jax.Array],
        operation_name: str | None = None,
    ) -> tuple[dict[str, Any], OptimizationReport]:
        """Profile a complete neural operator with comprehensive analysis."""

        if operation_name is None:
            operation_name = getattr(operator, "__class__", {}).get(
                "__name__", "neural_operator"
            )
        results: dict[str, Any] = {"operation_name": operation_name}

        # Hardware-specific analysis
        if "hardware" in self.profilers:
            try:
                if callable(operator):
                    # NNX Module
                    def operator_func(*args):
                        return operator(*args)
                else:
                    # Regular function
                    operator_func = operator  # pyright: ignore[reportAssignmentType]

                results["hardware_analysis"] = self.hardware_profiler.profile_operation(
                    operator_func, *inputs
                )
            except Exception as e:
                results["hardware_analysis"] = {"error": str(e)}

        # Roofline analysis
        if "roofline" in self.profilers:
            try:
                if callable(operator):

                    def operator_func(*args):
                        return operator(*args)
                else:
                    operator_func = operator  # pyright: ignore[reportAssignmentType]

                results["roofline_analysis"] = self.roofline_analyzer.analyze_operation(
                    operator_func, inputs, str(operation_name)
                )
            except Exception as e:
                results["roofline_analysis"] = {"error": str(e)}

        # Compilation analysis
        if "compilation" in self.profilers:
            try:
                if callable(operator):

                    def operator_func(*args):
                        return operator(*args)
                else:
                    operator_func = operator  # pyright: ignore[reportAssignmentType]

                # Profile JIT compilation
                instrumented_func = self.compilation_profiler.profile_jit_compilation(
                    operator_func
                )

                # Run a few times to collect compilation stats
                for _ in range(3):
                    _ = instrumented_func(*inputs)

                results["compilation_analysis"] = (
                    self.compilation_profiler.analyze_compilation_efficiency()
                )

                # XLA optimization analysis
                results["xla_analysis"] = (
                    self.compilation_profiler.estimate_xla_optimization_effectiveness(
                        operator_func, tuple(inputs)
                    )
                )

            except Exception as e:
                results["compilation_analysis"] = {"error": str(e)}
                results["xla_analysis"] = {"error": str(e)}

        # Generate comprehensive report
        report = self._generate_comprehensive_report(results)

        return results, report

    def profile_function(
        self, func: Callable, inputs: list[jax.Array], function_name: str | None = None
    ) -> tuple[dict[str, Any], OptimizationReport]:
        """Profile a JAX function with comprehensive analysis."""

        if function_name is None:
            function_name = getattr(func, "__name__", "jax_function")

        return self.profile_neural_operator(func, inputs, function_name)

    def compare_operations(
        self, operations: list[tuple[str, nnx.Module | Callable, list[jax.Array]]]
    ) -> dict[str, Any]:
        """Compare multiple operations and identify optimization opportunities."""

        comparison_results = {}

        for name, operation, inputs in operations:
            try:
                results, _ = self.profile_neural_operator(operation, inputs, name)
                comparison_results[name] = results
            except Exception as e:
                comparison_results[name] = {"error": str(e)}

        # Generate comparison analysis
        comparison_analysis = self._analyze_operation_comparison(comparison_results)

        return {
            "individual_results": comparison_results,
            "comparison_analysis": comparison_analysis,
            "recommendations": self._generate_comparison_recommendations(
                comparison_analysis
            ),
        }

    def _generate_comprehensive_report(
        self, results: dict[str, Any]
    ) -> OptimizationReport:
        """Generate a comprehensive optimization report."""

        report = OptimizationReport()

        # Executive Summary
        executive_summary = self._create_executive_summary(results)
        report.set_executive_summary(executive_summary)

        # Add detailed sections
        if (
            "hardware_analysis" in results
            and "error" not in results["hardware_analysis"]
        ):
            report.add_section(
                "Hardware Analysis",
                self._format_hardware_analysis(results["hardware_analysis"]),
            )

        if (
            "roofline_analysis" in results
            and "error" not in results["roofline_analysis"]
        ):
            report.add_section(
                "Roofline Analysis",
                self._format_roofline_analysis(results["roofline_analysis"]),
            )

        if (
            "compilation_analysis" in results
            and "error" not in results["compilation_analysis"]
        ):
            report.add_section(
                "Compilation Analysis",
                self._format_compilation_analysis(results["compilation_analysis"]),
            )

        if "xla_analysis" in results and "error" not in results["xla_analysis"]:
            report.add_section(
                "XLA Optimization Analysis",
                self._format_xla_analysis(results["xla_analysis"]),
            )

        # Generate priority recommendations
        priority_recommendations = self._extract_priority_recommendations(results)
        for rec in priority_recommendations:
            report.add_priority_recommendation(
                rec["text"], rec["impact"], rec["effort"]
            )

        return report

    def _create_executive_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Create executive summary from profiling results."""

        summary = {
            "Operation": results.get("operation_name", "Unknown"),
            "Backend": jax.default_backend(),
            "Device Count": jax.device_count(),
        }

        # Roofline metrics
        if (
            "roofline_analysis" in results
            and "error" not in results["roofline_analysis"]
        ):
            roofline = results["roofline_analysis"]
            summary["Arithmetic Intensity"] = (
                f"{roofline.get('arithmetic_intensity', 0):.1f} FLOPs/byte"
            )
            summary["Bottleneck"] = roofline.get("bottleneck", "Unknown")
            summary["Efficiency"] = f"{roofline.get('efficiency', 0):.2%}"

        # Hardware utilization
        if (
            "hardware_analysis" in results
            and "error" not in results["hardware_analysis"]
        ):
            hw_analysis = results["hardware_analysis"]
            backend = hw_analysis.get("backend", "unknown")

            if backend == "tpu" and "platform_analysis" in hw_analysis:
                mxu_analysis = hw_analysis["platform_analysis"].get("mxu_analysis", {})
                summary["MXU Utilization"] = (
                    f"{mxu_analysis.get('mxu_utilization', 0):.2%}"
                )
            elif backend == "gpu" and "platform_analysis" in hw_analysis:
                tc_analysis = hw_analysis["platform_analysis"].get(
                    "tensorcore_analysis", {}
                )
                summary["TensorCore Utilization"] = (
                    f"{tc_analysis.get('tensorcore_utilization', 0):.2%}"
                )

        # Compilation metrics
        if (
            "compilation_analysis" in results
            and "error" not in results["compilation_analysis"]
        ):
            comp_analysis = results["compilation_analysis"]
            cache_stats = comp_analysis.get("cache_statistics", {})
            summary["Cache Hit Rate"] = f"{cache_stats.get('cache_hit_rate', 0):.2%}"

        return summary

    def _format_hardware_analysis(self, hw_analysis: dict[str, Any]) -> dict[str, Any]:
        """Format hardware analysis for report."""

        formatted = {
            "Backend": hw_analysis.get("backend", "Unknown"),
            "Device Count": len(
                hw_analysis.get("hardware_info", {}).get("devices", [])
            ),
        }

        platform_analysis = hw_analysis.get("platform_analysis", {})

        if "mxu_analysis" in platform_analysis:
            mxu = platform_analysis["mxu_analysis"]
            formatted["TPU MXU Utilization"] = f"{mxu.get('mxu_utilization', 0):.2%}"
            formatted["TPU Recommendations"] = mxu.get("recommendations", [])

        if "tensorcore_analysis" in platform_analysis:
            tc = platform_analysis["tensorcore_analysis"]
            formatted["GPU TensorCore Utilization"] = (
                f"{tc.get('tensorcore_utilization', 0):.2%}"
            )
            formatted["GPU Recommendations"] = tc.get("recommendations", [])

        if "simd_analysis" in platform_analysis:
            simd = platform_analysis["simd_analysis"]
            formatted["CPU SIMD Alignment"] = (
                f"{simd.get('simd_alignment', {}).get('average_simd_alignment', 0):.2%}"
            )
            formatted["CPU Recommendations"] = simd.get("recommendations", [])

        return formatted

    def _format_roofline_analysis(self, roofline: dict[str, Any]) -> dict[str, Any]:
        """Format roofline analysis for report."""

        return {
            "Arithmetic Intensity": (
                f"{roofline.get('arithmetic_intensity', 0):.1f} FLOPs/byte"
            ),
            "Critical Intensity": (
                f"{roofline.get('critical_intensity', 0):.1f} FLOPs/byte"
            ),
            "Bottleneck": roofline.get("bottleneck", "Unknown"),
            "Efficiency": f"{roofline.get('efficiency', 0):.2%}",
            "Execution Time": f"{roofline.get('actual_time_ms', 0):.2f} ms",
            "FLOP Utilization": f"{roofline.get('flops_utilization', 0):.2%}",
            "Memory Bandwidth Utilization": (
                f"{roofline.get('memory_bandwidth_utilization', 0):.2%}"
            ),
            "Recommendations": roofline.get("optimization_recommendations", []),
        }

    def _format_compilation_analysis(
        self, comp_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Format compilation analysis for report."""

        cache_stats = comp_analysis.get("cache_statistics", {})
        comp_times = comp_analysis.get("compilation_times", {})

        return {
            "Cache Hit Rate": f"{cache_stats.get('cache_hit_rate', 0):.2%}",
            "Total Calls": cache_stats.get("total_calls", 0),
            "Average Compilation Time": f"{comp_times.get('average_ms', 0):.1f} ms",
            "Maximum Compilation Time": f"{comp_times.get('maximum_ms', 0):.1f} ms",
            "Unique Signatures": comp_analysis.get("unique_signatures", 0),
            "Recommendations": comp_analysis.get("recommendations", []),
        }

    def _format_xla_analysis(self, xla_analysis: dict[str, Any]) -> dict[str, Any]:
        """Format XLA analysis for report."""

        hlo_analysis = xla_analysis.get("hlo_analysis", {})

        return {
            "Optimization Score": f"{xla_analysis.get('optimization_score', 0):.2%}",
            "Fusion Ratio": f"{hlo_analysis.get('fusion_ratio', 0):.2%}",
            "Arithmetic Ratio": f"{hlo_analysis.get('arithmetic_ratio', 0):.2%}",
            "Memory Operation Ratio": f"{hlo_analysis.get('memory_ratio', 0):.2%}",
            "Total Operations": hlo_analysis.get("total_operations", 0),
            "Recommendations": xla_analysis.get("recommendations", []),
        }

    def _extract_priority_recommendations(
        self, results: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Extract and prioritize recommendations from all analyses."""

        all_recommendations = []

        # Collect recommendations from all analyses
        for analysis_name, analysis_data in results.items():
            if isinstance(analysis_data, dict) and "recommendations" in analysis_data:
                for rec in analysis_data["recommendations"]:
                    # Determine impact and effort based on recommendation content
                    impact, effort = self._assess_recommendation_priority(rec)
                    all_recommendations.append(
                        {
                            "text": rec,
                            "source": analysis_name,
                            "impact": impact,
                            "effort": effort,
                        }
                    )

        # Sort by impact (high impact first)
        impact_order = {"high": 3, "medium": 2, "low": 1}
        all_recommendations.sort(
            key=lambda x: (impact_order.get(x["impact"], 0), -len(x["text"])),
            reverse=True,
        )

        return all_recommendations[:10]  # Top 10 recommendations

    def _assess_recommendation_priority(self, recommendation: str) -> tuple[str, str]:
        """Assess the impact and effort of a recommendation."""

        rec_lower = recommendation.lower()

        # High impact indicators
        if any(
            indicator in rec_lower
            for indicator in [
                "batch size",
                "memory-bound",
                "low utilization",
                "cache hit",
            ]
        ):
            impact = "high"
        elif any(
            indicator in rec_lower for indicator in ["alignment", "precision", "fusion"]
        ):
            impact = "medium"
        else:
            impact = "low"

        # Effort assessment
        if any(indicator in rec_lower for indicator in ["increase", "use", "enable"]):
            effort = "low"
        elif any(
            indicator in rec_lower
            for indicator in ["optimize", "consider", "fine-tune"]
        ):
            effort = "medium"
        elif any(
            indicator in rec_lower
            for indicator in ["redesign", "implement", "break down"]
        ):
            effort = "high"
        else:
            effort = "medium"

        return impact, effort

    def _analyze_operation_comparison(self, results: dict[str, Any]) -> dict[str, Any]:
        """Analyze comparison between multiple operations."""

        valid_results = {k: v for k, v in results.items() if "error" not in v}

        if not valid_results:
            return {"error": "No valid results to compare"}

        # Extract key metrics for comparison
        metrics = {}
        for name, result in valid_results.items():
            metrics[name] = {}

            # Roofline metrics
            if "roofline_analysis" in result:
                roofline = result["roofline_analysis"]
                metrics[name]["efficiency"] = roofline.get("efficiency", 0)
                metrics[name]["arithmetic_intensity"] = roofline.get(
                    "arithmetic_intensity", 0
                )
                metrics[name]["execution_time_ms"] = roofline.get("actual_time_ms", 0)

        # Find best and worst performers
        if metrics:
            best_efficiency = max(
                metrics.keys(), key=lambda k: metrics[k].get("efficiency", 0)
            )
            worst_efficiency = min(
                metrics.keys(), key=lambda k: metrics[k].get("efficiency", 0)
            )
            fastest_execution = min(
                metrics.keys(),
                key=lambda k: metrics[k].get("execution_time_ms", float("inf")),
            )

            return {
                "best_efficiency": best_efficiency,
                "worst_efficiency": worst_efficiency,
                "fastest_execution": fastest_execution,
                "metrics_summary": metrics,
            }

        return {"error": "No comparable metrics found"}

    def _generate_comparison_recommendations(
        self, comparison: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on operation comparison."""

        recommendations = []

        if "error" in comparison:
            return [f"Comparison analysis failed: {comparison['error']}"]

        best_efficiency = comparison.get("best_efficiency")
        worst_efficiency = comparison.get("worst_efficiency")
        fastest_execution = comparison.get("fastest_execution")

        if best_efficiency and worst_efficiency and best_efficiency != worst_efficiency:
            recommendations.append(
                f"💡 Model '{best_efficiency}' shows best efficiency. "
                f"Consider applying its optimization patterns to '{worst_efficiency}'"
            )

        if fastest_execution:
            recommendations.append(
                f"⚡ '{fastest_execution}' has the fastest execution time. "
                f"Analyze its implementation for performance insights"
            )

        return recommendations

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary of all profiling sessions."""

        if not self.profiling_sessions:
            return {"message": "No profiling sessions recorded"}

        total_sessions = len(self.profiling_sessions)
        successful_sessions = sum(
            1 for s in self.profiling_sessions if s.get("success", False)
        )
        total_duration = sum(s.get("duration", 0) for s in self.profiling_sessions)

        return {
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "success_rate": successful_sessions / total_sessions
            if total_sessions > 0
            else 0,
            "total_duration_s": total_duration,
            "average_duration_s": total_duration / total_sessions
            if total_sessions > 0
            else 0,
            "profilers_used": list(self.profilers.keys()),
            "coordinator_summary": self.coordinator.get_profiling_summary(),
        }

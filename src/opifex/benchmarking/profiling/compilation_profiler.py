"""
JIT Compilation Profiler for JAX.

Analyzes JIT compilation efficiency, cache hit rates, XLA optimization
effectiveness, and provides recommendations for compilation optimization.
"""

import hashlib
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import jax

from .event_coordinator import EventCoordinator


class CompilationProfiler:
    """Analyzes JAX JIT compilation performance and optimization."""

    def __init__(self, coordinator: EventCoordinator | None = None):
        self.coordinator = coordinator
        self.compilation_cache = {}
        self.compilation_stats = defaultdict(list)
        self.cache_hit_count = 0
        self.cache_miss_count = 0

        if coordinator:
            coordinator.register_profiler("compilation_profiler")

    def profile_jit_compilation(self, func: Callable) -> Callable:
        """Create an instrumented function that profiles JIT compilation."""

        def instrumented_func(*args, **kwargs):
            # Create signature for cache analysis
            signature = self._create_function_signature(func, args, kwargs)

            compilation_start = time.time()

            if signature in self.compilation_cache:
                # Cache hit
                compiled_func = self.compilation_cache[signature]["compiled_func"]
                self.cache_hit_count += 1
                compilation_time = 0

                if self.coordinator:
                    self.coordinator.add_event(
                        "compilation_cache_hit",
                        "compilation_profiler",
                        {"signature": signature, "hit_count": self.cache_hit_count},
                    )
            else:
                # Cache miss - compile function
                compiled_func = jax.jit(func)

                # Trigger compilation by running once
                try:
                    warmup_result = compiled_func(*args, **kwargs)
                    if hasattr(warmup_result, "block_until_ready"):
                        warmup_result.block_until_ready()
                except Exception as e:
                    # Compilation failed
                    compilation_time = time.time() - compilation_start
                    if self.coordinator:
                        self.coordinator.add_event(
                            "compilation_error",
                            "compilation_profiler",
                            {"signature": signature, "error": str(e)},
                            duration_ms=compilation_time * 1000,
                        )
                    raise

                compilation_time = time.time() - compilation_start
                self.cache_miss_count += 1

                # Store in cache
                self.compilation_cache[signature] = {
                    "compiled_func": compiled_func,
                    "compilation_time": compilation_time,
                    "input_shapes": [getattr(arg, "shape", None) for arg in args],
                    "input_dtypes": [getattr(arg, "dtype", None) for arg in args],
                    "first_compiled": time.time(),
                }

                # Record compilation stats
                self.compilation_stats[signature].append(
                    {
                        "compilation_time": compilation_time,
                        "timestamp": time.time(),
                        "input_characteristics": self._analyze_input_characteristics(
                            args
                        ),
                    }
                )

                if self.coordinator:
                    self.coordinator.add_event(
                        "compilation_cache_miss",
                        "compilation_profiler",
                        {
                            "signature": signature,
                            "compilation_time_ms": compilation_time * 1000,
                            "input_shapes": [
                                getattr(arg, "shape", None) for arg in args
                            ],
                            "miss_count": self.cache_miss_count,
                        },
                        duration_ms=compilation_time * 1000,
                    )

            # Execute the compiled function
            execution_start = time.time()
            result = compiled_func(*args, **kwargs)

            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            elif isinstance(result, (tuple, list)):
                for item in result:
                    if hasattr(item, "block_until_ready"):
                        item.block_until_ready()

            execution_time = time.time() - execution_start

            # Record execution stats
            if self.coordinator:
                self.coordinator.add_event(
                    "jit_execution",
                    "compilation_profiler",
                    {
                        "signature": signature,
                        "execution_time_ms": execution_time * 1000,
                        "compilation_time_ms": compilation_time * 1000,
                        "cache_hit": compilation_time == 0,
                    },
                    duration_ms=execution_time * 1000,
                )

            return result

        return instrumented_func

    def analyze_compilation_efficiency(self) -> dict[str, Any]:
        """Analyze overall compilation efficiency."""

        total_calls = self.cache_hit_count + self.cache_miss_count
        cache_hit_rate = self.cache_hit_count / total_calls if total_calls > 0 else 0

        # Analyze compilation times
        all_compilation_times = []
        for signature_stats in self.compilation_stats.values():
            for stat in signature_stats:
                all_compilation_times.append(stat["compilation_time"])

        if all_compilation_times:
            avg_compilation_time = sum(all_compilation_times) / len(
                all_compilation_times
            )
        else:
            avg_compilation_time = 0

        # Find expensive compilations
        expensive_compilations = []
        for signature, stats_list in self.compilation_stats.items():
            for stat in stats_list:
                if stat["compilation_time"] > 1.0:  # > 1 second
                    expensive_compilations.append(
                        {
                            "signature": signature,
                            "compilation_time": stat["compilation_time"],
                            "input_characteristics": stat["input_characteristics"],
                        }
                    )

        # Sort by compilation time
        expensive_compilations.sort(key=lambda x: x["compilation_time"], reverse=True)

        return {
            "cache_statistics": {
                "total_calls": total_calls,
                "cache_hits": self.cache_hit_count,
                "cache_misses": self.cache_miss_count,
                "cache_hit_rate": cache_hit_rate,
            },
            "compilation_times": {
                "average_ms": avg_compilation_time * 1000,
                "maximum_ms": max(all_compilation_times) * 1000
                if all_compilation_times
                else 0,
                "minimum_ms": min(all_compilation_times) * 1000
                if all_compilation_times
                else 0,
                "total_compilations": len(all_compilation_times),
            },
            "expensive_compilations": expensive_compilations[:10],  # Top 10
            "unique_signatures": len(self.compilation_cache),
            "recommendations": self._generate_compilation_recommendations(
                cache_hit_rate, avg_compilation_time, expensive_compilations
            ),
        }

    def analyze_shape_consistency(self) -> dict[str, Any]:
        """Analyze input shape consistency to identify recompilation causes."""

        shape_patterns = defaultdict(list)

        # Group signatures by shape patterns
        for signature, cache_entry in self.compilation_cache.items():
            shapes = tuple(cache_entry["input_shapes"])
            dtypes = tuple(str(dt) for dt in cache_entry["input_dtypes"])
            pattern = (shapes, dtypes)
            shape_patterns[pattern].append(signature)

        # Analyze shape diversity
        total_patterns = len(shape_patterns)
        total_signatures = len(self.compilation_cache)

        # Find most common patterns
        pattern_frequency = [
            (pattern, len(signatures)) for pattern, signatures in shape_patterns.items()
        ]
        pattern_frequency.sort(key=lambda x: x[1], reverse=True)

        # Identify potential shape optimization opportunities
        optimization_opportunities = []
        for pattern, signatures in shape_patterns.items():
            if len(signatures) > 1:
                shapes, dtypes = pattern
                optimization_opportunities.append(
                    {
                        "pattern": {"shapes": shapes, "dtypes": dtypes},
                        "signature_count": len(signatures),
                        "recommendation": self._suggest_shape_optimization(shapes),
                    }
                )

        return {
            "total_shape_patterns": total_patterns,
            "total_signatures": total_signatures,
            "pattern_diversity_ratio": total_patterns / total_signatures
            if total_signatures > 0
            else 0,
            "most_common_patterns": pattern_frequency[:5],
            "optimization_opportunities": optimization_opportunities,
            "recommendations": self._generate_shape_recommendations(
                total_patterns, total_signatures
            ),
        }

    def estimate_xla_optimization_effectiveness(
        self, func: Callable, sample_args: tuple
    ) -> dict[str, Any]:
        """Estimate XLA optimization effectiveness by analyzing HLO."""

        try:
            # Create JIT-compiled version
            jit_func = jax.jit(func)

            # Get lowered representation
            lowered = jit_func.lower(*sample_args)

            # Get HLO text representation
            hlo_text = lowered.compile().as_text() or ""

            # Analyze HLO for optimization patterns
            optimization_analysis = self._analyze_hlo_optimizations(hlo_text)

            return {
                "hlo_analysis": optimization_analysis,
                "optimization_score": self._calculate_optimization_score(
                    optimization_analysis
                ),
                "recommendations": self._generate_xla_recommendations(
                    optimization_analysis
                ),
            }

        except Exception as e:
            return {
                "error": f"Failed to analyze XLA optimizations: {e!s}",
                "hlo_analysis": {},
                "optimization_score": 0.0,
                "recommendations": ["Unable to analyze XLA optimizations"],
            }

    def _create_function_signature(
        self, func: Callable, args: tuple, kwargs: dict
    ) -> str:
        """Create a unique signature for function + arguments."""

        # Function identifier
        func_id = getattr(func, "__name__", str(func))

        # Argument shapes and types
        arg_signature = []
        for arg in args:
            if hasattr(arg, "shape") and hasattr(arg, "dtype"):
                arg_signature.append(f"{arg.shape}:{arg.dtype}")
            else:
                arg_signature.append(str(type(arg)))

        # Static kwargs (non-array arguments)
        static_kwargs = {k: v for k, v in kwargs.items() if not hasattr(v, "shape")}

        # Create hash
        signature_str = f"{func_id}({','.join(arg_signature)}){static_kwargs}"
        return hashlib.md5(signature_str.encode(), usedforsecurity=False).hexdigest()

    def _analyze_input_characteristics(self, args: tuple) -> dict[str, Any]:
        """Analyze characteristics of input arguments."""

        characteristics = {
            "num_args": len(args),
            "total_elements": 0,
            "total_bytes": 0,
            "dtypes": set(),
            "shapes": [],
            "max_dimension": 0,
        }

        for arg in args:
            if hasattr(arg, "shape") and hasattr(arg, "dtype"):
                characteristics["total_elements"] += arg.size
                characteristics["total_bytes"] += arg.size * arg.dtype.itemsize
                characteristics["dtypes"].add(str(arg.dtype))
                characteristics["shapes"].append(arg.shape)

                if arg.shape:
                    characteristics["max_dimension"] = max(
                        characteristics["max_dimension"], *arg.shape
                    )

        characteristics["dtypes"] = list(characteristics["dtypes"])
        return characteristics

    def _analyze_hlo_optimizations(self, hlo_text: str) -> dict[str, Any]:
        """Analyze HLO text for optimization patterns."""

        lines = hlo_text.split("\n")

        analysis = {
            "total_kernels": 0,
            "fused_kernels": 0,
            "memory_operations": 0,
            "arithmetic_operations": 0,
            "communication_operations": 0,
            "optimization_patterns": [],
            "memory_usage": 0,
            "max_memory": 0,
        }

        # Pattern matching for common HLO operations
        fusion_patterns = ["fusion", "kLoop", "kInput", "kOutput"]
        memory_patterns = ["copy", "transpose", "reshape", "broadcast"]
        arithmetic_patterns = ["add", "multiply", "dot", "convolution", "reduce"]
        communication_patterns = ["all-reduce", "all-gather", "reduce-scatter"]

        for line in lines:
            line_lower = line.lower()

            # Count operations
            if any(pattern in line_lower for pattern in fusion_patterns):
                analysis["fused_kernels"] += 1
                analysis["optimization_patterns"].append("fusion")

            if any(pattern in line_lower for pattern in memory_patterns):
                analysis["memory_operations"] += 1

            if any(pattern in line_lower for pattern in arithmetic_patterns):
                analysis["arithmetic_operations"] += 1

            if any(pattern in line_lower for pattern in communication_patterns):
                analysis["communication_operations"] += 1

            analysis["total_kernels"] += 1

            # Heuristic memory parsing (e.g., "total bytes: 1234")
            if "total bytes" in line_lower:
                try:
                    # Extract number after "total bytes"
                    parts = line_lower.split("total bytes")
                    if len(parts) > 1:
                        # Look for digits in the part after "total bytes"
                        import re

                        match = re.search(r"(\d+)", parts[1])
                        if match:
                            mem_bytes = int(match.group(1))
                            analysis["memory_usage"] = max(
                                analysis["memory_usage"], mem_bytes
                            )
                            analysis["max_memory"] = max(
                                analysis["max_memory"], mem_bytes
                            )
                except Exception:
                    pass

        # Calculate ratios
        if analysis["total_kernels"] > 0:
            analysis["fusion_ratio"] = (
                analysis["fused_kernels"] / analysis["total_kernels"]
            )
            analysis["arithmetic_ratio"] = (
                analysis["arithmetic_operations"] / analysis["total_kernels"]
            )
            analysis["memory_ratio"] = (
                analysis["memory_operations"] / analysis["total_kernels"]
            )
        else:
            analysis["fusion_ratio"] = analysis["arithmetic_ratio"] = analysis[
                "memory_ratio"
            ] = 0

        return analysis

    def _calculate_optimization_score(self, hlo_analysis: dict[str, Any]) -> float:
        """Calculate an optimization effectiveness score (0-1)."""

        score = 0.0

        # Fusion is generally good for performance
        fusion_score = min(hlo_analysis.get("fusion_ratio", 0) * 2, 1.0)  # Cap at 1.0
        score += fusion_score * 0.4

        # High arithmetic ratio is good
        arithmetic_score = hlo_analysis.get("arithmetic_ratio", 0)
        score += arithmetic_score * 0.3

        # Lower memory operation ratio is generally better
        memory_ratio = hlo_analysis.get("memory_ratio", 0)

        memory_score = max(0, 1 - memory_ratio)
        score += memory_score * 0.3

        return min(score, 1.0)

    def _generate_compilation_recommendations(
        self,
        cache_hit_rate: float,
        avg_compilation_time: float,
        expensive_compilations: list[dict],
    ) -> list[str]:
        """Generate compilation optimization recommendations."""

        recommendations = []

        # Cache hit rate recommendations
        if cache_hit_rate < 0.5:
            recommendations.extend(
                [
                    f"⚠️ Low cache hit rate ({cache_hit_rate:.2%})",
                    "⚠️  Low compilation cache hit rate. "
                    "Ensure consistent shapes and static arguments.",
                    "💡 Use `static_argnums` for non-array arguments in `jax.jit`.",
                ]
            )
        elif cache_hit_rate < 0.8:
            recommendations.extend(
                [
                    f"⚡ Moderate cache hit rate ({cache_hit_rate:.2%})",
                    "💡 Fine-tune input preprocessing for better shape consistency",
                ]
            )
        else:
            recommendations.append(f"✅ Good cache hit rate ({cache_hit_rate:.2%})")

        # Compilation time recommendations
        if avg_compilation_time > 5.0:
            recommendations.extend(
                [
                    f"⚠️ High average compilation time ({avg_compilation_time:.2f}s)",
                    "💡 Break down large functions into smaller, composable parts",
                    "💡 Use hierarchical compilation strategies",
                    "💡 Consider pre-compilation for critical paths",
                ]
            )
        elif avg_compilation_time > 2.0:
            recommendations.extend(
                [
                    f"⚡ Moderate compilation time ({avg_compilation_time:.2f}s)",
                    "💡 Monitor for complex control flow that may slow compilation",
                ]
            )

        # Expensive compilation recommendations
        if expensive_compilations:
            recommendations.extend(
                [
                    f"⚠️ {len(expensive_compilations)} expensive compilations detected",
                    "💡 Profile expensive compilations for optimization opportunities",
                    "💡 Consider compilation caching strategies for development",
                ]
            )

        return recommendations

    def _generate_shape_recommendations(
        self, total_patterns: int, total_signatures: int
    ) -> list[str]:
        """Generate shape consistency recommendations."""

        recommendations = []

        diversity_ratio = (
            total_patterns / total_signatures if total_signatures > 0 else 0
        )

        if diversity_ratio > 0.8:
            recommendations.extend(
                [
                    f"⚠️ High shape diversity ({diversity_ratio:.2%})",
                    "💡 Implement shape bucketing to group similar shapes",
                    "💡 Use padding to standardize tensor dimensions",
                    "💡 Consider dynamic shape handling strategies",
                ]
            )
        elif diversity_ratio > 0.5:
            recommendations.extend(
                [
                    f"⚡ Moderate shape diversity ({diversity_ratio:.2%})",
                    "💡 Look for opportunities to standardize input shapes",
                ]
            )
        else:
            recommendations.append(f"✅ Good shape consistency ({diversity_ratio:.2%})")

        return recommendations

    def _suggest_shape_optimization(self, shapes: tuple) -> str:
        """Suggest optimization for a specific shape pattern."""

        if not shapes:
            return "No shape optimization needed"

        # Analyze shape characteristics
        max_dims = []
        for shape in shapes:
            if shape:
                max_dims.append(max(shape))

        if max_dims:
            max_dim = max(max_dims)

            # Suggest padding to next power of 2 or common multiple
            if max_dim < 128:
                target = 128
            elif max_dim < 256:
                target = 256
            elif max_dim < 512:
                target = 512
            else:
                target = ((max_dim + 127) // 128) * 128  # Round up to multiple of 128

            return (
                f"Consider padding dimensions to {target} for better hardware alignment"
            )

        return "Shape optimization not applicable"

    def _generate_xla_recommendations(self, hlo_analysis: dict[str, Any]) -> list[str]:
        """Generate XLA optimization recommendations."""

        recommendations = []

        fusion_ratio = hlo_analysis.get("fusion_ratio", 0)
        arithmetic_ratio = hlo_analysis.get("arithmetic_ratio", 0)
        memory_ratio = hlo_analysis.get("memory_ratio", 0)

        if fusion_ratio < 0.2:
            recommendations.extend(
                [
                    "⚠️ Low fusion ratio - operations may not be well-fused",
                    "💡 Combine operations to enable better fusion",
                    "💡 Avoid unnecessary intermediate variables",
                ]
            )

        if arithmetic_ratio < 0.3:
            recommendations.extend(
                [
                    "⚠️ Low arithmetic intensity",
                    "💡 Increase computational density of operations",
                    "💡 Consider batching to improve arithmetic intensity",
                ]
            )

        if memory_ratio > 0.5:
            recommendations.extend(
                [
                    "⚠️ High memory operation ratio",
                    "💡 Reduce unnecessary data movement",
                    "💡 Optimize tensor layouts and access patterns",
                ]
            )

        if not recommendations:
            recommendations.append("✅ XLA optimizations appear effective")

        return recommendations

    def get_compilation_summary(self) -> dict[str, Any]:
        """Get a comprehensive compilation performance summary."""

        efficiency_analysis = self.analyze_compilation_efficiency()
        shape_analysis = self.analyze_shape_consistency()

        return {
            "compilation_efficiency": efficiency_analysis,
            "shape_consistency": shape_analysis,
            "overall_health": self._assess_overall_compilation_health(
                efficiency_analysis, shape_analysis
            ),
        }

    def _assess_overall_compilation_health(
        self, efficiency: dict, shapes: dict
    ) -> dict[str, Any]:
        """Assess overall compilation health."""

        cache_hit_rate = efficiency["cache_statistics"]["cache_hit_rate"]
        avg_compilation_time = efficiency["compilation_times"]["average_ms"] / 1000
        shape_diversity = shapes["pattern_diversity_ratio"]

        # Calculate health score (0-1)
        health_score = 0.0

        # Cache efficiency (40% weight)
        health_score += cache_hit_rate * 0.4

        # Compilation speed (30% weight)
        compilation_speed_score = max(
            0, 1 - (avg_compilation_time / 10)
        )  # Normalize to 10s max
        health_score += compilation_speed_score * 0.3

        # Shape consistency (30% weight)
        shape_consistency_score = max(0, 1 - shape_diversity)
        health_score += shape_consistency_score * 0.3

        # Determine health level
        if health_score > 0.8:
            health_level = "excellent"
        elif health_score > 0.6:
            health_level = "good"
        elif health_score > 0.4:
            health_level = "moderate"
        else:
            health_level = "poor"

        return {
            "health_score": health_score,
            "health_level": health_level,
            "key_metrics": {
                "cache_hit_rate": cache_hit_rate,
                "avg_compilation_time_s": avg_compilation_time,
                "shape_diversity_ratio": shape_diversity,
            },
        }

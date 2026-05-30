"""
Opifex Core Optimization Module.

This module provides advanced optimization techniques for JAX-based neural operators,
including operation fusion, memory layout optimization, and hardware-specific
performance enhancements.
"""

from opifex.core.optimization.fusion_optimizer import (
    create_fused_computation_graph,
    fused_conv_activation,
    fused_elementwise_chain,
    fused_linear_activation,
    fused_spectral_conv_activation,
    FusedFourierLayer,
    FusionOptimizedOperator,
    optimize_memory_layout_for_fusion,
)
from opifex.core.optimization.memory_layout import (
    benchmark_layout_performance,
    create_layout_optimization_report,
    LayoutOptimizer,
    MemoryLayout,
    OptimizedConvolution,
    OptimizedLinear,
)


__all__ = [
    # Fusion optimization
    "FusedFourierLayer",
    "FusionOptimizedOperator",
    # Memory layout optimization
    "LayoutOptimizer",
    "MemoryLayout",
    "OptimizedConvolution",
    "OptimizedLinear",
    "benchmark_layout_performance",
    "create_fused_computation_graph",
    "create_layout_optimization_report",
    "fused_conv_activation",
    "fused_elementwise_chain",
    "fused_linear_activation",
    "fused_spectral_conv_activation",
    "optimize_memory_layout_for_fusion",
]

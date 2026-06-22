"""
Opifex Core Module

This module provides the core functionality for the Opifex framework,
including problem definitions, conditions, and quantum chemistry components.
"""

from opifex.core.conditions import (
    DirichletBC,
    InitialCondition,
    NeumannBC,
    RobinBC,
)
from opifex.core.device_utils import (
    configure_jax_precision,
    get_device_info,
    get_platform,
    is_gpu_available,
)
from opifex.core.gpu_acceleration import (
    AsyncMemoryManager,
    CachedProgressiveTester,
    MemoryPoolManager,
    MixedPrecisionOptimizer,
    OptimizedGPUManager,
    RooflineMemoryManager,
)
from opifex.core.problems import (
    create_neural_dft_problem,
    create_ode_problem,
    create_optimization_problem,
    create_pde_problem,
    ElectronicStructureProblem,
    ODEProblem,
    OptimizationProblem,
    PDEProblem,
    Problem,
    QuantumProblem,
)
from opifex.core.quantum.molecular_system import (
    create_molecular_system,
    MolecularSystem,
)


__all__ = [
    "AsyncMemoryManager",
    "CachedProgressiveTester",
    "DirichletBC",
    "ElectronicStructureProblem",
    "InitialCondition",
    "MemoryPoolManager",
    "MixedPrecisionOptimizer",
    "MolecularSystem",
    "NeumannBC",
    "ODEProblem",
    "OptimizationProblem",
    "OptimizedGPUManager",
    "PDEProblem",
    "Problem",
    "QuantumProblem",
    "RobinBC",
    "RooflineMemoryManager",
    "configure_jax_precision",
    "create_molecular_system",
    "create_neural_dft_problem",
    "create_ode_problem",
    "create_optimization_problem",
    "create_pde_problem",
    "get_device_info",
    "get_platform",
    "is_gpu_available",
]

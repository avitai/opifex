"""PDEBench Benchmark Configurations.

Standard benchmark configurations matching PDEBench dataset specifications.
These follow the DRY principle by using explicit loader_type in
computational_requirements rather than inferring from name.
"""

from opifex.benchmarking.benchmark_registry import BenchmarkConfig, BenchmarkRegistry


PDEBENCH_BENCHMARKS = [
    BenchmarkConfig(
        name="PDEBench_2D_DarcyFlow",
        domain="fluid_dynamics",
        problem_type="operator_learning",
        input_shape=(128, 128, 1),
        output_shape=(128, 128, 1),
        physics_constraints={
            "equation": "-div(a * grad(u)) = f",
            "boundary_conditions": "dirichlet",
        },
        computational_requirements={
            "loader_type": "darcy",  # DRY: Explicit loader type, no name parsing
            "batch_size": 32,
            "n_epochs": 500,
        },
    ),
    BenchmarkConfig(
        name="PDEBench_1D_Burgers",
        domain="fluid_dynamics",
        problem_type="time_evolution",
        input_shape=(1024, 1),
        output_shape=(1024, 201, 1),
        physics_constraints={
            "equation": "u_t + u*u_x = nu*u_xx",
            "viscosity": [0.001, 0.01, 0.1],
        },
        computational_requirements={
            "loader_type": "burgers",  # DRY: Explicit loader type
            "batch_size": 32,
            "n_epochs": 500,
        },
    ),
    BenchmarkConfig(
        name="PDEBench_2D_NavierStokes",
        domain="fluid_dynamics",
        problem_type="time_evolution",
        input_shape=(128, 128, 2),
        output_shape=(128, 128, 21, 2),
        physics_constraints={
            "equation": "incompressible_navier_stokes",
            "reynolds_number": [100, 1000],
        },
        computational_requirements={
            "loader_type": "navier_stokes",  # DRY: Explicit loader type
            "batch_size": 16,
            "n_epochs": 500,
        },
    ),
    BenchmarkConfig(
        name="PDEBench_2D_ShallowWater",
        domain="fluid_dynamics",
        problem_type="time_evolution",
        input_shape=(128, 128, 3),
        output_shape=(128, 128, 101, 3),
        physics_constraints={
            "equation": "shallow_water_equations",
            "gravity": 9.81,
        },
        computational_requirements={
            "loader_type": "shallow_water",
            "batch_size": 16,
            "n_epochs": 500,
        },
    ),
    BenchmarkConfig(
        name="PDEBench_2D_Diffusion",
        domain="fluid_dynamics",
        problem_type="time_evolution",
        input_shape=(64, 64, 1),
        output_shape=(64, 64, 5, 1),
        physics_constraints={
            "equation": "u_t = D * laplacian(u) + v * grad(u)",
            "diffusion_coefficient": 0.01,
        },
        computational_requirements={
            "loader_type": "diffusion",
            "batch_size": 32,
            "n_epochs": 500,
        },
    ),
]


REALPDEBENCH_BENCHMARKS = [
    BenchmarkConfig(
        name="RealPDEBench_Cylinder",
        domain="fluid_dynamics",
        problem_type="time_evolution",
        input_shape=(128, 256, 2),
        output_shape=(128, 256, 2),
        physics_constraints={
            "equation": "navier_stokes_vortex_shedding",
            "reynolds_number": [100, 200, 400],
        },
        computational_requirements={
            "loader_type": "navier_stokes",  # Uses NS loader
            "batch_size": 16,
            "n_epochs": 500,
            "training_paradigm": "numerical",  # numerical, real, or finetune
        },
    ),
    BenchmarkConfig(
        name="RealPDEBench_FSI",
        domain="fluid_dynamics",
        problem_type="time_evolution",
        input_shape=(128, 256, 2),
        output_shape=(128, 256, 2),
        physics_constraints={
            "equation": "fluid_structure_interaction",
            "coupling": "two_way",
        },
        computational_requirements={
            "loader_type": "navier_stokes",
            "batch_size": 16,
            "n_epochs": 500,
        },
    ),
    BenchmarkConfig(
        name="RealPDEBench_Combustion",
        domain="fluid_dynamics",
        problem_type="time_evolution",
        input_shape=(128, 128, 4),  # u, v, T, species
        output_shape=(128, 128, 4),
        physics_constraints={
            "equation": "reacting_navier_stokes",
            "includes_temperature": True,
            "includes_species": True,
        },
        computational_requirements={
            "loader_type": "navier_stokes",
            "batch_size": 8,
            "n_epochs": 500,
        },
    ),
]


def register_pdebench_benchmarks(registry: BenchmarkRegistry) -> None:
    """Register all PDEBench benchmarks with a registry.

    Args:
        registry: BenchmarkRegistry instance to register benchmarks with
    """
    for benchmark in PDEBENCH_BENCHMARKS:
        registry.register_benchmark(benchmark)


def register_realpdebench_benchmarks(registry: BenchmarkRegistry) -> None:
    """Register all RealPDEBench benchmarks with a registry.

    Args:
        registry: BenchmarkRegistry instance to register benchmarks with
    """
    for benchmark in REALPDEBENCH_BENCHMARKS:
        registry.register_benchmark(benchmark)


def register_all_benchmarks(registry: BenchmarkRegistry) -> None:
    """Register all benchmarks (PDEBench + RealPDEBench) with a registry.

    Args:
        registry: BenchmarkRegistry instance to register benchmarks with
    """
    register_pdebench_benchmarks(registry)
    register_realpdebench_benchmarks(registry)


__all__ = [
    "PDEBENCH_BENCHMARKS",
    "REALPDEBENCH_BENCHMARKS",
    "register_all_benchmarks",
    "register_pdebench_benchmarks",
    "register_realpdebench_benchmarks",
]

"""Physics-specific core modules for Opifex framework.

This package contains physics-related functionality that is shared
across the entire Opifex framework, including:

- Conservation law enforcement
- Physics-informed loss composition
- Boundary condition application
- Autodiff utilities for spatial derivatives
"""

from .autodiff_engine import (
    AutoDiffEngine,
    compute_divergence,
    compute_gradient,
    compute_gradient_nnx,
    compute_hessian,
    compute_laplacian,
    compute_laplacian_nnx,
)
from .boundaries import (
    apply_boundary_condition,
    apply_dirichlet,
    apply_neumann,
    apply_periodic,
    apply_robin,
    BoundaryType,
)
from .conservation import ConservationLaw
from .losses import (
    AdaptiveWeightScheduler,
    ConservationLawEnforcer,
    PhysicsInformedLoss,
    PhysicsLossComposer,
    PhysicsLossConfig,
    ResidualComputer,
)
from .pde_registry import PDEResidualRegistry
from .quantum_constraints import (
    density_positivity_violation,
    hermiticity_violation,
    probability_conservation,
    wavefunction_normalization,
)


__all__ = [
    # Physics losses
    "AdaptiveWeightScheduler",
    # Autodiff utilities
    "AutoDiffEngine",
    # Boundary conditions
    "BoundaryType",
    # Conservation laws
    "ConservationLaw",
    "ConservationLawEnforcer",
    # PDE Registry
    "PDEResidualRegistry",
    "PhysicsInformedLoss",
    "PhysicsLossComposer",
    "PhysicsLossConfig",
    "ResidualComputer",
    "apply_boundary_condition",
    "apply_dirichlet",
    "apply_neumann",
    "apply_periodic",
    "apply_robin",
    "compute_divergence",
    "compute_gradient",
    "compute_gradient_nnx",
    "compute_hessian",
    "compute_laplacian",
    "compute_laplacian_nnx",
    # Quantum constraints
    "density_positivity_violation",
    "hermiticity_violation",
    "probability_conservation",
    "wavefunction_normalization",
]

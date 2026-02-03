"""Physics-specific core modules for Opifex framework.

This package contains physics-related functionality that is shared
across the entire Opifex framework, including:

- Conservation law enforcement
- Physics-informed loss composition
- Boundary condition application
- Autodiff utilities for spatial derivatives
"""

from opifex.core.physics.autodiff_engine import (
    AutoDiffEngine,
    compute_divergence,
    compute_gradient,
    compute_gradient_nnx,
    compute_hessian,
    compute_laplacian,
    compute_laplacian_nnx,
)
from opifex.core.physics.boundaries import (
    apply_boundary_condition,
    apply_dirichlet,
    apply_neumann,
    apply_periodic,
    apply_robin,
    BoundaryType,
)
from opifex.core.physics.conservation import ConservationLaw
from opifex.core.physics.losses import (
    AdaptiveWeightScheduler,
    ConservationLawEnforcer,
    PhysicsInformedLoss,
    PhysicsLossComposer,
    PhysicsLossConfig,
    ResidualComputer,
)
from opifex.core.physics.pde_registry import PDEResidualRegistry
from opifex.core.physics.quantum_constraints import (
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

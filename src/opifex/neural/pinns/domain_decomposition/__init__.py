"""Domain Decomposition Physics-Informed Neural Networks.

This module implements various domain decomposition strategies for PINNs,
enabling scalable training on complex geometries and multi-scale problems.

Key Components:
    - Subdomain: Representation of a subdomain region
    - Interface: Representation of interface between subdomains
    - DomainDecompositionPINN: Base class for all DD-PINNs
    - XPINN: Extended PINN with interface conditions
    - FBPINN: Finite Basis PINN with window functions
    - CPINN: Conservative PINN with flux conservation
    - APINN: Augmented PINN with learnable gating

References:
    - Survey: arXiv:2601.10222v1 Section 8.3
    - XPINNs: https://github.com/AmeyaJagtap/XPINNs
    - FBPINNs: https://github.com/benmoseley/FBPINNs
"""

from opifex.neural.pinns.domain_decomposition.apinn import (
    APINN,
    APINNConfig,
    GatingNetwork,
)
from opifex.neural.pinns.domain_decomposition.base import (
    DomainDecompositionPINN,
    Interface,
    Subdomain,
    uniform_partition,
)
from opifex.neural.pinns.domain_decomposition.cpinn import (
    compute_flux,
    compute_flux_conservation_residual,
    CPINN,
    CPINNConfig,
)
from opifex.neural.pinns.domain_decomposition.fbpinn import (
    CosineWindow,
    create_window_function,
    FBPINN,
    FBPINNConfig,
    GaussianWindow,
    WindowFunction,
)
from opifex.neural.pinns.domain_decomposition.xpinn import XPINN, XPINNConfig


__all__ = [
    "APINN",
    "CPINN",
    "FBPINN",
    "XPINN",
    "APINNConfig",
    "CPINNConfig",
    "CosineWindow",
    "DomainDecompositionPINN",
    "FBPINNConfig",
    "GatingNetwork",
    "GaussianWindow",
    "Interface",
    "Subdomain",
    "WindowFunction",
    "XPINNConfig",
    "compute_flux",
    "compute_flux_conservation_residual",
    "create_window_function",
    "uniform_partition",
]

"""
Quantum mechanical primitives for the Opifex framework.

This module provides fundamental quantum mechanical data structures and
utilities for Neural DFT and other quantum machine learning applications.
"""

from opifex.core.quantum._ri import (
    AuxiliaryBasis,
    fitted_eri,
    three_center_eri,
    two_center_metric,
)
from opifex.core.quantum._spherical import (
    apply_left,
    apply_matrix,
    build_block_transform,
    cart_to_spherical_matrix,
)
from opifex.core.quantum.backend import JaxGaussianBackend, QCBackend
from opifex.core.quantum.basis import AtomicOrbitalBasis, GaussianShell
from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.operators import (
    DensityMatrix,
    HamiltonianOperator,
    KineticEnergyOperator,
    MomentumOperator,
    Observable,
    OperatorComposition,
    PotentialEnergyOperator,
    QuantumOperator,
    SparseOperator,
)
from opifex.core.quantum.protocols import (
    AtomisticModel,
    Backbone,
    NeighborList,
    PropertyHead,
    RadiusNeighborList,
    Space,
)
from opifex.core.quantum.registry import (
    AtomisticModelRegistry,
    BackboneRegistry,
    PropertyHeadRegistry,
    register_atomistic_model,
    register_backbone,
    register_property_head,
)
from opifex.core.quantum.space import free, FreeSpace, periodic, PeriodicSpace


__all__ = [
    "AtomicOrbitalBasis",
    "AtomisticModel",
    "AtomisticModelRegistry",
    "AuxiliaryBasis",
    "Backbone",
    "BackboneRegistry",
    "DensityMatrix",
    "FreeSpace",
    "GaussianShell",
    "HamiltonianOperator",
    "JaxGaussianBackend",
    "KineticEnergyOperator",
    "MolecularSystem",
    "MomentumOperator",
    "NeighborList",
    "Observable",
    "OperatorComposition",
    "PeriodicSpace",
    "PotentialEnergyOperator",
    "PropertyHead",
    "PropertyHeadRegistry",
    "QCBackend",
    "QuantumOperator",
    "RadiusNeighborList",
    "Space",
    "SparseOperator",
    "apply_left",
    "apply_matrix",
    "build_block_transform",
    "cart_to_spherical_matrix",
    "fitted_eri",
    "free",
    "periodic",
    "register_atomistic_model",
    "register_backbone",
    "register_property_head",
    "three_center_eri",
    "two_center_metric",
]

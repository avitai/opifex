"""
Quantum mechanical primitives for the Opifex framework.

This module provides fundamental quantum mechanical data structures and
utilities for Neural DFT and other quantum machine learning applications.
"""

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
    "AtomisticModel",
    "AtomisticModelRegistry",
    "Backbone",
    "BackboneRegistry",
    "DensityMatrix",
    "FreeSpace",
    "HamiltonianOperator",
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
    "QuantumOperator",
    "RadiusNeighborList",
    "Space",
    "SparseOperator",
    "free",
    "periodic",
    "register_atomistic_model",
    "register_backbone",
    "register_property_head",
]

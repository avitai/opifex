"""Native E(3)-equivariant building-block library (irreps, Clebsch-Gordan, etc.).

A dependency-free, JAX/Flax-NNX-native reimplementation of the equivariant
primitives used by ``e3nn``/``e3nn-jax`` (Geiger & Smidt 2022, arXiv:2207.09453),
shared across opifex's quantum-SciML families (interatomic potentials,
equivariant Hamiltonian prediction).
"""

from opifex.neural.equivariant.gate import Gate, gate
from opifex.neural.equivariant.graph import (
    radius_graph,
    scatter_max,
    scatter_mean,
    scatter_sum,
)
from opifex.neural.equivariant.irreps import Irrep, Irreps, IrrepsArray
from opifex.neural.equivariant.linear import EquivariantLinear
from opifex.neural.equivariant.radial import (
    BesselBasis,
    cosine_cutoff,
    GaussianBasis,
    polynomial_cutoff,
)
from opifex.neural.equivariant.spherical_harmonics import spherical_harmonics
from opifex.neural.equivariant.tensor_product import (
    FullyConnectedTensorProduct,
    TensorProduct,
)


__all__ = [
    "BesselBasis",
    "EquivariantLinear",
    "FullyConnectedTensorProduct",
    "Gate",
    "GaussianBasis",
    "Irrep",
    "Irreps",
    "IrrepsArray",
    "TensorProduct",
    "cosine_cutoff",
    "gate",
    "polynomial_cutoff",
    "radius_graph",
    "scatter_max",
    "scatter_mean",
    "scatter_sum",
    "spherical_harmonics",
]

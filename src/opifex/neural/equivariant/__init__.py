"""Native E(3)-equivariant building-block library (irreps, Clebsch-Gordan, etc.).

A dependency-free, JAX/Flax-NNX-native reimplementation of the equivariant
primitives used by ``e3nn``/``e3nn-jax`` (Geiger & Smidt 2022, arXiv:2207.09453),
shared across opifex's quantum-SciML families (interatomic potentials,
equivariant Hamiltonian prediction).
"""

from opifex.neural.equivariant._assembly import apply_scalar_weights
from opifex.neural.equivariant._invariants import inner_product, norm, rms_normalize
from opifex.neural.equivariant.cartesian import (
    CARTESIAN_IRREPS,
    CartesianLinear,
    CartesianTensor,
    from_irreps_array,
    to_irreps_array,
)
from opifex.neural.equivariant.gate import Gate, gate, NormGate
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
    PiecewiseLinearBasis,
    polynomial_cutoff,
)
from opifex.neural.equivariant.spherical_harmonics import spherical_harmonics
from opifex.neural.equivariant.symmetric_contraction import SymmetricContraction
from opifex.neural.equivariant.tensor_product import (
    ChannelwiseTensorProduct,
    FullyConnectedTensorProduct,
    TensorProduct,
)


__all__ = [
    "CARTESIAN_IRREPS",
    "BesselBasis",
    "CartesianLinear",
    "CartesianTensor",
    "ChannelwiseTensorProduct",
    "EquivariantLinear",
    "FullyConnectedTensorProduct",
    "Gate",
    "GaussianBasis",
    "Irrep",
    "Irreps",
    "IrrepsArray",
    "NormGate",
    "PiecewiseLinearBasis",
    "SymmetricContraction",
    "TensorProduct",
    "apply_scalar_weights",
    "cosine_cutoff",
    "from_irreps_array",
    "gate",
    "inner_product",
    "norm",
    "polynomial_cutoff",
    "radius_graph",
    "rms_normalize",
    "scatter_max",
    "scatter_mean",
    "scatter_sum",
    "spherical_harmonics",
    "to_irreps_array",
]

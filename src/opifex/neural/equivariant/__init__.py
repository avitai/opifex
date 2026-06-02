"""Native E(3)-equivariant building-block library (irreps, Clebsch-Gordan, etc.).

A dependency-free, JAX/Flax-NNX-native reimplementation of the equivariant
primitives used by ``e3nn``/``e3nn-jax`` (Geiger & Smidt 2022, arXiv:2207.09453),
shared across opifex's quantum-SciML families (interatomic potentials,
equivariant Hamiltonian prediction).
"""

from opifex.neural.equivariant.irreps import Irrep, Irreps, IrrepsArray
from opifex.neural.equivariant.spherical_harmonics import spherical_harmonics


__all__ = ["Irrep", "Irreps", "IrrepsArray", "spherical_harmonics"]

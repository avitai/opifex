"""
Physics Spectral Methods Module

This module provides spectral methods for quantum mechanical calculations,
including momentum and kinetic energy operators using Fourier transforms.
Part of the consolidated spectral framework for Opifex.

Key Components:
- quantum_spectral: Quantum mechanical spectral operators
- hamiltonian_spectral: Hamiltonian operator spectral methods
"""

from opifex.physics.spectral.quantum_spectral import (
    spectral_gradient,
    spectral_kinetic_energy,
    spectral_momentum,
    spectral_second_derivative,
)


__all__ = [
    "spectral_gradient",
    "spectral_kinetic_energy",
    "spectral_momentum",
    "spectral_second_derivative",
]

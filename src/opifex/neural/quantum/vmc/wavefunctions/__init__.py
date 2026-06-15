"""Neural-network wavefunction ansaetze for variational Monte Carlo."""

from opifex.neural.quantum.vmc.wavefunctions._blocks import (
    construct_input_features,
    logdet_matmul,
    slogdet,
)
from opifex.neural.quantum.vmc.wavefunctions.ferminet import FermiNet


__all__ = [
    "FermiNet",
    "construct_input_features",
    "logdet_matmul",
    "slogdet",
]

"""Variational Monte Carlo with neural-network wavefunctions.

This package implements a scalable, ``jit``/``grad``/``vmap``-clean VMC stack
built entirely on JAX, Flax-NNX and optax (no external QMC dependency):

* :mod:`~opifex.neural.quantum.vmc.wavefunctions` -- the FermiNet-core
  generalized-Slater ansatz (:class:`~.wavefunctions.FermiNet`) and the
  :class:`~.wavefunctions.PsiFormer` self-attention ansatz;
* :mod:`~opifex.neural.quantum.vmc.laplacian` -- the kinetic-energy Laplacian,
  with a reference ``jvp``-over-``grad`` oracle and a native forward-Laplacian;
* :mod:`~opifex.neural.quantum.vmc.hamiltonian` -- the molecular local energy;
* :mod:`~opifex.neural.quantum.vmc.sampler` -- harmonic-mean Metropolis-Hastings;
* :mod:`~opifex.neural.quantum.vmc.mala` -- Metropolis-adjusted Langevin (MALA);
* :mod:`~opifex.neural.quantum.vmc.optimizers` -- Adam, MinSR and SPRING updates;
* :mod:`~opifex.neural.quantum.vmc.training` -- the VMC energy-minimisation loop.
"""

from opifex.neural.quantum.vmc.hamiltonian import local_energy
from opifex.neural.quantum.vmc.laplacian import (
    forward_laplacian,
    jvp_grad_laplacian,
)
from opifex.neural.quantum.vmc.mala import langevin_drift, MALASampler
from opifex.neural.quantum.vmc.optimizers import (
    minsr_update,
    spring_update,
    SpringState,
)
from opifex.neural.quantum.vmc.protocols import Sampler, Wavefunction
from opifex.neural.quantum.vmc.sampler import MetropolisHastingsSampler
from opifex.neural.quantum.vmc.training import (
    VMCConfig,
    VMCDriver,
    VMCResult,
)
from opifex.neural.quantum.vmc.wavefunctions import FermiNet, PsiFormer


__all__ = [
    "FermiNet",
    "MALASampler",
    "MetropolisHastingsSampler",
    "PsiFormer",
    "Sampler",
    "SpringState",
    "VMCConfig",
    "VMCDriver",
    "VMCResult",
    "Wavefunction",
    "forward_laplacian",
    "jvp_grad_laplacian",
    "langevin_drift",
    "local_energy",
    "minsr_update",
    "spring_update",
]

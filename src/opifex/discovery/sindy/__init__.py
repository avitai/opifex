"""SINDy: Sparse Identification of Nonlinear Dynamics.

JAX-native implementation of the SINDy algorithm family for discovering
governing equations from time-series data.

Reference:
    Brunton et al. (2016) "Discovering governing equations from data
    by sparse identification of nonlinear dynamical systems"
"""

from opifex.discovery.sindy._uq_capabilities import SINDY_CAPABILITIES
from opifex.discovery.sindy.config import EnsembleSINDyConfig, SINDyConfig, WeakSINDyConfig
from opifex.discovery.sindy.ensemble_sindy import EnsembleSINDy
from opifex.discovery.sindy.library import CandidateLibrary
from opifex.discovery.sindy.optimizers import SR3, STLSQ
from opifex.discovery.sindy.sindy import SINDy
from opifex.discovery.sindy.ude_distillation import distill_ude_residual
from opifex.discovery.sindy.utils import finite_difference, smooth_data
from opifex.discovery.sindy.weak_sindy import WeakSINDy
from opifex.uncertainty.registry import UQRegistry


# UQ capability registration — Task 7.5. Guarded against duplicate
# registration on repeat imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in SINDY_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = [
    "SINDY_CAPABILITIES",
    "SR3",
    "STLSQ",
    "CandidateLibrary",
    "EnsembleSINDy",
    "EnsembleSINDyConfig",
    "SINDy",
    "SINDyConfig",
    "WeakSINDy",
    "WeakSINDyConfig",
    "distill_ude_residual",
    "finite_difference",
    "smooth_data",
]

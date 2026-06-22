"""Uncertainty-guided training utilities for Opifex.

The unified training loop is :class:`opifex.core.training.Trainer`. This package hosts the
uncertainty-guided sample-selection helpers (``uncertainty_trainers``) and registers their UQ
capabilities.
"""

from opifex.training._uq_capabilities import TRAINING_CAPABILITIES
from opifex.uncertainty.registry import UQRegistry


# UQ capability registration — Task 7.5. Guarded against duplicate
# registration on repeat imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in TRAINING_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = ["TRAINING_CAPABILITIES"]

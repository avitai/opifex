"""Multilevel training framework.

This module implements coarse-to-fine multilevel training strategies
where models are trained from coarse (fewer parameters) to fine (more parameters).

Key Components:
    - CascadeTrainer: Sequential level training
    - MultilevelMLP: MLP with width-based hierarchy
    - MultilevelFNOTrainer: FNO with mode-based hierarchy
    - SimpleFNO: Simple FNO for multilevel training
"""

from .cascade_training import CascadeTrainer
from .coarse_to_fine import (
    CascadeTrainer as LegacyCascadeTrainer,
    create_network_hierarchy,
    MultilevelConfig,
    MultilevelMLP,
    prolongate,
    restrict,
)
from .multilevel_adam import MultilevelAdam
from .multilevel_fno import (
    create_fno_hierarchy,
    create_mode_hierarchy,
    MultilevelFNOConfig,
    MultilevelFNOTrainer,
    prolongate_fno_modes,
    restrict_fno_modes,
    SimpleFNO,
    SpectralConv1d,
)


__all__ = [
    "CascadeTrainer",
    "LegacyCascadeTrainer",
    "MultilevelAdam",
    "MultilevelConfig",
    "MultilevelFNOConfig",
    "MultilevelFNOTrainer",
    "MultilevelMLP",
    "SimpleFNO",
    "SpectralConv1d",
    "create_fno_hierarchy",
    "create_mode_hierarchy",
    "create_network_hierarchy",
    "prolongate",
    "prolongate_fno_modes",
    "restrict",
    "restrict_fno_modes",
]

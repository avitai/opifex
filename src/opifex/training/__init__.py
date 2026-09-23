"""Uncertainty-guided training utilities for Opifex.

The unified training loop is :class:`opifex.core.training.Trainer`. This package hosts the
uncertainty-guided sample-selection helpers (``uncertainty_trainers``) and registers their UQ
capabilities.
"""

from opifex.training._uq_capabilities import TRAINING_CAPABILITIES


__all__ = ["TRAINING_CAPABILITIES"]

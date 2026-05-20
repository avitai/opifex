"""Uncertainty-aware loss functions used by opifex UQ surfaces."""

from __future__ import annotations

from opifex.uncertainty.losses.pointwise_quantile import PointwiseQuantileLoss


__all__ = ["PointwiseQuantileLoss"]

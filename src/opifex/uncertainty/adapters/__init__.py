"""Distribution / model-uncertainty adapter protocols + concrete adapters."""

from __future__ import annotations

from opifex.uncertainty.adapters.base import (
    DistributionAdapterProtocol,
    DistributionAdapterSpec,
    ModelUncertaintyAdapterProtocol,
)
from opifex.uncertainty.adapters.ensemble import (
    BatchEnsembleAdapterSpec,
    BatchEnsembleState,
    DeepEnsembleAdapter,
    DeepEnsembleState,
    DUEAdapterSpec,
    SnapshotEnsembleAdapterSpec,
    SnapshotEnsembleState,
    SWAGAdapterSpec,
    SWAGState,
    TestTimeAugmentationAdapterSpec,
)
from opifex.uncertainty.adapters.model import (
    BayesianLastLayerAdapterSpec,
    LaplaceAdapterSpec,
    MCDropoutAdapter,
    MCDropoutState,
    ModelUncertaintyAdapter,
    SNGPAdapterSpec,
    VBLLAdapterSpec,
)


__all__ = [
    "BatchEnsembleAdapterSpec",
    "BatchEnsembleState",
    "BayesianLastLayerAdapterSpec",
    "DUEAdapterSpec",
    "DeepEnsembleAdapter",
    "DeepEnsembleState",
    "DistributionAdapterProtocol",
    "DistributionAdapterSpec",
    "LaplaceAdapterSpec",
    "MCDropoutAdapter",
    "MCDropoutState",
    "ModelUncertaintyAdapter",
    "ModelUncertaintyAdapterProtocol",
    "SNGPAdapterSpec",
    "SWAGAdapterSpec",
    "SWAGState",
    "SnapshotEnsembleAdapterSpec",
    "SnapshotEnsembleState",
    "TestTimeAugmentationAdapterSpec",
    "VBLLAdapterSpec",
]

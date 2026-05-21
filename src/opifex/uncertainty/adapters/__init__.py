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
    LaplaceState,
    MCDropoutAdapter,
    MCDropoutState,
    ModelUncertaintyAdapter,
    SNGPAdapterSpec,
    VBLLAdapterSpec,
)
from opifex.uncertainty.adapters.operators import (
    DeepONetConformalAdapterSpec,
    DeepONetDeepEnsembleAdapterSpec,
    DeepONetMCDropoutAdapterSpec,
    FNOConformalAdapterSpec,
    FNODeepEnsembleAdapterSpec,
    FNOMCDropoutAdapterSpec,
    OperatorAdapterSpec,
)


__all__ = [
    "BatchEnsembleAdapterSpec",
    "BatchEnsembleState",
    "BayesianLastLayerAdapterSpec",
    "DUEAdapterSpec",
    "DeepEnsembleAdapter",
    "DeepEnsembleState",
    "DeepONetConformalAdapterSpec",
    "DeepONetDeepEnsembleAdapterSpec",
    "DeepONetMCDropoutAdapterSpec",
    "DistributionAdapterProtocol",
    "DistributionAdapterSpec",
    "FNOConformalAdapterSpec",
    "FNODeepEnsembleAdapterSpec",
    "FNOMCDropoutAdapterSpec",
    "LaplaceAdapterSpec",
    "LaplaceState",
    "MCDropoutAdapter",
    "MCDropoutState",
    "ModelUncertaintyAdapter",
    "ModelUncertaintyAdapterProtocol",
    "OperatorAdapterSpec",
    "SNGPAdapterSpec",
    "SWAGAdapterSpec",
    "SWAGState",
    "SnapshotEnsembleAdapterSpec",
    "SnapshotEnsembleState",
    "TestTimeAugmentationAdapterSpec",
    "VBLLAdapterSpec",
]

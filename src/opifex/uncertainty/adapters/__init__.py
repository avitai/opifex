"""Distribution / model-uncertainty adapter protocols + concrete adapters."""

from __future__ import annotations

from opifex.uncertainty.adapters._uq_capabilities import ADAPTER_CAPABILITIES
from opifex.uncertainty.adapters.base import (
    DistributionAdapterProtocol,
    DistributionAdapterSpec,
    ModelUncertaintyAdapterProtocol,
)
from opifex.uncertainty.adapters.ensemble import (
    BatchEnsembleAdapter,
    BatchEnsembleState,
    DeepEnsembleAdapter,
    DeepEnsembleState,
    SnapshotEnsembleAdapter,
    SnapshotEnsembleState,
    SWAGAdapter,
    SWAGState,
    TestTimeAugmentationAdapter,
    TestTimeAugmentationState,
)
from opifex.uncertainty.adapters.gp import (
    BayesnewtonAdapterSpec,
    GPJaxAdapterSpec,
    KalmanJaxAdapterSpec,
    MarkovflowAdapterSpec,
    TinygpAdapterSpec,
)
from opifex.uncertainty.adapters.model import (
    BayesianLastLayerAdapter,
    BayesianLastLayerState,
    DUEAdapter,
    DUEState,
    MCDropoutAdapter,
    MCDropoutState,
    ModelUncertaintyAdapter,
    SNGPAdapter,
    SNGPState,
    VBLLAdapter,
    VBLLState,
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
from opifex.uncertainty.registry import UQRegistry


# UQ capability registration — Task 7.2. Singleton :class:`UQRegistry`
# guarded against duplicate registration on repeat imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in ADAPTER_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = [
    "ADAPTER_CAPABILITIES",
    "BatchEnsembleAdapter",
    "BatchEnsembleState",
    "BayesianLastLayerAdapter",
    "BayesianLastLayerState",
    "BayesnewtonAdapterSpec",
    "DUEAdapter",
    "DUEState",
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
    "GPJaxAdapterSpec",
    "KalmanJaxAdapterSpec",
    "MCDropoutAdapter",
    "MCDropoutState",
    "MarkovflowAdapterSpec",
    "ModelUncertaintyAdapter",
    "ModelUncertaintyAdapterProtocol",
    "OperatorAdapterSpec",
    "SNGPAdapter",
    "SNGPState",
    "SWAGAdapter",
    "SWAGState",
    "SnapshotEnsembleAdapter",
    "SnapshotEnsembleState",
    "TestTimeAugmentationAdapter",
    "TestTimeAugmentationState",
    "TinygpAdapterSpec",
    "VBLLAdapter",
    "VBLLState",
]

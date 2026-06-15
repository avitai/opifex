"""UQ monitoring utilities: typed inputs + reliability-report builder."""

from __future__ import annotations

from opifex.uncertainty.monitoring._uq_capabilities import MONITORING_CAPABILITIES
from opifex.uncertainty.monitoring.reliability_report import (
    build_reliability_report,
    MonitoringInputs,
)
from opifex.uncertainty.registry import UQRegistry


# UQ capability registration — Task 7.5. Guarded against duplicate
# registration on repeat imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in MONITORING_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = ["MONITORING_CAPABILITIES", "MonitoringInputs", "build_reliability_report"]

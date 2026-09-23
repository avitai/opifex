"""Every uncertainty-quantification capability opifex declares, in one read-only catalogue.

Each package keeps the declarations for what it offers beside its own code, as a constant
table of :class:`~opifex.core.capabilities.UQCapability` records. This module collects the
tables from the packages' public namespaces into :data:`UQ_CAPABILITIES`, once, when it is
first imported. Nothing registers at import time and no global state changes, so importing
any package has no side effect; a name two tables both declare is refused while the
catalogue is built. The module sits above every package it collects, so none of them
imports it.
"""

from collections.abc import Mapping
from types import MappingProxyType

from opifex.core.capabilities import UQCapability
from opifex.discovery import DISCOVERY_CAPABILITIES
from opifex.discovery.sindy import SINDY_CAPABILITIES
from opifex.mlops import MLOPS_CAPABILITIES
from opifex.neural.bayesian import BAYESIAN_MODEL_CAPABILITIES
from opifex.neural.operators import OPERATOR_CAPABILITIES
from opifex.neural.quantum import QUANTUM_CAPABILITIES
from opifex.optimization.l2o import L2O_CAPABILITIES
from opifex.solvers import SOLVER_CAPABILITIES
from opifex.training import TRAINING_CAPABILITIES
from opifex.uncertainty.active import ACTIVE_CAPABILITIES
from opifex.uncertainty.adapters import ADAPTER_CAPABILITIES
from opifex.uncertainty.assimilation import ASSIMILATION_CAPABILITIES
from opifex.uncertainty.curvature import CURVATURE_CAPABILITIES
from opifex.uncertainty.inference_backends import INFERENCE_BACKEND_CAPABILITIES
from opifex.uncertainty.linalg import LINALG_CAPABILITIES
from opifex.uncertainty.monitoring import MONITORING_CAPABILITIES
from opifex.uncertainty.pac_bayes import PAC_BAYES_CAPABILITIES
from opifex.uncertainty.quadrature import QUADRATURE_CAPABILITIES
from opifex.uncertainty.sbi import SBI_CAPABILITIES
from opifex.uncertainty.scientific import SCIENTIFIC_FIELD_CAPABILITIES
from opifex.uncertainty.statespace import STATESPACE_CAPABILITIES


def assemble(*tables: Mapping[str, UQCapability]) -> Mapping[str, UQCapability]:
    """Merge capability tables into one read-only mapping.

    Args:
        *tables: Tables of declarations, keyed by capability name.

    Returns:
        The merged declarations, as a read-only mapping.

    Raises:
        ValueError: If two tables declare the same name; a declaration is canonical, so a
            second one is a conflict, not an override.
    """
    merged: dict[str, UQCapability] = {}
    for table in tables:
        for name, capability in table.items():
            if name in merged:
                msg = f"capability {name!r} is declared twice"
                raise ValueError(msg)
            merged[name] = capability
    return MappingProxyType(merged)


UQ_CAPABILITIES: Mapping[str, UQCapability] = assemble(
    DISCOVERY_CAPABILITIES,
    SINDY_CAPABILITIES,
    MLOPS_CAPABILITIES,
    BAYESIAN_MODEL_CAPABILITIES,
    OPERATOR_CAPABILITIES,
    QUANTUM_CAPABILITIES,
    L2O_CAPABILITIES,
    SOLVER_CAPABILITIES,
    TRAINING_CAPABILITIES,
    ACTIVE_CAPABILITIES,
    ADAPTER_CAPABILITIES,
    ASSIMILATION_CAPABILITIES,
    CURVATURE_CAPABILITIES,
    INFERENCE_BACKEND_CAPABILITIES,
    LINALG_CAPABILITIES,
    MONITORING_CAPABILITIES,
    PAC_BAYES_CAPABILITIES,
    QUADRATURE_CAPABILITIES,
    SBI_CAPABILITIES,
    SCIENTIFIC_FIELD_CAPABILITIES,
    STATESPACE_CAPABILITIES,
)
"""Every capability opifex declares, keyed by name, read-only."""


def require_uq_capability(name: str) -> UQCapability:
    """Return the capability declared under ``name``.

    Args:
        name: A capability name, such as ``"model:ProbabilisticPINN"``.

    Returns:
        Its declaration.

    Raises:
        KeyError: If nothing is declared under ``name``; the message lists what is.
    """
    if name not in UQ_CAPABILITIES:
        msg = f"capability {name!r} is not declared; declared: {sorted(UQ_CAPABILITIES)!r}"
        raise KeyError(msg)
    return UQ_CAPABILITIES[name]


__all__ = ["UQ_CAPABILITIES", "assemble", "require_uq_capability"]
